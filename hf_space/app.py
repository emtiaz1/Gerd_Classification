"""
GERD Classification with LIME Explanations
Hugging Face Spaces Application
"""

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, models
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import io
import os

# ============================================
# Configuration
# ============================================
IMAGE_SIZE = 224
PATCH_SIZE = 8
PROJECTION_DIM = 64
TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
MLP_HEAD_UNITS = [128, 64]


CLASS_NAMES = ['GERD', 'GERD NORMAL', 'POLYP', 'POLYP NORMAL']

# ============================================
# Model Architecture (same as training)
# ============================================


class Patches(L.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(L.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = L.Dense(units=projection_dim)
        self.position_embedding = L.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = L.Dense(units, activation='gelu')(x)
        x = L.Dropout(dropout_rate)(x)
    return x


def create_lightweight_vit(n_classes):
    inputs = L.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

    # Patches + Encoding
    patches = Patches(PATCH_SIZE)(inputs)
    encoded_patches = PatchEncoder(num_patches, PROJECTION_DIM)(patches)

    # Transformer blocks
    x = encoded_patches
    for _ in range(TRANSFORMER_LAYERS):
        x1 = L.LayerNormalization(epsilon=1e-6)(x)
        attention_output = L.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM // NUM_HEADS,
            dropout=0.1
        )(x1, x1)
        x2 = L.Add()([attention_output, x])
        x3 = L.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[PROJECTION_DIM,
                 PROJECTION_DIM], dropout_rate=0.1)
        x = L.Add()([x3, x2])

    # Classification head
    representation = L.LayerNormalization(epsilon=1e-6)(x)
    representation = L.GlobalAveragePooling1D()(representation)
    representation = L.Dropout(0.3)(representation)
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS,
                   dropout_rate=0.3)
    logits = L.Dense(n_classes, activation='softmax')(features)

    model = models.Model(inputs=inputs, outputs=logits)
    return model


# ============================================
# Load Model
# ============================================
def load_model():
    """Load the trained model - rebuild architecture and load weights"""
    model_path = "best_fold_model.h5"

    # Always rebuild the model architecture first
    model = create_lightweight_vit(len(CLASS_NAMES))

    if os.path.exists(model_path):
        try:
            # Try loading the full model first
            custom_objects = {
                'Patches': Patches,
                'PatchEncoder': PatchEncoder
            }
            model = tf.keras.models.load_model(
                model_path, custom_objects=custom_objects, compile=False)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Could not load full model: {e}")
            print("Attempting to load weights only...")
            try:
                # Load weights into rebuilt architecture
                model.load_weights(model_path)
                print(f"Weights loaded from {model_path}")
            except Exception as e2:
                print(f"Could not load weights: {e2}")
                print("Using untrained model architecture")
    else:
        print("Model file not found. Using untrained model architecture...")

    return model


# Initialize model globally
model = load_model()


# ============================================
# Image Preprocessing
# ============================================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    img = img.convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    return img_array


# ============================================
# Prediction Function
# ============================================
def predict(image):
    """Make prediction on a single image"""
    img_array = preprocess_image(image)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]

    result = {CLASS_NAMES[i]: float(predictions[i])
              for i in range(len(CLASS_NAMES))}
    return result


# ============================================
# LIME Explanation
# ============================================
def generate_lime_explanation(image):
    """Generate LIME explanation for the image"""
    img_array = preprocess_image(image)

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Define prediction function for LIME
    def predict_fn(images):
        return model.predict(images, verbose=0)

    # Generate explanation
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=len(CLASS_NAMES),
        hide_color=0,
        num_samples=500,  # Reduced for faster inference
        batch_size=32
    )

    # Get prediction info
    predictions = model.predict(
        np.expand_dims(img_array, axis=0), verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    predicted_prob = predictions[predicted_class_idx]

    # Create visualization figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # LIME explanation - positive features only
    temp, mask = explanation.get_image_and_mask(
        predicted_class_idx,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(f'LIME: {predicted_class}\n(Confidence: {predicted_prob:.2%})',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # LIME explanation - positive and negative features
    temp, mask = explanation.get_image_and_mask(
        predicted_class_idx,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    axes[2].imshow(mark_boundaries(temp, mask))
    axes[2].set_title('Positive (Green) & Negative (Red)\nContributions',
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Convert figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close(fig)

    return result_image


# ============================================
# Combined Prediction and LIME
# ============================================
def predict_and_explain(image):
    """Combined function for prediction and LIME explanation"""
    if image is None:
        return None, None

    # Get predictions
    predictions = predict(image)

    # Generate LIME explanation
    lime_img = generate_lime_explanation(image)

    return predictions, lime_img


# ============================================
# Gradio Interface (3.x API)
# ============================================
title = "🔬 GERD & Polyp Endoscopy Classification"
description = """
Upload an endoscopy image to classify GERD and Polyp conditions with LIME explanations.

**Classes:** GERD | GERD NORMAL | POLYP | POLYP NORMAL

### About LIME Explanations
- 🟢 **Green boundaries**: Regions that **support** the prediction
- 🔴 **Red boundaries**: Regions that **oppose** the prediction
"""

article = """
### API Usage

```python
from gradio_client import Client

client = Client("nahid112376/gerd-classifier")
result = client.predict(image="path/to/image.jpg", api_name="/predict")
print(result)
```

---
*Powered by Lightweight Vision Transformer (ViT) | Built with 🤗 Gradio*
"""

# Create interface
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="numpy", label="Upload Endoscopy Image"),
    outputs=[
        gr.Label(num_top_classes=4, label="Classification Results"),
        gr.Image(type="pil", label="LIME Explanation")
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging="never"
)

# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
