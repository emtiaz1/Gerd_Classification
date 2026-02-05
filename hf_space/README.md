---
title: GERD & Polyp Endoscopy Classification
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
header: mini
---

# 🔬 GERD & Polyp Endoscopy Classification with LIME Explanations

A Lightweight Vision Transformer (ViT) model for classifying GERD and Polyp conditions from endoscopy images.

## Features

- **Image Classification**: Upload endoscopy images to get GERD and Polyp predictions
- **LIME Explanations**: Visual explanations showing which image regions influenced the prediction
- **4 Classes**: GERD, GERD NORMAL, POLYP, POLYP NORMAL
- **API Access**: RESTful API endpoint for programmatic access

## Model Architecture

- **Type**: Lightweight Vision Transformer (ViT)
- **Input Size**: 224×224×3
- **Patch Size**: 8×8
- **Transformer Layers**: 4
- **Attention Heads**: 4
- **Projection Dimension**: 64

## API Usage

### Using Python requests

```python
import requests
from PIL import Image
import io

# Your HF Space URL
API_URL = "https://YOUR-USERNAME-gerd-classifier.hf.space/api/predict"

# Load and send image
with open("endoscopy_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

print(response.json())
```

### Using Gradio Client

```python
from gradio_client import Client

client = Client("YOUR-USERNAME/gerd-classifier")
result = client.predict(
    image="path/to/image.jpg",
    api_name="/predict"
)
print(result)
```

### Using JavaScript (Browser)

```javascript
async function classifyImage(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(
    "https://YOUR-USERNAME-gerd-classifier.hf.space/api/predict",
    {
      method: "POST",
      body: formData,
    },
  );

  return await response.json();
}
```

## LIME Explanation

LIME (Local Interpretable Model-agnostic Explanations) provides visual insights:

- 🟢 **Green regions**: Areas supporting the prediction
- 🔴 **Red regions**: Areas opposing the prediction

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{gerd_classifier_2024,
  title={GERD Endoscopy Classification using Lightweight ViT},
  year={2024}
}
```

## License

MIT License
