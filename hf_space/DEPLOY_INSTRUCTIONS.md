# 🚀 Deploying GERD & Polyp Classifier to Hugging Face Spaces

Follow these steps to deploy your GERD and Polyp classification model to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co/join
2. **Hugging Face CLI**: Install with `pip install huggingface_hub`

## Step-by-Step Deployment

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted. Get your token from: https://huggingface.co/settings/tokens

### 2. Create a New Space

Go to https://huggingface.co/new-space and:

- **Space name**: `gerd-classifier` (or your preferred name)
- **SDK**: Select `Gradio`
- **Hardware**: Choose `CPU Basic` (free tier) or upgrade if needed
- **Visibility**: Public or Private

### 3. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR-USERNAME/gerd-classifier
cd gerd-classifier
```

### 4. Copy Files

Copy the contents of the `hf_space` folder and your model:

```bash
# Copy app files
cp /path/to/hf_space/app.py .
cp /path/to/hf_space/requirements.txt .
cp /path/to/hf_space/README.md .

# Copy the model file (IMPORTANT!)
cp /path/to/best_fold_model.h5 .
```

### 5. Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment with GERD classifier and LIME"
git push
```

## Alternative: Direct Upload via Python

```python
from huggingface_hub import HfApi, upload_folder

api = HfApi()

# Upload entire folder
upload_folder(
    folder_path="/path/to/hf_space",
    repo_id="YOUR-USERNAME/gerd-classifier",
    repo_type="space"
)

# Upload model separately
api.upload_file(
    path_or_fileobj="/path/to/best_fold_model.h5",
    path_in_repo="best_fold_model.h5",
    repo_id="YOUR-USERNAME/gerd-classifier",
    repo_type="space"
)
```

## CORS Configuration

Gradio automatically handles CORS for Hugging Face Spaces. Your API will be accessible from any domain.

### Testing CORS from Browser

```javascript
// Test from browser console
fetch("https://YOUR-USERNAME-gerd-classifier.hf.space/api/predict", {
  method: "POST",
  body: formData, // FormData with image
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

## API Endpoints

Once deployed, your Space will expose these endpoints:

| Endpoint                   | Method | Description       |
| -------------------------- | ------ | ----------------- |
| `/`                        | GET    | Web interface     |
| `/api/predict`             | POST   | Prediction API    |
| `/api/predict_and_explain` | POST   | Prediction + LIME |

### Example API Call

```python
import requests
import base64

# Using Gradio Client (recommended)
from gradio_client import Client

client = Client("YOUR-USERNAME/gerd-classifier")

# For prediction + LIME explanation
result = client.predict(
    image="path/to/image.jpg",
    api_name="/predict_and_explain"
)

predictions, lime_image = result
print(predictions)
```

### Raw HTTP API Call

```python
import requests

# Get the API info first
info_url = "https://YOUR-USERNAME-gerd-classifier.hf.space/info"
response = requests.get(info_url)

# Make prediction
api_url = "https://YOUR-USERNAME-gerd-classifier.hf.space/run/predict"

import base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

payload = {
    "data": [f"data:image/jpeg;base64,{image_base64}"]
}

response = requests.post(api_url, json=payload)
print(response.json())
```

## Troubleshooting

### Model Not Loading

- Ensure `best_fold_model.h5` is in the root of your Space
- Check if the file was uploaded correctly in the HF Files tab

### CORS Issues

- Gradio handles CORS automatically
- If using custom domains, ensure proper headers

### Memory Errors

- Upgrade to a higher-tier Space (GPU or more RAM)
- Reduce `num_samples` in LIME explanation

## File Structure

```
gerd-classifier/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── README.md           # Space documentation
└── best_fold_model.h5  # Trained model weights
```

## Updating the Model

To update your model:

```bash
# Remove old model
git rm best_fold_model.h5

# Add new model (use Git LFS for large files)
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add best_fold_model.h5
git commit -m "Update model"
git push
```

---

🎉 Your GERD classifier should now be live at:
`https://huggingface.co/spaces/YOUR-USERNAME/gerd-classifier`
