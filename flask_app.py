import os
from flask import Flask, render_template, request, redirect, url_for
import torch
from funcyou.utils import DotDict
from PIL import Image
from torchvision import transforms
import numpy as np
from model import VisionTransformer  # Import your model class

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

def predict_image(image, model, device='cpu'):
    model.eval()
    image = image.to(device)
    outputs, attention_weights = model(image.unsqueeze(0))
    prediction = torch.sigmoid(outputs).cpu().detach().numpy()
    attention_weights = [aw.squeeze().cpu().detach().numpy() for aw in attention_weights]
    return prediction, attention_weights

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            config = DotDict.from_toml('config.toml')

            if torch.backends.mps.is_available():
                config.device = 'mps'
            elif torch.cuda.is_available():
                config.device = 'cuda'
            else:
                config.device = 'cpu'

            model = VisionTransformer(config)
            model.load_state_dict(torch.load(config.model_path))
            model.to(config.device)
            model.eval()

            transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(), 
            ])

            preprocessed_image = load_and_preprocess_image(file_path, transform)
            prediction, attention_weights = predict_image(preprocessed_image, model, config.device)

            confidance = float(abs(0.5 - prediction[0].item())) * 2
            diagnosis = "Pneumonia Positive" if np.round(prediction[0]) == 1 else "Normal"
            confidence_str = f"Confidence: {confidance:.2%}"

            return render_template('result.html', diagnosis=diagnosis, confidence=confidence_str, image_path=file_path)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
