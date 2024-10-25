from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from model import VisionTransformer  # Import your model class
from funcyou.utils import DotDict

app = Flask(__name__)

def load_and_preprocess_image(image, transform):
    image = Image.open(image).convert("RGB")
    image = transform(image)
    return image

def predict_image(image, model, device='cpu'):
    model.eval()
    image = image.to(device)
    outputs, attention_weights = model(image.unsqueeze(0))
    prediction = torch.sigmoid(outputs).cpu().detach().numpy()
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    image = Image.open(file)

    # Preprocess and predict
    config = DotDict.from_toml('config.toml')
    # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = 'cpu'

    model = VisionTransformer(config)
    model.load_state_dict(torch.load(config.model_path))
    model.to(config.device)

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), 
    ])

    preprocessed_image = load_and_preprocess_image(file, transform)
    prediction = predict_image(preprocessed_image, model, config.device)

    confidance = float(abs(0.5 - prediction[0].item())) * 2

    result = {
        'prediction': int(np.round(prediction[0])),
        'confidence': f"{confidance:.2%}"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
