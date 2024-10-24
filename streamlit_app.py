# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st
import torch
from funcyou.utils import DotDict
from PIL import Image
from torchvision import transforms

from model import VisionTransformer  # Import your model class


def load_and_preprocess_image(image, transform):
    image = Image.open(image).convert("RGB")
    image = transform(image)
    return image

def predict_image(image, model, transform, device='cpu'):
    model.eval()
    image = image.to(device)
    outputs, attention_weights = model(image.unsqueeze(0))
    prediction = torch.sigmoid(outputs).cpu().detach().numpy()
    attention_weights = [aw.squeeze().cpu().detach().numpy() for aw in attention_weights]  # Convert attention weights to NumPy arrays
    return prediction, attention_weights

def main():
    st.title("Pneumonia Detection using Vision Transformer")
        # Introduction and Project Information
    st.write("This is a Streamlit app for performing Pneumonia Detection on Xrays.")


    # Upload image
    uploaded_image = st.file_uploader("Upload an Lung X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        config = DotDict.from_toml('config.toml')
        # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        preprocessed_image = load_and_preprocess_image(uploaded_image, transform)
        prediction, attention_weights = predict_image(preprocessed_image, model, transform, config.device)

        # Display prediction
        st.write("Prediction")
        
        confidance = float(abs(0.5 - prediction[0].item())) * 2
        # st.write(prediction[0])
        if np.round(prediction[0])==1:
            st.error(f"Pneumonia Positive.  confidance({confidance:.2%})")
            
        else:
            st.success('Normal. Pneumonia not detected. confidance({confidance:.2%})')
            
            


    # Project Usage and Links
    st.sidebar.write("## Project Usage")
    st.sidebar.write("This project performs Pneumonia detection on X-ray image and return result as positive or negative.")
    st.info('Note: The Default model has been trained on a imbalance dataset containing 27% Normal and 73% Pneumonia xrays. We used weighted sampling Technique to overcome the Imbalance. Train the Model on a fair dataset and try again.')
    st.sidebar.write("## GitHub Repository")
    st.sidebar.write("Source Code here [GitHub repository](https://github.com/tikendraw/pneumonia-detection-with-vision-transformer).")
    st.sidebar.write("If you have any feedback or suggestions, feel free to open an issue or a pull request.")
    st.sidebar.write("## Like the Project?")
    st.sidebar.write("If you find this project interesting or useful, don't forget to give it a star on GitHub!")
    st.sidebar.markdown('![GitHub Repo stars](https://img.shields.io/github/stars/tikendraw/pneumonia-detection-with-vision-transformer?style=flat&logo=github&logoColor=white&label=Github%20Stars)', unsafe_allow_html=True)


        


if __name__ == "__main__":
    main()
