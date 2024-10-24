import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from funcyou.utils import DotDict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data import load_image_data
from model import VisionTransformer
from utils import predict_model, set_seed

# Configure logger
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Training function
def train_model(
    model,
    dataloader,
    optimizer,
    loss_function,
    num_epochs=10,
    device="cpu",
    data_percent=1.0,
    steps_per_epoch=None,
    save_on_every_n_epochs=5,
    model_path=None,
):
    model.to(device)
    print(f"{model.__class__.__name__} Running on: {device}")

    data_size = int(data_percent * len(dataloader)) if steps_per_epoch is None else steps_per_epoch

    Path(model_path).parent.mkdir(exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        epoch_progress = tqdm(
            dataloader, desc=f"Epoch [{epoch + 1:2}/{num_epochs:2}]"
        )
        
        last_update_time = time.time() - 1.0  

        for j, batch in enumerate(epoch_progress):
            image, label = batch
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            outputs, _ = model(image)
            outputs = outputs.squeeze()
            predictions = torch.round(outputs)

            loss = loss_function(outputs, label.to(outputs.dtype))
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            total_correct_predictions += (predictions.to(torch.int32) == label.to(torch.int32)).sum().item()
            total_samples += label.size()[0]

            formatted_loss = f"{loss.item():.8f}"
            accuracy = (total_correct_predictions / total_samples) * 100
            formatted_accuracy = f"{accuracy:.2f}%"
            
            
            current_time = time.time()
            if current_time - last_update_time > epoch_progress.mininterval:
                epoch_progress.set_postfix(
                    {"Loss": formatted_loss, "Accuracy": formatted_accuracy}
                )
                last_update_time = current_time

            if steps_per_epoch is not None and j + 1 >= steps_per_epoch:
                break
        
        average_loss = total_loss / data_size
        average_accuracy = (total_correct_predictions / (total_samples + 1e-7)) * 100

        print(
            f"\nEpoch [{epoch + 1:2}/{num_epochs:2}] - Average Loss: {average_loss:.8f} - Average Accuracy: {average_accuracy:.2f}%"
        )
        logger.info(
            f"Epoch [{epoch + 1:2}/{num_epochs:2}] - Average Loss: {average_loss:.8f} - Average Accuracy: {average_accuracy:.2f}%"
        )

        if (epoch+1) % save_on_every_n_epochs == 0 and model_path is not None:
            torch.save(model.state_dict(), model_path)




# Main training function
def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Training and Testing')
    parser.add_argument('image_directory', type=str, help='Path to the directory containing training images')
    parser.add_argument('--test', action='store_true', default=False, help='Flag to test a directory')
    parser.add_argument('--test-directory', type=str, help='Path to the directory containing test images when testing')

    args = parser.parse_args()
    
    config = DotDict.from_toml('config.toml')  # Load configuration
    set_seed(config.seed)
    # config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        config.device = 'mps'
    elif torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(), 
    ])

    train_dataloader = load_image_data(args.image_directory, config.image_size, config.batch_size, use_sampler=True, transform=train_transform)
    
    # Initialize your VIT model and optimizer
    model = VisionTransformer(config)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Optional: Load pretrained model weights
    model_path = Path(config.model_path)
    model_path.parent.mkdir(exist_ok=True)
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model weights from {model_path}")
    except FileNotFoundError:
        print('No saved model weights found.')
    except Exception as e:
        raise e
    finally:
        logger.info(f'Training data: {args.image_directory}')
        # Train the model
        train_model(model, train_dataloader, optimizer, loss_function, num_epochs=config.num_epochs, device=config.device, model_path=model_path)
        torch.save(model.state_dict(), config.model_path)
        
    if args.test:
        logger.info(f'Testing data: {args.test_directory}')

        test_dataloader = load_image_data(args.test_directory, config.image_size, config.batch_size)
        
        # plotting
        y_true, y_pred = predict_model(model, test_dataloader, device=config.device)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Log the metrics
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Recall: {recall}')
        logger.info(f'F1 Score: {f1}')
        logger.info('-'*40)
        # Print metrics in the terminal
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')


if __name__ == "__main__":
    main()
