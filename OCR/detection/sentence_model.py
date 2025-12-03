import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.optim as optim

from OCR.prepare_image import preprocess_single_image
from .extract_from_masks import main as efm, extract_individual_sentence_regions2
from .extract_from_masks import extract_images_from_mask_for_ocr
import os
from PIL import Image
from .calculate_sentence_weight import (calculate_sentence_weight)
import matplotlib

matplotlib.use("TkAgg")

class CustomCRAFT(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomCRAFT, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.refine_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.refine_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence = nn.Conv2d(64, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.refine_conv1(x))
        x = self.dropout(x)
        x = F.relu(self.refine_conv2(x))
        x_upsampled = self.upsample(x)
        sentence = torch.sigmoid(self.sentence(x_upsampled))

        return sentence


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        # Load grayscale image (force 'L' mode)
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('L')  # <-- force grayscale

        # Load corresponding mask (same name) as grayscale
        mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # <-- force grayscale
        else:
            mask = Image.new('L', image.size, 0)  # black mask if missing

        # Apply optional transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class CustomDataset2(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        # Load grayscale image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('L')  # <-- force grayscale

        # Load mask as grayscale
        mask_path = os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # <-- force grayscale
        else:
            mask = Image.new('L', image.size, 0)

        # Apply optional transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def visualize(image, sentence, predicted_mask):
    """Visualize the original image, ground truth masks, and predicted masks."""
    # Convert tensors to numpy for visualization
    image_np = image.squeeze(0).cpu().numpy()

    ground_truth_masks_np = sentence.squeeze(0).cpu().numpy()
    predicted_masks_np = predicted_mask.squeeze(0).cpu().numpy()

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 15))

    axs[0, 0].imshow(image_np, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(ground_truth_masks_np, cmap='gray')
    axs[1, 0].set_title('GT Sentence')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(predicted_masks_np, cmap='gray')
    axs[1, 1].set_title('Predicted Sentence')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def show_image_with_masks(image, mask):
    """
    Function to overlay each mask in a tuple onto the grayscale image and display them.
    Only the mask areas with confidence >= 0.5 are shown.

    :param image: The original input image tensor.
    :param masks_tuple: A tuple of predicted masks for each object/segment in the image.
    """
    masks = ["Number", "Character", "Break", "Space", "Line", "Sentence", "Error", "Multiple", "Notes", "Switch", "Detail"]

    # Convert image to grayscale
    image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    grayscale_image = np.mean(image_np, axis=2)  # Convert to grayscale
    grayscale_image = np.clip(grayscale_image, 0, 1)  # Clip values for display

    # Create a figure to display the image and masks
    fig, axs = plt.subplots(1, 2, figsize=(18, 15))

    # Display the original grayscale image
    axs[0].imshow(grayscale_image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original Grayscale Image')

    mask_np = mask.cpu().numpy()  # Move each mask to CPU and convert to NumPy

    # Apply a confidence threshold of 0.5
    mask_np = np.where(mask_np >= 0.8, mask_np, 0)  # Set values < 0.5 to 0

    # Normalize mask to [0, 1] for visualization
    mask_normalized = mask_np / np.max(mask_np) if np.max(mask_np) != 0 else mask_np

    # Create an RGB mask overlay (Red channel)
    colored_mask = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3))
    colored_mask[:, :, 0] = mask_normalized  # Red channel for mask

    # Overlay the mask onto the grayscale image with transparency
    overlayed_image = 0.3 * np.stack([grayscale_image] * 3, axis=-1) + 0.6 * colored_mask  # Alpha blend

    # Display the overlayed image
    axs[1].imshow(overlayed_image)
    axs[1].axis('off')
    axs[1].set_title(f'Sentence mask')

    plt.tight_layout()
    plt.show()


def train(model, dataloader, criterion, optimizer, device, num_epochs, modelname, validate_dataloader, patience=20):
    model.train()  # Set the model to training mode
    best_loss = load_best_loss(f'best_loss_{modelname}.json')
    last_best_epoch = 0

    for epoch in range(num_epochs):
        running_loss = 0.0  # Track the total loss for this epoch

        for batch_idx, (image, sentence_masks) in enumerate(dataloader):

            # Move data to the appropriate device (GPU or CPU)
            image = image.to(device)

            sentence_masks = sentence_masks.to(device)

            # Forward pass
            sentence_output= model(image)

            # Get the target size from ground truth masks
            target_size = sentence_masks.shape[-2:]  # Assuming all masks have the same size


            if sentence_output.shape[-2:] != target_size:
                sentence_output = F.interpolate(sentence_output, size=target_size, mode='bilinear', align_corners=False)

            # Compute loss for each output
            loss_sentence = criterion(sentence_output, sentence_masks)

            # Sum the losses
            total_loss = loss_sentence

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += total_loss.item()

        # Print the average loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        validation_loss = validate(model, validate_dataloader, criterion, device, epoch)
        if validation_loss < best_loss:
            torch.save(model.state_dict(), f'{modelname}.pth')
            print('Saved model with average loss:', validation_loss)
            # Update best_loss and save to JSON
            best_loss = validation_loss
            save_best_loss(best_loss, f'best_loss_{modelname}.json')
            last_best_epoch = epoch
        elif (epoch - last_best_epoch) > patience:
            print("Early stopping")
            break
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


        #Visualization after each epoch
        # if (epoch) % 10 == 0:  # Adjust the interval as needed
        #     with torch.no_grad():  # Disable gradient calculation
        #         # Visualize the first image in the epoch
        #         visualize(image[0], sentence_masks[0], sentence_output[0])

        if epoch == num_epochs-1:
            with torch.no_grad():  # Disable gradient calculation
                # Visualize the first image in the epoch
                visualize(image[0], sentence_masks[0], sentence_output[0])
                efm(image[0], sentence_output[0])


def validate(model, dataloader, criterion, device, epoch):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        running_loss = 0.0  # Track the total loss for this epoch

        for batch_idx, (image, sentence_masks) in enumerate(dataloader):

            # Move data to the appropriate device (GPU or CPU)
            image = image.to(device)

            sentence_masks = sentence_masks.to(device)

            # Forward pass
            sentence_output= model(image)

            # Get the target size from ground truth masks
            target_size = sentence_masks.shape[-2:]  # Assuming all masks have the same size


            if sentence_output.shape[-2:] != target_size:
                sentence_output = F.interpolate(sentence_output, size=target_size, mode='bilinear', align_corners=False)

            # Compute loss for each output
            loss_sentence = criterion(sentence_output, sentence_masks)

            # Sum the losses
            total_loss = loss_sentence

            # Accumulate loss
            running_loss += total_loss.item()

        # Print the average loss for this epoch
        validation_loss = running_loss / len(dataloader)
        print("Validation Loss: {:.4f}".format(validation_loss))
        #Visualization after each epoch
        # if (epoch) % 10 == 0:  # Adjust the interval as needed
        #     with torch.no_grad():  # Disable gradient calculation
        #         # Visualize the first image in the epoch
        #         visualize(image[0], sentence_masks[0], sentence_output[0])
        return validation_loss


def test(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (image, *_) in enumerate(dataloader):  # We no longer need the ground truth masks
            # Move the input images to the appropriate device (GPU or CPU)
            image = image.to(device)

            # Forward pass to generate predictions
            sentence_output = model(image)

            # Get the target size from the input images
            target_size = image.shape[-2:]  # Assume images and outputs should match in size

            # Interpolate outputs to match the input image size if needed
            if sentence_output.shape[-2:] != target_size:
                sentence_output[batch_idx] = F.interpolate(sentence_output, size=target_size, mode='bilinear', align_corners=False)

            # Visualize the first batch's results
            if batch_idx == 0:
                show_image_with_masks(image[0], sentence_output[0])
                #efm(image[0], sentence_output[0])

    print("Testing completed.")


def load_best_loss(filename='best_loss.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get('best_loss', float('inf'))  # Return inf if not found
    return float('inf')  # Return inf if the file doesn't exist


def save_best_loss(best_loss, filename='best_loss.json'):
    with open(filename, 'w') as f:
        json.dump({'best_loss': best_loss}, f)


def main():
    # Set device to GPU if available, otherwise fall back to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transform for image and mask processing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor
    ])

    # Load dataset
    base_dir = 'data'
    train_dir = os.path.join(base_dir, 'trainv2')
    validate_dir = os.path.join(base_dir, 'validate')
    test_dir = os.path.join(base_dir, 'test')
    train_image_dir = os.path.join(train_dir, 'images')  # Path to input images
    train_mask_dir = os.path.join(train_dir, 'masks')  # Path to mask images
    validate_image_dir = os.path.join(validate_dir, 'images')  # Path to input images
    validate_mask_dir = os.path.join(validate_dir, 'masks')  # Path to mask images
    train_dataset = CustomDataset2(train_image_dir, train_mask_dir, transform=transform)
    validate_dataset = CustomDataset(validate_image_dir, validate_mask_dir, transform=transform)

    # DataLoader
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    # Model training setup
    model = CustomCRAFT().to(device)
    # criterion = nn.BCELoss()
    weight = (torch.tensor(calculate_sentence_weight(), dtype=torch.float32) / 1.3).to(device)
    criterion = nn.BCELoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model_path = "sentence_model.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Number of epochs
    num_epochs = 300

    # Train for multiple epochs
    train(model, train_dataloader, criterion, optimizer, device, num_epochs, "sentence_model", validate_dataloader, patience=100)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Load the test dataset (you can use ImageFolder or a custom dataset class)
    test_dataset = CustomDataset2(test_dir, test_dir, transform=transform)

    # Create the test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    test(model, test_loader, device)


def main_with_split():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # === Load and split dataset ===
    base_dir = 'data'
    # full_dir = os.path.join(base_dir, 'trainv2')
    full_dir = os.path.join(base_dir, 'trainSmall')
    image_dir = os.path.join(full_dir, 'images')
    mask_dir = os.path.join(full_dir, 'masks')

    full_dataset = CustomDataset2(image_dir, mask_dir, transform=transform)

    # Split 80% train / 20% validation
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, validate_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset split: {train_size} train, {val_size} validate")

    # === DataLoaders ===
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    # === Model setup ===
    model = CustomCRAFT().to(device)
    weight = (torch.tensor(calculate_sentence_weight(), dtype=torch.float32) / 1.3).to(device)
    criterion = nn.BCELoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model_path = "small_sentence_model.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded existing model weights.")

    # === Training ===
    num_epochs = 300
    train(model, train_dataloader, criterion, optimizer, device, num_epochs, "small_sentence_model", validate_dataloader, patience=100)

    # === Testing ===
    test_dataset = CustomDataset2(os.path.join(base_dir, 'test'),
                                  os.path.join(base_dir, 'test'),
                                  transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test(model, test_loader, device)

def extract_sentences_from_dataset(model, dataset_dir, output_dir=None, device='cuda', batch_size=1):
    """
    Extracts all sentence regions from the images in dataset_dir using the model
    and saves them into output_dir as individual images.

    :param model: Trained sentence prediction model (CustomCRAFT)
    :param dataset_dir: Directory containing the images
    :param output_dir: Directory to save extracted sentences. Default: ../ocr/sentences
    :param device: 'cuda' or 'cpu'
    :param batch_size: Batch size for inference
    """

    # Default output folder
    if output_dir is None:
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
        output_dir = os.path.join(parent_dir, "classifier", "further_data2")
    os.makedirs(output_dir, exist_ok=True)

    # Transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Custom dataset for inference
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, image_name)
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)
            return img, image_name

    # Create DataLoader
    dataset = InferenceDataset(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(dataloader):
            images = images.to(device)
            predictions = model(images)

            # Process each image in the batch
            for i in range(images.size(0)):
                image_tensor = images[i].cpu()
                pred_mask = predictions[i].cpu()
                filename = os.path.splitext(image_names[i])[0]

                # Extract and save individual sentences
                extract_images_from_mask_for_ocr(image_tensor, pred_mask, filename)

    print("All sentence regions extracted and saved.")


def extract_all_sentences():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomCRAFT()
    model.load_state_dict(torch.load("small_sentence_model.pth", map_location=device))
    model.to(device)

    # extract_sentences_from_dataset(model, dataset_dir="data/trainSmall2/images")
    extract_sentences_from_dataset(model, dataset_dir="data/further_data2")

def save_debug_image(img, name):
    """
    Saves a NumPy or Torch tensor as a PNG for debugging.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # Remove batch/channel dims if present
    while img.ndim > 2:
        img = img[0]

    # Normalize if mask is float/activation-like
    if img.dtype != np.uint8:
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(f"debug_{name}.png")
    print(f"[DEBUG] Saved debug_{name}.png")
def extract_sentences_from_image(model, image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Preprocess
    img = preprocess_single_image(image)  # NumPy array, grayscale (H, W)
    save_debug_image(img, "preprocessed")

    # Convert to float tensor and normalize to [0,1]
    img_tensor = torch.from_numpy(img).float() / 255.0  # shape (H, W)

    # Add channel dimension (1) and batch dimension (1)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # shape (1, 1, H, W)
    save_debug_image(img_tensor, "tensor")

    # Forward pass
    with torch.no_grad():
        prediction = model(img_tensor)

    # Move back to CPU for further processing
    image_tensor = img_tensor.cpu()
    pred_mask = prediction.cpu()
    save_debug_image(pred_mask, "predmask")

    return extract_individual_sentence_regions2(image_tensor, pred_mask)



if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    print("GPU devices:", torch.cuda.device_count())
    # main()
    # main_with_split()
    extract_all_sentences()