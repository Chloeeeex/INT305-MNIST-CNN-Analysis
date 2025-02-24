import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
    transforms.RandomHorizontalFlip(p=0.1),  # Data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]), download=True)

# DataLoader for batching and shuffling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model configuration with increased capacity
config = ViTConfig(
    image_size=224,
    num_channels=1,  # Grayscale input for MNIST
    num_labels=10,   # 10 classes for MNIST
    hidden_size=256,  # Larger hidden size
    num_hidden_layers=8,  # Increased number of layers
    num_attention_heads=8,
    intermediate_size=512  # Larger intermediate size
)
model = ViTForImageClassification(config).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with accuracy tracking
def train_vit_with_accuracy(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        # Training step
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        # Track and log training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy after each epoch
        test_accuracy = evaluate_accuracy(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1} completed. Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # Plot metrics after training
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, marker='o', label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test accuracy calculation
def evaluate_accuracy(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total  # Return test accuracy

# Testing loop with confidence display
def test_vit_with_confidence(model, test_loader):
    model.eval()
    correct_images = []
    incorrect_images = []
    correct_preds = []
    incorrect_preds = []

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probs = torch.softmax(outputs, dim=1)  # Compute softmax probabilities
            conf, predicted = torch.max(probs, 1)

            for i in range(images.size(0)):
                if predicted[i] == labels[i]:
                    correct_images.append(images[i].cpu())
                    correct_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))
                else:
                    incorrect_images.append(images[i].cpu())
                    incorrect_preds.append((predicted[i].item(), labels[i].item(), conf[i].item()))

    # Visualize correct and incorrect predictions
    show_images(correct_images, correct_preds, "Correct Predictions (with Confidence)", max_images=5)
    show_images(incorrect_images, incorrect_preds, "Incorrect Predictions (with Confidence)", max_images=5)

# Visualization helper function
def show_images(images, preds_labels, title, max_images=10, images_per_row=5):
    rows = (max_images + images_per_row - 1) // images_per_row
    plt.figure(figsize=(15, rows * 3))
    for i in range(min(max_images, len(images))):
        plt.subplot(rows, images_per_row, i + 1)
        img = images[i].squeeze(0).numpy()
        pred, label, conf = preds_labels[i]
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}\nPred: {pred} ({conf:.2f})")
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Train and Test the model
train_vit_with_accuracy(model, train_loader, test_loader, criterion, optimizer, epochs=5)
test_vit_with_confidence(model, test_loader)