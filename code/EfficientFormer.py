import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import timm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List available models from timm library
print(timm.list_models())

# 1. Data preparation with augmentation
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 for EfficientFormer input
    transforms.Grayscale(3),  # Convert MNIST from 1 channel to 3 channels (RGB)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Define EfficientFormer model (pretrained)
model = timm.create_model('efficientformer_l1', pretrained=True, num_classes=10).to(device)

# 3. Set loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 4. Training and evaluation loop
train_losses = []  # To record training loss per epoch
train_accuracies = []  # To record training accuracy
test_accuracies = []  # To record test accuracy

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=3):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training accuracy and loss
        train_accuracy = 100 * correct_train / total_train
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        train_accuracies.append(train_accuracy)

        # Update learning rate
        scheduler.step()

        # Evaluate model on test data
        test_accuracy = test_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        # Print training stats for each epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"efficientformer_epoch_{epoch + 1}.pth")

# 5. Evaluate model accuracy
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total  # Return test accuracy

# 6. Plot loss curve and confusion matrix
def plot_loss_curve():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(10)])

    # Annotate confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

# Call visualization functions after training
plot_loss_curve()
plot_confusion_matrix(model, test_loader)

final_test_accuracy = test_model(model, test_loader)
print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")