import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm  # For displaying progress bar

# Device configuration (Use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data preparation with transformation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Define Depthwise Separable Convolution (Core of MobileNet)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: applies 3x3 convolution for each channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        # Pointwise convolution: 1x1 convolution to combine output channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)  # Apply depthwise convolution
        x = self.pointwise(x)  # Apply pointwise convolution
        return x

# 3. Define MobileNet model
class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Standard convolution
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64),  # Depthwise separable convolutions
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(128, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)  # Global average pooling to reduce dimensions
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten output
        x = self.classifier(x)  # Classifier layer
        return x

# Instantiate the model
model = MobileNet().to(device)

# 4. Set loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 5. Training and evaluation loop
train_losses = []  # To record training loss for each epoch
train_accuracies = []  # To record training accuracy for each epoch
test_accuracies = []  # To record test accuracy for each epoch

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=5):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()  # Calculate correct predictions

            progress_bar.set_postfix(loss=loss.item())

        # Calculate and store metrics
        train_accuracy = 100 * correct_train / total_train
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        train_accuracies.append(train_accuracy)

        # Update learning rate scheduler
        scheduler.step()

        # Evaluate on test dataset
        test_accuracy = test_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        # Print training stats
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"mobilenet_epoch_{epoch + 1}.pth")

# 6. Evaluate accuracy on test dataset
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Calculate correct predictions
    return 100 * correct / total  # Return accuracy

# 7. Plot training loss curve
def plot_loss_curve():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# 8. Plot confusion matrix
def plot_confusion_matrix(model, loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed during evaluation
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

# Start training
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=5)

# Visualize loss curve and confusion matrix
plot_loss_curve()
plot_confusion_matrix(model, test_loader)

final_test_accuracy = test_model(model, test_loader)
print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")