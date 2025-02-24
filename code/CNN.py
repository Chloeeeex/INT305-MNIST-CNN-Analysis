import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# DataLoader for batching and shuffling
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Define simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)  # Convolution layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.fc1 = nn.Linear(16 * 14 * 14, 64)  # Fully connected layer 1
        self.fc2 = nn.Linear(64, 10)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply conv and pooling
        x = x.view(-1, 16 * 14 * 14)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))  # Apply ReLU
        x = self.fc2(x)  # Output
        return x

# Initialize model and move it to device
model = SimpleCNN().to(device)

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD optimizer

# 4. Train and evaluate the model
# List for storing info from each epoch
train_losses = []
train_accuracies = []
test_accuracies = []

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=3):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
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
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()  # Correct predictions

        # Calculate and store metrics
        train_accuracy = 100 * correct_train / total_train
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        train_accuracies.append(train_accuracy)

        # Test model
        test_accuracy = test_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No gradient computation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total  # Test accuracy

# Start training
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=3)

# 5. Plot training loss curve
def plot_loss_curve():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# 6. Visualize correct and incorrect predictions
def visualize_predictions(model, loader, num_images=5):
    model.eval()  # Set model to evaluation mode
    correct_images = []
    incorrect_images = []

    with torch.no_grad():  # No gradient computation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert outputs to probabilities
            confidences, predicted = torch.max(probabilities, 1)  # Get predictions and confidence

            for i in range(len(images)):
                if predicted[i] == labels[i] and len(correct_images) < num_images:
                    correct_images.append((images[i], labels[i], predicted[i], confidences[i]))
                elif predicted[i] != labels[i] and len(incorrect_images) < num_images:
                    incorrect_images.append((images[i], labels[i], predicted[i], confidences[i]))

                if len(correct_images) >= num_images and len(incorrect_images) >= num_images:
                    break

    # Function to plot images
    def plot_images(images, title):
        plt.figure(figsize=(15, 6))
        for idx, (image, label, pred, confidence) in enumerate(images):
            plt.subplot(1, num_images, idx + 1)
            plt.imshow(image.cpu().squeeze(), cmap='gray')
            plt.title(f"Label: {label}\nPred: {pred}\nConf: {confidence:.2f}", fontsize=10)
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    plot_images(correct_images, "Correct Predictions")
    plot_images(incorrect_images, "Incorrect Predictions")

# 7. Plot confusion matrix
def plot_confusion_matrix(model, loader):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradient computation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)  # Compute confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(10)])

    # Annotate matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

# Call visualization functions
plot_loss_curve()
visualize_predictions(model, test_loader)
plot_confusion_matrix(model, test_loader)

# Print final test accuracy
final_test_accuracy = test_model(model, test_loader)
print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")