import small_version_of_project.model.train as train
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from torch.utils.data import DataLoader
import small_version_of_project.model.train as train

def test_dataset():
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))])
    train_dataset = train.get_data_loaders(transform_train= transform_train , batch_size=128)
    print(train_dataset)

def plotting_train_data(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plotting the loss and accuracy over epochs
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_data(class_names, data_loader):
    # Create a dictionary to store one sample image per class
    class_samples = {class_name: None for class_name in class_names}
    # Find one sample image for each class
    for images, labels in data_loader:
        for image, label in zip(images, labels):
            if class_samples[class_names[label]] is None:
                class_samples[class_names[label]] = image
                # Break once a sample is found for each class
                if all(sample is not None for sample in class_samples.values()):
                    break

    # Visualize one sample from each class
    plt.figure(figsize=(12, 8))
    for i, (class_name, sample_image) in enumerate(class_samples.items()):
        plt.subplot(2, 5, i + 1)
        sample_image = sample_image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C) for plotting
        plt.imshow((sample_image + 1) / 2)  # Denormalize the image
        plt.title(class_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
