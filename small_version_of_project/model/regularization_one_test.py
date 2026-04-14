import CNN_model

# Import necesary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from torch.utils.data import DataLoader
import train
train_loader, val_loader, test_loader, class_names = train.get_data_loaders(None)


#Prep work.
device = CNN_model.to_devices()


def train_with_data_augmentation(model_d1, epochs):
    # Set the number of epochs and batch size

    # Create a CNN model with dropout
    #model_d1 = regularization_in_one_file.SimpleCNN_dropout(dropout_prob=0.3).to(regularization_in_one_file.to_device()) # Changed dropout probability to 0.3 since 0.5 was too high in combination with data augmentation
   
    print("Device", device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 )
    optimizer2 = optim.Adam(model_d1.parameters(), lr=0.001)# old
    optimizer = torch.optim.Adam(model_d1.parameters(), lr=0.001, weight_decay=1e-4) #normal adam has given the best results
    #optimizer 3
    optimizer3 = torch.optim.AdamW(model_d1.parameters(), lr=0.001, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)


    # Define data augmentation transformations for training (hint: think about what transformations can help with learning more robust reopresentations)
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
]) #YOUR CODE HERE #

    # Create data loaders for training, validation, and testing using the defined transformations in transform_train
    train_loader, val_loader, test_loader, _ = train.get_full_db(transform_train, batch_size=128)# YOUR CODE HERE #
    
    val_losses_data_aug = train.train_model(model_d1, train_loader, val_loader, epochs, criterion, optimizer, scheduler)
    return val_losses_data_aug, test_loader


# Test the Model
def test_model(model, test_loader):

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize counters
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # Disable gradient computation during testing
    with torch.no_grad():
        # Iterate through the test loader
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for further analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())


    # Calculate accuracy
    accuracy = correct / total

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')

    # Build confusion matrix
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix with class names
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Display test results
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f"F1 = {f1:.2f}")

    return accuracy, f1



model_d1 = CNN_model.SimpleCNN_dropout(dropout_prob=0.2).to(device)
epochs = 200
val_looses_data_aug, test_loader = train_with_data_augmentation(model_d1, epochs)
acc, f1 = test_model(model_d1, test_loader)
