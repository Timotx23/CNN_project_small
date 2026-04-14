import small_version_of_project.model.CNN_model as CNN_model

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
import small_version_of_project.model.train as train
train_loader, val_loader, test_loader, class_names = train.get_data_loaders(None)
import small_version_of_project.model.CNN_model as CNN_model

device = CNN_model.to_devices()


def visualize_data():   
    train.visualize_data(class_names, train_loader)

    # Extract labels from the subset
    subset_labels = []

    for _, labels in train_loader:
        # Assuming labels are a tensor, we convert them to a list and extend our accumulating list
        subset_labels.extend(labels.tolist())

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(subset_labels, bins=range(11), edgecolor='black', align='left', rwidth=0.8)
    plt.title('Histogram of Labels in the train set')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(range(10))
    plt.show()

    
def test_simple_cnn():
    # Simple test script for SimpleCNN initialization and forward pass
    # Create model instance
    model = CNN_model.SimpleCNN()

    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    dummy_input = torch.randn(2, 3, 32, 32)

    # Move model and input to the appropriate device
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, 10)")
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print("✓ Model initialization and forward pass successful!")




def train_simple():

    # Setting Hyperparameters and Training the Model

    # Number of training epochs
    epochs = 40

    # Create an instance of the SimpleCNN model and move it to the specified device (GPU if available)
    model = CNN_model.SimpleCNN().to(device)
    print("Device", device)
    # Define the loss criterion (CrossEntropyLoss) and the optimizer (Adam) for training the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model using the defined training function
    val_losses_simple = train.train_model(model, train_loader, val_loader, epochs, criterion, optimizer)
    return val_losses_simple


def test_dropout():
        # Simple test for SimpleCNN_dropout
    model_d = CNN_model.SimpleCNN_dropout(dropout_prob=0.5)
    print(f"Model created with dropout_prob={model_d.dropout_prob}")

    # Test with dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model_d(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 10)")
    assert output.shape == (2, 10), f"Shape mismatch: got {output.shape}"
    print("✓ SimpleCNN_dropout test passed!")

def train_with_dropout():
    epochs = 100
    model_d = CNN_model.SimpleCNN_dropout(dropout_prob=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_d.parameters(), lr=0.001)
    val_losses_dropout = train.train_model(model_d, train_loader, val_loader, epochs, criterion, optimizer)
    return val_losses_dropout

def comparing_models():
    #Call models
    val_losses_simple = train_simple()
    val_losses_dropout = train_with_dropout()
    #val_losses_data_aug = train_with_data_augmentation()


    plt.plot(val_losses_simple, label='No regularization')
    plt.plot(val_losses_dropout, label='Dropout')
    #plt.plot(val_losses_data_aug, label='Data augmentation + Dropout')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()