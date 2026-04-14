

# Import necesary libraries
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
import CNN_model
device = CNN_model.to_devices()

# Function to Get Data Loaders for CIFAR-10 Dataset

# This function takes a transformation for training data and a batch size as input and returns train, validation, and test data loaders along with class names.

def get_data_loaders(transform_train, batch_size=128):
    # Create a transform to convert images to tensors and normalize them
    transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    



    # Load the full CIFAR10 dataset
    dataset_ = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # If a train_transform is given, create a new dataset with that transform for the train DataLoader
    if transform_train is None:
        dataset_train = dataset_
    else:
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

    # Get the class names of CIFAR-10
    class_names = dataset_.classes

    # Define the size of the train, validation, and test sets
    train_size = 5000
    val_size = 1000
    test_size = 1000
    subset_size = train_size + val_size + test_size 

    # Create a balanced subset
    # Use a dictionary to store indices for each class
    # Initialize an empty dictionary, where keys are class indices and values are lists to store indices for each class.
    class_indices_dict = {class_idx: [] for class_idx in range(len(class_names))}

    # Iterate over the dataset to collect indices for each class
    # Iterate through the dataset, extract the index `i` and label from each element, and append the index to the corresponding class in the dictionary.
    for i, (_, label) in enumerate(dataset_):
        class_indices_dict[label].append(i)

    # Split the indices into training, validation, and test sets
    train_indices = []
    val_indices = []
    test_indices = []

    train_size_per_class = train_size // len(class_names)
    val_size_per_class = val_size // len(class_names)
    test_size_per_class = test_size // len(class_names)
    for idx in range(len(class_names)):
        indices = class_indices_dict[idx][:subset_size]
        train_indices.extend(indices[:train_size_per_class])
        val_indices.extend(indices[train_size_per_class:train_size_per_class + val_size_per_class])
        test_indices.extend(indices[train_size_per_class + val_size_per_class:train_size_per_class + val_size_per_class + test_size_per_class])

    # Create subsets
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_, val_indices)
    test_dataset = Subset(dataset_, test_indices)

    # Create data loader for the training set with WeightedRandomSampler
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Create data loader for the validation set
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    # Create data loader for the test set
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Train batches per epoch:", len(train_loader))

    return train_loader, val_loader, test_loader, class_names





def get_full_db(transform_train=None, batch_size=128):
    # Default transform (for val/test)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Load full CIFAR-10 training dataset
    dataset_full = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    # Training dataset (with optional augmentation)
    if transform_train is None:
        dataset_train = dataset_full
    else:
        dataset_train = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transform_train,
            download=True
        )

    class_names = dataset_full.classes
    num_classes = len(class_names)

    # 📌 Number of samples per class (CIFAR-10 has 5000 per class)
    samples_per_class = int(5000 * 1)

    # Split ratios
    train_split = int(0.8 * samples_per_class)
    val_split = int(0.1 * samples_per_class)
    test_split = samples_per_class - train_split - val_split

    # Build class index dictionary
    class_indices_dict = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset_full):
        class_indices_dict[label].append(idx)

    # Collect indices
    train_indices, val_indices, test_indices = [], [], []

    for cls in range(num_classes):
        indices = class_indices_dict[cls][:samples_per_class]

        train_indices.extend(indices[:train_split])
        val_indices.extend(indices[train_split:train_split + val_split])
        test_indices.extend(indices[train_split + val_split:])

    # Create subsets
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_full, val_indices)
    test_dataset = Subset(dataset_full, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Debug prints
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Train batches per epoch:", len(train_loader))

    return train_loader, val_loader, test_loader, class_names




# Training Loop with Validation
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler):
    model.to(device)
    # Lists to store training and validation losses, and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Loop over epochs
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Calculate training accuracy
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Validation without gradient computation
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

                _, predicted_val = torch.max(val_outputs.data, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted_val == val_labels).sum().item()

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation accuracy
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        scheduler.step(avg_val_loss) #adapt learning rate
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, '
                f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
        #plotting_train_data(train_losses, val_losses, train_accuracies, val_accuracies)
        torch.save(model.state_dict(), "model_d4.pth")
    return val_losses



