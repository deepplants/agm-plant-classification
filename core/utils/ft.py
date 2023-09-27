import torch
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import os

# Define a custom dataset to hold the extracted features
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)

# Function to fine-tune a linear layer on the extracted features
def fine_tune_linear(features, labels, num_classes, num_epochs, learning_rate, batch_size=32):
    # Create a DataLoader for the dataset
    dataset = FeatureDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define a linear layer for fine-tuning
    linear_layer = torch.nn.Linear(features.size(1), num_classes)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_layer.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Forward pass
            outputs = linear_layer(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return linear_layer

# Function to train an SVM on the extracted features
def train_svm(features, labels):
    svm_classifier = SVC()
    svm_classifier.fit(features, labels)
    return svm_classifier

# Function to train a kNN classifier on the extracted features
def train_knn(features, labels, n_neighbors):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(features, labels)
    return knn_classifier

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs)
            labels.append(targets)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def main_ft(feature_extractor, train_dataloader, val_dataloader, log_dir, run_id, device,
         num_classes=2, batch_size=32, num_epochs=10, learning_rate=0.01, n_neighbors=3,
         linear_layer=True, svm=True, knn=True):

    print("Running simple fine-tuning")
    # Set up log file
    log_file = os.path.join(log_dir, f'fine_tuning{run_id}.log')    

    train_features, train_labels = extract_features(feature_extractor, train_dataloader, device)
    val_features, val_labels = extract_features(feature_extractor, val_dataloader, device)

    # bring to cpu
    train_features = train_features.detach().cpu()
    train_labels = train_labels.detach().cpu()
    val_features = val_features.detach().cpu()
    val_labels = val_labels.detach().cpu()

    if linear_layer:
        print("Fine-tuning a linear layer")
        # Fine-tune a linear layer
        linear_layer = fine_tune_linear(train_features, train_labels, num_classes=num_classes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

        # Evaluate on validation set
        linear_layer.eval()
        with torch.no_grad():
            val_outputs = linear_layer(val_features)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            log_to_file(log_file, f"Linear Layer Validation Accuracy: {val_accuracy}")

    if svm:
        print("Training an SVM")
        # Train an SVM
        svm_classifier = train_svm(train_features, train_labels)

        # Evaluate on validation set
        svm_val_predictions = svm_classifier.predict(val_features)
        svm_val_accuracy = accuracy_score(val_labels, svm_val_predictions)
        log_to_file(log_file, f"SVM Validation Accuracy: {svm_val_accuracy}")

    if knn:
        print("Training a kNN classifier")
        # Train a kNN classifier
        knn_classifier = train_knn(train_features, train_labels, n_neighbors=n_neighbors)

        # Evaluate on validation set
        knn_val_predictions = knn_classifier.predict(val_features)
        knn_val_accuracy = accuracy_score(val_labels, knn_val_predictions)
        log_to_file(log_file, f"kNN Validation Accuracy: {knn_val_accuracy}")



if __name__ == "__main__":
    main_ft()