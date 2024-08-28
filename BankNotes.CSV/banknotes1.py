import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import csv
import random

# Define the neural network model
class BanknoteClassifier(nn.Module):
    def __init__(self):
        super(BanknoteClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input layer (4 features) to hidden layer
        self.fc2 = nn.Linear(64, 32)  # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(32, 1)  # Hidden layer to output layer (1 output)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Read data from the file
with open("banknotes_large.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 0 if row[4] == "0" else 1  # Use 0 for Authentic, 1 for Counterfeit
        })

# Shuffle and split data
random.shuffle(data)
X = [row["evidence"] for row in data]
y = [row["label"] for row in data]

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the model, loss function, and optimizer
model = BanknoteClassifier()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()  # Apply threshold to get binary output
    accuracy = (predicted_classes == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')
