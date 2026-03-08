import torch
import torch.nn as nn
import torch.optim as optim

# Sample dataset: AND logic gate
x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[1]], dtype=torch.float32)

# Define Perceptron model
class Perceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)  # single neuron
        self.sigmoid = nn.Sigmoid()        # activation function

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Create model
model = Perceptron(input_size=2)

# Loss and optimizer
criterion = nn.BCELoss()               # binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test
with torch.no_grad():
    predictions = model(x).round()
    print("Predictions:\n", predictions)
