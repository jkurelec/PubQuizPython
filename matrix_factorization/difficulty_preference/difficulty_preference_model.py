import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Simulate sample data: rating difference and difficulty rating
# rating_diff is in range [-200, 200], difficulty_rating is in {-1, 0, 1}
np.random.seed(42)
rating_diff = np.random.uniform(-200, 200, 300)
difficulty_rating = np.where(rating_diff < -50, -1, np.where(rating_diff > 50, 1, 0))

# Normalize rating difference to [0, 1]
rating_diff_normalized = (rating_diff + 200) / 400

# Convert to torch tensors
X = torch.tensor(rating_diff_normalized, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(difficulty_rating, dtype=torch.float32).unsqueeze(1)

# Define a simple linear regression model
class DifficultyRegressionModel(nn.Module):
    def __init__(self):
        super(DifficultyRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate model, define loss and optimizer
model = DifficultyRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Generate predictions for visualization
x_vals = torch.linspace(0, 1, 100).unsqueeze(1)
with torch.no_grad():
    y_preds = model(x_vals)

# Plot the learned difficulty curve
plt.figure(figsize=(8, 5))
plt.plot(x_vals.numpy(), y_preds.numpy(), label='Predicted Difficulty')
plt.xlabel('Normalized Rating Difference')
plt.ylabel('Predicted Difficulty Rating')
plt.title('Learned Difficulty Preference Curve (PyTorch)')
plt.grid(True)
plt.legend()
plt.savefig("difficulty_preference_curve.png")
plt.show()

