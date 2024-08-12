import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

# ----- creating dataset with lineear regression -----

weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print("Input numbers", X[:10])
print("Output numbers", y[:10])

# ------SPLITTING DATA INTO TRAINING AND TEST SETS-----

# create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))


# vizualize the data

def plot_prediction(train_data=X_train, train_labels=y_train,
                    test_data=X_test, test_labels=y_test, predictions=None):
    # plot training data test data and comparing predictions
    plt.figure(figsize=(10, 7))

    # plot taining data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training Data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing Data")

    # Are there predicitions?
    if predictions is not None:
        # plot existing predictions
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


plot_prediction()


# Building PyTorch Linear Model

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # use nn.Linear() for creating model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1)

# training model (loss function, optimizer, training loop, testing loop)

# loss function
loss_fn = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=.01)
torch.manual_seed(42)
epochs = 200

for epoch in range(epochs):
    model_1.train()

    # Forward pass
    y_pred = model_1(X_train)

    # Calculate loss
    loss = loss_fn(y_pred, y_train)

    # optimize zero grad
    optimizer.zero_grad()

    # perform backprop
    loss.backward()

    # optimzer step
    optimizer.step()

    # testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")

