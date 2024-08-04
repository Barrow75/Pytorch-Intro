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

print("\n")


# Create Linear Regression Model Class start with random values, look at training data and adjust the random values, or
# better represent (get closer to) ideal values (aka weight and bias we used to create the data) by:
# 1. Gradient Descent
# 2. Back Propagation
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,  # start with random weight and try to adjust it to the ideal weight
                                                requires_grad=True,  # can parameter be upddated by gradient descent?
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,  # start with random weight and try to adjust it to the ideal bias
                                             requires_grad=True,  # can parameter be upddated by gradient descent?
                                             dtype=torch.float))

        # Forward method to define computation in model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias  # Linear Regression Formula


# creating random seed
rand_seed = torch.manual_seed(42)

# create an instance of themodel (sublacc for nn.Module)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))

# making predictions using torch.inference_mode() -
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
plot_prediction(predictions=y_preds)

# -- Train the Model --

# Loss Function - Meaures how wrong the model predictions are (lower = better)
# Optimizer - Takes into account the loss of a model and adjusts the models parameters (weights/bias) to improve LF
param = list(model_0.parameters())
print(param)
print("\n")
# models parameter
para = model_0.state_dict()
print("Modles Parameter: ", para)
# set up loss function
loss_fn = nn.L1Loss()

# set up optimizers (stochastic gradeint descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # learning rate- important hyper parameter set ourselves

# Buildng training & testing loop
# 1. Loop through data
# 2. Forward Pass/Forward Propagation (data moves through forward function(s))
# 3. Calculate loss
# 4. Optimizer zero grad
# 5. Loss Backward - move backwards through the network to calculate gradient of each parameter with respect to loss
# 6. Optimizer Step - Use optimizer to adjust our models arameter to try and improve loss

# epoch one loop through data
epochs = 200
# track different values
epoch_count = []
loss_value = []
test_loss_values = []
# 1. Loop through data
for epoch in range(epochs):
    # set model to train mode
    model_0.train()  # train sets all parameters to require gradients

    # 2. Forward Pass
    y_pred = model_0(X_train)

    # 3. Calculate Loss
    loss = loss_fn(y_pred, y_train)

    # 4. Optimizer zero grad
    optimizer.zero_grad()

    # 5. Perform back propagation on loss w/ respect to parameter of the model
    loss.backward()

    # Step the optimizer (perform gradient descent)
    optimizer.step()

    # print(f'Loss: {loss.item()}')
    model_0.eval()  # turns off gradient tracking

    with torch.inference_mode():  # turns off gradient tracking
        # 1. Forward pass
        test_pred = model_0(X_test)
        # 2. Calculate loss
        test_loss = loss_fn(test_pred, y_test)

    epoch_count.append(epoch)
    loss_value.append(loss.item())
    test_loss_values.append(test_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

plot_prediction(predictions=test_pred)

# plot loss curve
plt.plot(epoch_count, np.array(loss_value), label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test Loss")
plt.title("Training and test loss")
plt.ylabel("Loss")
plt.xlabel("Epoches")
plt.legend()
plt.show()

