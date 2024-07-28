import torch
import numpy as np

print(np.__version__)
print(torch.__version__)

# tensors - a way to represent data

# scalar
scalar = torch.tensor(7)
# print('scalar in pytorch:', scalar)

# vector
vector = torch.tensor([7, 7])
# print('vector in pytorch:', vector)

# matrix
matrix = torch.tensor([[7, 8],
                       [9, 10]])
# print('matrix in pytorch:', matrix)

# tensor

tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
# print('tensor in pytorch:', tensor)
size = tensor.ndim
# print("The size of this tensor is:", size)

print('\n')

# creating a random tensor
random_tensor = torch.rand(3, 4)
# print(random_tensor)


# create random tensor with similar shape to image tensor
rand_img_size_tensor = torch.rand(size=(3, 224, 224))
size1 = rand_img_size_tensor.ndim
# print(size1)

# create a range of tensors and tensor like
# 1. torch.range
rnge = torch.range(0, 10)
print(rnge)

# 2. torch.arange
rnge1 = torch.arange(0, 10)
print(rnge1)

# tensor datatypes
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # what datatype is the tensor (float32 or float16)
                               device=None,  # what device tensor is on (gpu, tpu, cpu)
                               requires_grad=False)  # wether ort not to track graident with this tensor operation
print(float_32_tensor)

some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Data type of this tensor: {some_tensor.dtype}")
print(f"Shape of tensor:", {some_tensor.shape})
print(f"Device tensor is on:", some_tensor.device)

# Manipulating Tensors
tensor2 = torch.tensor([1, 2, 3])
print(tensor2)
print("This tensor added with 10:", tensor2 + 10)
print("This tensor multiplied with 10:", tensor2 * 10)
print("This tensor subtracted with 10:", tensor2 - 10)
print("This tensor divided with 10:", tensor2 / 10)

# Another way of Manipulating
multi = torch.mul(tensor2, 10)
print("This tensor multiplied with 10 (another way):", multi)

sub = torch.sub(tensor2, 10)
print("This tensor subtracted with 10 (another way):", sub)

# Matrix Multiplication aka dot product
tensor3 = torch.tensor([1, 2, 3])
print("The original tensor:", tensor3)
print(tensor3, "*", tensor3)
print("Dot multiplication of the tensor is:", tensor3 * tensor3)
matrix_mul = torch.matmul(tensor3, tensor3)
print("The matrix mult is:", matrix_mul)

# longer way to do this
value = 0
for i in range(len(tensor3)):
    value += tensor3[i] * tensor3[i]
print(value)

# another way
another_multiply = tensor3 @ tensor3
print(another_multiply)

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
trans = tensor_B.T
print(trans)
multiply_matrix = torch.matmul(tensor_A, trans)
print(multiply_matrix)
# print(tensor_A @ trans)

#  Min, Max, Mean, and Sum
x = torch.arange(0, 100, 10)
print(x)

# min
mini = torch.min(x)
print("The min of the tensor is:", mini)

# max
maxs = torch.max(x)
print("The max of the tensor is:", maxs)

# mean
# this needs an input dtype -> mean = torch.mean(x) changes to:
mean = torch.mean(x.type(torch.float32))
print("Mean of the tensor is:", mean)

# find sum
sums = torch.sum(x)
print("The sum of the tensor is:", sums)

#  Finding the position of min and max
min_pos = x.argmin()
print(min_pos)

max_pos = x.argmax()
print(max_pos)

# Reshaping Tensors (add dimensions)
x1 = torch.arange(1., 10.)
print(x1, x1.shape)

# rehspaped adding dimensions
x_reshaped = x1.reshape(1, 9)  # has to be compatible with original size
print(x_reshaped)

# change the view
z = x1.view(1, 9)
print('View', z)

# changing
change = z[:, 0] = 5
print(z)

# stack tensors on top of each other
x_stack = torch.stack([x1, x1, x1, x1], dim=0)
print("Stacked:", x_stack)

# torch.squeeze() - removes all single dimensions from a target tensor
squeeze = x_reshaped.squeeze()
print(squeeze)

print(f"Previous Tensor for x_reshaped: {x_reshaped}")
print(f"Previous shape for x_reshaped: {x_reshaped.shape}")

print(f"New Tensor for x_reshaped with squeeze: {squeeze}")
print(f"New Tensor shape for x_reshaped with squeeze: {squeeze.shape}")

# torch.unsqueeze() - adds a single dimensions from a target tensor at a specific dimension
print(f"Previous squeezed tensor: {squeeze}")
print(f"Previous squeezed shape: {squeeze.shape}")

# add extra dimension with unsqueezed
unsqueeze = squeeze.unsqueeze(dim=0)
print(f"New unsqueezed tensor: {unsqueeze}")
print(f"New unsqueezed tensor shape: {unsqueeze.shape}")

print('\n')

# torch.permute - rearranges the dimensions of a target tensor in specified order (used in images)
x_original = torch.rand(size=(224, 224, 3))  # [Height, Width and Color_channels]
# permute original tensor to rearrange axis (or dim) order
x_permute = x_original.permute(2, 0, 1)  # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New Shape: {x_permute.shape}")  # [color_channels, height, width]

print('\n')
# Selecting data (indexing)

x2 = torch.arange(1, 10).reshape(1, 3, 3)
print(x2, x2.shape)

# index the tensor
x2_index = x2[0]
print("Index the new tensor", x2_index)

# index on middle bracket (dim=1)
x2_mid = x2[0][0]
print("Middle bracket", x2_mid)

# index on most inner bracket (last dim)
x2_in = x2[0][1][1]
print("Last dimension", x2_in)

# use ':' to select all of a target dimension
x2_colon = x2[:, 0]
print("use ':' to select all", x2_colon)

# all values of 0th and 1st dimensions but only 1 of 2nd dimension
x2_value1 = x2[:, :, 1]
print(x2_value1)

# all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
x2_value2 = x2[:, 1, 1]
print(x2_value2)

# get index 0 of 0th and 1st dimension and all values of 2nd dimension

x2_value3 = x2[0, 0, :]
print(x2_value3)

# index to return 9
print(x2[0][2][2])

# return 3, 6, 9
print(x2[:, :, 2])

# PyTorch tensor and Numpy

array = np.arange(1.0, 8.0)
tensor_py = torch.from_numpy(array)
print("This is the numpy array", array)
print("This is the array converted to tensor:", tensor_py)
print("\n")

# reproductability (taking the random out of random) start with random numbers-> tensor operations -> update random
# numbers to try and make them better representations of the data (repeat)
new_tensorA = torch.rand(3, 4)
print(new_tensorA)
new_tensorB = torch.rand(3, 4)
print(new_tensorB)
print(new_tensorA == new_tensorB)

print("\n")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)




