import torch
import numpy as np

def activation(x):
	return 1/(1+torch.exp(-x))

'''
#Generate some data
torch.manual_seed(7)
features = torch.randn((1,5))

#Initialise weights and biases
weights = torch.randn_like(features)
bias = torch.randn((1,1))

#y = activation(torch.sum(features * weights) + bias)
#y = activation((features * weights).sum() + bias)
y = activation(torch.mm(features, weights.view(5,1)) + bias)
'''

'''
tensor.shape
weights.reshape(a, b) #can return clone (inefficient memory) or new
weights.resize_(a, b) #inplace that can remove elements or add uninitialised elements
weights.view(a, b) #returns new
'''

#Two layer network

#Generate some data
torch.manual_seed(7)
features = torch.randn((1,3))

#NN architecture
n_input = features.shape[1]
n_hidden = 2
n_output = 1


#Initialise weights and biases
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)

print(output)

#memory is shared between arr and tensor
np_arr1 = np.random.randn(4,3)
pt_tensor = torch.from_numpy(np_arr)
np_arr2 = pt_tensor.numpy()

print(pt_tensor)