from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim

class Network(nn.Module):
	def __init__(self):
		super().__init__()

		#Inputs to hidden layer linear transformation
		self.hidden = nn.Linear(784, 256)

		#Output layer, 10 units - one per digit
		self.output = nn.Linear(256, 10)

		#Define sigmoid activation and softmax output
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		#Pass input tensor through each operation
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.output(x)
		x = self.softmax(x)

		return x

class Network_C(nn.Module):
	def __init__(self):
		super().__init__()

		#Inputs to hidden layer linear transformation
		self.hidden = nn.Linear(784, 256)

		#Output layer, 10 units - one per digit
		self.output = nn.Linear(256, 10)

	def forward(self, x):
		#Pass input tensor through each operation
		x = F.sigmoid(self.hidden(x))
		x = F.softmax(self.output(x), dim=1)

		return x

class Network_Dynamic(nn.Module):
	def __init__(self, dims, activations, loss, opt, lr):
		super().__init__()

		layers = []
		activations_list = []
		modules = []

		for i in range(0, len(dims)-1):
			layers.append(nn.Linear(dims[i], dims[i+1]))
		
		for i in range(0, len(activations)):
			if activations[i] == 'relu':
				activations_list.append(nn.modules.activation.ReLU())
			elif activations[i] == 'sigmoid':
				activations_list.append(nn.modules.activation.Sigmoid())
			elif activations[i] == 'softmax':
				activations_list.append(nn.modules.activation.Softmax(dim=1))
			elif activations[i] == 'logsoftmax':
				activations_list.append(nn.modules.activation.LogSoftmax(dim=1))
			elif activations[i] == 'tanh':
				activations_list.append(nn.modules.activation.Tanh())

		for i in range(0, len(layers)):
			modules.append(layers[i])
			if i < len(activations_list):
				modules.append(activations_list[i])

		self.model = nn.Sequential(*modules)

		self.model.parameters = 
		if loss=="cross_entropy":
			self.loss = nn.CrossEntropyLoss()
		elif loss=="nlll":
			self.loss = nn.NLLLoss()

		if opt == 'sgd':
			self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

	def train(self, x, y, epochs):
		for i in range(0, epochs):
			print(self.model[0].weight)
			yhat = self.model(x)
			cost = self.loss(x, y)
			cost.backward()
			self.optimizer.zero_grad()
			self.optimizer.step()

class NetworkDynamic(nn.Sequential):
	def __init__(self, dims, activations, loss, opt, lr):
		super().__init__()


transform = transforms.Compose([transforms.ToTensor(), 
	transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('MNIST_data/', 
	download=False, 
	train=True, 
	transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
	batch_size=64, 
	shuffle=True)

images, lables = next(iter(trainloader))

inputs = images.view(images.shape[0], -1)

layer_dims = [784, 128, 64, 10]
h_layer_activations = ['relu', 'relu','logsoftmax']
model = Network_Dynamic(layer_dims, 
	h_layer_activations, 
	'nlll',
	'sgd',
	0.01)
model.train(inputs, lables, 1)