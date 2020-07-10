from torchvision import datasets, transforms
import torch
import numpy as np


transform = transforms.Compose([transforms.ToTensor(), 
	transforms.Normalize((0.5,), (0.5,))
	])

trainset = datasets.MNIST('MNIST_data/', download=False, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, lables = dataiter.next()


def activation(x):
	return 1/(1+torch.exp(-x))

def softmax(x):
	return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1,1)

inputs = images.view(images.shape[0], -1)

n_i = images.shape[1]*images.shape[2]*images.shape[3]
n_h = 256
n_o = 10

w1 = torch.randn(n_i, n_h)
b1 = torch.randn(n_h)

w2 = torch.randn(n_h, n_o)
b2 = torch.randn(n_o)

a1 = activation(torch.mm(inputs, w1)+b1)
a2 = activation(torch.mm(a1, w2)+b2)