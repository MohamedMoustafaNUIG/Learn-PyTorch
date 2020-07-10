from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim


def load_data():
	transform = transforms.Compose([transforms.ToTensor(), 
		transforms.Normalize((0.5,), (0.5,))])

	trainset = datasets.FashionMNIST('MNIST_data/', 
		download=False, 
		train=True, 
		transform=transform)

	trainloader = torch.utils.data.DataLoader(trainset, 
		batch_size=64, 
		shuffle=True)

	return trainloader

def explore_data():
	data = load_data()
	images, labels = next(iter(data))
	print(images.shape)
	print(labels.shape)
	
def train(data):
	model = nn.Sequential(
		nn.Linear(784,392),
		nn.ReLU(),
		nn.Linear(392,784),
		nn.ReLU(),
		nn.Linear(784,98),
		nn.ReLU(),
		nn.Linear(98,49),
		nn.ReLU(),
		nn.Linear(49,10),
		nn.LogSoftmax(dim=1)
		)

	if torch.cuda.is_available():
		model.cuda()

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.05)

	epochs = 100

	for e in range(epochs):
		running_loss = 0
		for images, labels in data:
			if torch.cuda.is_available():
				#flatten images 64x1x28x28 => 64x784
				images = images.view(images.shape[0],-1).cuda()
				labels=labels.cuda()
			else:
				#flatten images 64x1x28x28 => 64x784
				images = images.view(images.shape[0],-1)

			optimizer.zero_grad()
			output = model.forward(images)
			loss = criterion(output, labels)
			
			loss.backward()
			optimizer.step()

			running_loss+=loss.item()
		else:
			print("Training loss is : "+str(running_loss/len(data)))

	return model

def store_model(model, name, weights=False):
	if weights:
		torch.save(model.state_dict(), './model/'+name+'.pth')
	else:
		torch.save(model, './model/'+name+'.pth')

def load_model(name, weights=False):
	if weights:
		model = nn.Sequential(
			nn.Linear(784,392),
			nn.ReLU(),
			nn.Linear(392,784),
			nn.ReLU(),
			nn.Linear(784,98),
			nn.ReLU(),
			nn.Linear(98,49),
			nn.ReLU(),
			nn.Linear(49,10),
			nn.LogSoftmax(dim=1)
			)
		model.load_state_dict(torch.load('./model/'+name+'.pth'))
	else:
		model = torch.load('./model/model.pth')

	return model

store_model(train(load_data()), 'mnist_fashion', False)