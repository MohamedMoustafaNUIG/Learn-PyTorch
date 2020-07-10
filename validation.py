from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim
from skimage import util, io, color, feature, viewer
import helper

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 392)
		self.fc2 = nn.Linear(392, 784)
		self.fc3 = nn.Linear(784, 392)
		self.fc4 = nn.Linear(392, 784)
		self.fc5 = nn.Linear(784, 392)
		self.fc6 = nn.Linear(392, 196)
		self.fc7 = nn.Linear(196, 98)
		self.fc8 = nn.Linear(98, 49)
		self.fc9 = nn.Linear(49, 10)
		
		self.dropout = nn.Dropout(p=0.2)
		
	def forward(self, x, dropout=False):
		x = x.view(-1, 784)
		if dropout:
			x = self.dropout(nn.ReLU() (self.fc1(x)))
			x = self.dropout(nn.ReLU() (self.fc2(x)))
			x = self.dropout(nn.ReLU() (self.fc3(x)))
			x = self.dropout(nn.ReLU() (self.fc4(x)))
			x = self.dropout(nn.ReLU() (self.fc5(x)))
			x = self.dropout(nn.ReLU() (self.fc6(x)))
			x = self.dropout(nn.ReLU() (self.fc7(x)))
			x = self.dropout(nn.ReLU() (self.fc8(x)))
			x = self.dropout(nn.ReLU() (self.fc9(x)))
		else:
			x = nn.ReLU() (self.fc1(x))
			x = nn.ReLU() (self.fc2(x))
			x = nn.ReLU() (self.fc3(x))
			x = nn.ReLU() (self.fc4(x))
			x = nn.ReLU() (self.fc5(x))
			x = nn.ReLU() (self.fc6(x))
			x = nn.ReLU() (self.fc7(x))
			x = nn.ReLU() (self.fc8(x))
			x = nn.ReLU() (self.fc9(x))
		x = nn.LogSoftmax(dim=1)(x)
		return x
			
def load_data():
	
	train_transform = transforms.Compose([
		transforms.RandomRotation(30, fill=(0,)),
		transforms.ToTensor()
	])
	
	test_transform = transforms.Compose([transforms.ToTensor()])

	train_data = datasets.MNIST(
		root="./MNIST_data/", 
		train=True, 
		download=False, 
		transform=train_transform)
	
	test_data = datasets.MNIST(
		root="./MNIST_data/", 
		train=False, 
		download=False, 
		transform=test_transform)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
	
	return trainloader, testloader

def train_model():
	train, test = load_data()
	model = Classifier()
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)

	epochs = 10
	train_losses, test_losses = [], []

	for e in range(epochs):
		running_loss = 0
		for images, labels in train:
			model.train()
			optimizer.zero_grad()
			output = model.forward(images, True)
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()
			running_loss+=loss.item()

		else:
			test_loss=0
			accuracy=0
			with torch.no_grad():
				for images, labels in test:
					model.eval()
					log_ps = model.forward(images)
					test_loss += criterion(log_ps, labels)

					ps = torch.exp(log_ps)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class == labels.view(*top_class.shape)
					accuracy += torch.mean(equals.type(torch.FloatTensor))

			train_losses.append(running_loss/len(train))
			test_losses.append(test_loss/len(test))

			print("Epoch {}/{}..".format(e+1, epochs),
					"Training Loss: {:.3f}..".format(running_loss/len(train)),
					"Test Loss: {:.3f}..".format(test_loss/len(test)))
				
train_model()