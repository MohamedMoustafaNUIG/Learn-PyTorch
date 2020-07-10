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
		self.fc1 = nn.Linear(784,392)
		self.fc2 = nn.Linear(392,784)
		self.fc3 = nn.Linear(784,392)
		self.fc4 = nn.Linear(392,784)
		self.fc5 = nn.Linear(784,98)
		self.fc6 = nn.Linear(98,49)
		self.fc7 = nn.Linear(49,98)
		self.fc8 = nn.Linear(98,49)
		self.fc9 = nn.Linear(49,10)

	def forward(self, x):
		x=x.view(x.shape[0], -1)

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = F.relu(self.fc6(x))
		x = F.relu(self.fc7(x))
		x = F.relu(self.fc8(x))
		
		x = F.log_softmax(self.fc9(x), dim=1)

		return x


def load_data():
	transform = transforms.Compose([transforms.ToTensor(), 
		transforms.Normalize((0.5,), (0.5,))])

	trainset = datasets.FashionMNIST('MNIST_data/', 
		download=False, 
		train=True, 
		transform=transform)

	testset = datasets.FashionMNIST('MNIST_data/', 
		download=True, 
		train=False, 
		transform=transform)

	trainloader = torch.utils.data.DataLoader(trainset, 
		batch_size=64, 
		shuffle=True)

	testloader = torch.utils.data.DataLoader(testset, 
		batch_size=64, 
		shuffle=True)

	return trainloader, testloader

def explore_data():
	test, train = load_data()
	images, labels = next(iter(test))
	print(images.shape)
	print(labels.shape)
	
def train():
	train, test = load_data()
	model = Classifier()
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)

	epochs = 10
	steps=0
	
	train_losses, test_losses = [], []
	
	for e in range(epochs):
		running_loss = 0
		for images, labels in train:
			optimizer.zero_grad()
			log_ps = model(images)
			loss = criterion(log_ps, labels)			
			loss.backward()
			optimizer.step()

			running_loss+=loss.item()
		else:
			test_loss=0
			accuracy=0
			with torch.no_grad():
				for images, labels in test:
					log_ps = model(images)
					test_loss += criterion(log_ps, labels)

					ps = torch.exp(log_ps)
					top_p, top_class = ps.topk(1, dim=1)
					#equals is a tensor with 0=> top and labels dont agree and 1=> top and labels agree
					equals = top_class == labels.view(*top_class.shape)
					#torch mean wont work on byte tensor equals so we need to convert
					accuracy += torch.mean(equals.type(torch.FloatTensor))
			train_losses.append(running_loss/len(train))
			test_losses.append(test_loss/len(test))

			print("Epoch {}/{}..".format(e+1, epochs),
				"Training Loss: {:.3f}..".format(running_loss/len(train)),
				"Test Loss: {:.3f}..".format(test_loss/len(test)))

	store_model(model, 'mnist_fashion', False)

def store_model(model, name, weights=False):
	if weights:
		torch.save(model.state_dict(), './model/'+name+'.pth')
	else:
		torch.save(model, './model/'+name+'.pth')

def load_model(name, weights=False):
	if weights:
		model = Classifier()
		model.load_state_dict(torch.load('./model/'+name+'.pth'))
	else:
		model = torch.load('./model/'+name+'.pth')

	return model

def test(load_data=False):
	model = load_model('mnist_fashion')
	if load_data:
		_, test = load_data()
		dataiter = iter(load_data())
		images, labels = dataiter.next()
		img = images[11].view(1, 784).cuda()
		labels=labels.cuda()
		prediction = torch.exp(model(img.float()))
		torch.Tensor.cpu(labels)
		print(labels[11])
		helper.view_classify(torch.Tensor.cpu(img.view(1,28,28)), torch.Tensor.cpu(prediction), version='Fashion')
	else:
		image = torch.tensor(io.imread(fname='./red_shirt.png', as_gray=True))
		img = image.view(1, 784).cuda()
		prediction = torch.exp(model(img.float()))
		helper.view_classify(torch.Tensor.cpu(img.view(1,28,28)), torch.Tensor.cpu(prediction), version='Fashion')

train()