from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim
from skimage import util, io, color, feature, viewer
from matplotlib import pyplot as plt

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(50176,25088)
		self.fc2 = nn.Linear(25088,50176)
		self.fc3 = nn.Linear(50176,12544)
		self.fc4 = nn.Linear(12544,6272)
		self.fc5 = nn.Linear(6272,3136)
		self.fc6 = nn.Linear(3136,1568)
		self.fc7 = nn.Linear(1568,784)
		self.fc8 = nn.Linear(784,196)
		self.fc9 = nn.Linear(196,2)
		
		self.dropout = nn.Dropout(p=0.2)
	def forward(self, x):
		x=x.view(x.shape[0], -1)

		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = self.dropout(F.relu(self.fc3(x)))
		x = self.dropout(F.relu(self.fc4(x)))
		x = self.dropout(F.relu(self.fc5(x)))
		x = self.dropout(F.relu(self.fc6(x)))
		x = self.dropout(F.relu(self.fc7(x)))
		x = self.dropout(F.relu(self.fc8(x)))
		
		x = F.log_softmax(self.fc9(x), dim=1)

		return x


	
def load_data():
	train_transform = transforms.Compose(
		[transforms.Grayscale(num_output_channels=1),
		 transforms.RandomRotation(30, fill=(0,)),
		 transforms.RandomResizedCrop(224),
		 transforms.RandomHorizontalFlip(),
		 transforms.ToTensor()])

	test_transform = transforms.Compose(
		[transforms.Grayscale(num_output_channels=1),
		 transforms.Resize(255),
		 transforms.CenterCrop(224),
		 transforms.ToTensor()])

	train_data = datasets.ImageFolder('./Cat_Dog_data/train', transform=train_transform)
	test_data = datasets.ImageFolder('./Cat_Dog_data/test', transform=test_transform)

	trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
	return trainloader, testloader

def store_model(model, name, weights=False):
	if weights:
		torch.save(model.state_dict(), './model/'+name+'.pth')
def train():
	train, test = load_data()
	tain = train
	test = test
	model = Classifier()
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.003)
	
	epochs = 10
	steps=0
	
	train_losses, test_losses = [], []
	
	for e in range(epochs):
		running_loss = 0
		for images, labels in iter(train):
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
				for images, labels in iter(test):
					images=images
					labels=labels
					model.eval()
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

	store_model(model, 'cat_or_dog', False)
	
	
train()