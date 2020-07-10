from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim

model = nn.Sequential(nn.Linear(784,128),
	nn.ReLU(),
	nn.Linear(128,64),
	nn.ReLU(),
	nn.Linear(64,10),
	nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

transform = transforms.Compose([transforms.ToTensor(), 
	transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('MNIST_data/', 
	download=False, 
	train=True, 
	transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
	batch_size=64, 
	shuffle=True)

epochs = 10
'''
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		images = images.view(images.shape[0],-1)
		
		optimizer.zero_grad()
		
		logits = model(images)
		loss = criterion(logits, labels)
		
		running_loss+=loss.item()
		
		loss.backward()
		optimizer.step()
	else:
		print("Training losss is : "+str(running_loss/len(trainloader)))
'''
for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		images = images.view(images.shape[0],-1)
		
		optimizer.zero_grad()
		output = model.forward(images)
		loss = criterion(output, labels)
		
		loss.backward()
		optimizer.step()

		running_loss+=loss.item()
	else:
		print("Training losss is : "+str(running_loss/len(trainloader)))


'''
torch.save(the_model, './model/model.pth')
saved_model = torch.load('./model/model.pth')

or

torch.save(the_model.state_dict(), './model/model.pth')
saved_model = nn.Sequential(nn.Linear(784,128),
	nn.ReLU(),
	nn.Linear(128,64),
	nn.ReLU(),
	nn.Linear(64,10),
	nn.LogSoftmax(dim=1))
saved_model.load_state_dict(torch.load('./model/model.pth'))
'''