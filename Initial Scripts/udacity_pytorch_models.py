from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch
import numpy as np
from torch import optim
from skimage import util, io, color, feature, viewer
import helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
	[transforms.RandomResizedCrop(64),
	 transforms.ToTensor()])

train_data = datasets.ImageFolder('./Cat_Dog_data/train', transform=data_transform)
test_data = datasets.ImageFolder('./Cat_Dog_data/test', transform=data_transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

model = models.resnet50(pretrained=True)

#turn off grads for model
for param in model.parameters():
	param.requires_grad = False

calssifier = nn.Sequential(
	nn.Linear(2048, 512),
	nn.ReLU(),
	nn.Dropout(p=0.02),
	nn.Linear(512,2),
	nn.LogSoftmax(dim=1))

model.fc = calssifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.003)
model.to(device)

epochs = 5
steps=0
running_loss=0
print_every=5

for epoch in range(epochs):
	for images, labels in trainloader:
		steps+=1
		
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		logps = model(images)
		loss = criterion(logps, labels)
		optimizer.step()
		running_loss+=loss.item()
		
		if(steps % print_every == 0):
			model.eval()
			test_loss = 0
			accuracy = 0
			
			for images, labels in testloader:
				images, labels = images.to(device), labels.to(device)
				logps = model(images)
				loss = criterion(logps, labels)
				test_loss+=loss.item()
				
				#calc acc
				ps = torch.exp(logps)
				top_ps, top_class = ps.topk(1, dim=1)
				equality = top_class == labels.view(*top_class.shape)
				accuracy += torch.mean(equality.type(torch.FloatTensor))	
				
			print("Epoch {}/{}..".format(epoch+1, epochs),
				"Training Loss: {:.3f}..".format(running_loss/len(trainloader)),
				"Test Loss: {:.3f}..".format(test_loss/len(testloader)))
			running_loss=0
			model.train()