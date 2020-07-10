from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import helper


model = torch.load('./model/model.pth')

transform = transforms.Compose([transforms.ToTensor(), 
	transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('MNIST_data/', 
	download=False, 
	train=True, 
	transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
	batch_size=1, 
	shuffle=True)

images, labels = next(iter(trainloader))
images = images.view(images.shape[0],-1)
output = model.forward(images)
ps = F.softmax(output, dim=1)
values, indices = torch.max(ps, 1)
print(indices)
helper.view_classify(images.view(1,28,28), ps)