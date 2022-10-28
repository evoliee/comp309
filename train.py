# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import splitfolders
from pathlib import Path
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Use GPU
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Global Variables
SEED = 309

IMG_SIZE = 300
ROT_SIZE = 90
TRAIN_PATH = './traindata'
TEST_PATH = './testdata'
BATCH = 16

LEARNING_RATE = 0.001
MOMENTUM = 0.9
N_EPOCHS = 80

KERNEL_SIZE = 3
PADDING = 0
STRIDE = 1

CHANNEL_OUT_1 = 32
CHANNEL_OUT_2 = 64
N_CLASSES = 3
N_CHANNELS = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

# train a given model 
# also from  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, surprise
def trainModel(model, trainLoadern):
    criterionLossFunction = nn.CrossEntropyLoss() # loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
    
        for i, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) # BUG [?]
            loss = criterionLossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
                
# Models
# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from datetime import datetime

class MLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        CHANNEL_OUT_1 = 32
        CHANNEL_OUT_2 = 64
        N_CLASSES = 3
        N_CHANNELS = 3    
        
        self.fc1 = nn.Linear(270000, CHANNEL_OUT_2)         #dont change this nr idk why??? but it won't work without???
        self.fc2 = nn.Linear(CHANNEL_OUT_2, CHANNEL_OUT_1)
        self.fc3 = nn.Linear(CHANNEL_OUT_1, N_CLASSES)
       
        
    def forward(self, x):
       # x = F.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(datetime.now().strftime('%H:%M'))
        return x

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, CHANNEL_OUT_1, KERNEL_SIZE)
        self.conv2 = nn.Conv2d(CHANNEL_OUT_1, CHANNEL_OUT_2, KERNEL_SIZE)
        self.pool  = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(341056, CHANNEL_OUT_2)                    #dont change this nr idk why??? but it won't work without???
        self.fc2 = nn.Linear(CHANNEL_OUT_2, CHANNEL_OUT_1)
        self.fc3 = nn.Linear(CHANNEL_OUT_1, N_CLASSES)
       
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = nn.Softmax(x) # do it essentially propability between 0 and 1 for each class
        #print(datetime.now().strftime('%H:%M'))
        return x

# Perform Data Split
# splitfolders.ratio('./origData', output="./", seed=SEED, ratio=(.7, 0,0.3)) 

trainDataTransform = transforms.Compose( [
                         transforms.Resize((IMG_SIZE,IMG_SIZE)),
                         transforms.RandomRotation(ROT_SIZE),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor() ])

train_folder = torchvision.datasets.ImageFolder(TRAIN_PATH, transform=trainDataTransform)
trainLoader = torch.utils.data.DataLoader(train_folder, shuffle=True, batch_size = BATCH)
classes = ('cherry', 'strawberry', 'tomato')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = CNN1().to(device)
trainModel(net, trainLoader)
totalAccuracy, classAccuracy = accuracy(net, testLoader)

# save model
PATH = './model.pth'
torch.save(net.state_dict(), PATH)