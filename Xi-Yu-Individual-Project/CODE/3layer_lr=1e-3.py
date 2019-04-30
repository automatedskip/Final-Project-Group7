from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable

print(os.listdir('./natural_images'))

# Visualising Data
classes = []
img_classes = []
n_image = []
height = []
width = []
dim = []

# Using folder names to identify classes
for folder in os.listdir('./natural_images'):
    classes.append(folder)

    # Number of each image
    images = os.listdir('./natural_images/' + folder)
    n_image.append(len(images))

    for i in images:
        img_classes.append(folder)
        img = np.array(Image.open('./natural_images/' + folder + '/' + i))
        height.append(img.shape[0])
        width.append(img.shape[1])
    dim.append(img.shape[2])

df = pd.DataFrame({
    'classes': classes,
    'number': n_image,
    "dim": dim
})
print("Random heights:" + str(height[10]), str(height[123]))
print("Random Widths:" + str(width[10]), str(width[123]))
df

image_df = pd.DataFrame({
    "classes": img_classes,
    "height": height,
    "width": width
})
img_df = image_df.groupby("classes").describe()
print(img_df)

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def imshow_tensor(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image
#-------------------------------
# Hyperparameters Defined
#-------------------------------
batch_size = 128

#input_size = 1000
#hidden_size = 120
#num_classes = 8
num_epochs = 5
#batch_size = 64
learning_rate =1e-3
#----------------------------------------------------------------------------
all_data = datasets.ImageFolder(root='./natural_images')
train_data_len = int(len(all_data)*0.8)
valid_data_len = int((len(all_data) - train_data_len)/2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['val']
test_data.dataset.transform = image_transforms['test']
print(len(train_data), len(val_data), len(test_data))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

trainiter = iter(train_loader)
images, labels = next(trainiter)
print(images.shape, labels.shape)

#-----------------------------------------------------------------------------------------------
# Class for 2 convolutinal - layer network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.fc = nn.Linear(28 * 28 * 64, 8)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out

#-------------------------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
              % (epoch + 1, num_epochs, i + 1, (len(train_data) // batch_size) + 1, loss.item()))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in train_loader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on Training Data: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
correct = 0
total = 0
for images, labels in val_loader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on Validation Data: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------

torch.cuda.empty_cache()