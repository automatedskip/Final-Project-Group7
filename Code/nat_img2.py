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
learning_rate =1e-4
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
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(56 * 56 * 32, 8)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#-------------------------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()

#--------------------------------------------------
niter =50
test_interval = 100
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
#--------------------------------------------------
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

        # make track of train_loss
        train_loss[i] = loss.item()

    if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, loss.item()))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
#acc_list = []



for images, labels in test_loader:
    images = Variable(images).cuda()
    labels = Variable(labels).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    #acc_list.append(correct/total)
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the 690 test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')

#----------------------------------------
# Plot accuracy and loss
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')
plt.savefig('loss_2.png')
'''
for it in range(niter):
    #solver.step(1)  # SGD by Caffe

    # store the train loss
    #train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc
'''


# ---------------------------------------
ya = pd.Series(labels, name = 'Actual')
yp = pd.Series(predicted, name = 'Predicted')
df_confusion = pd.crosstab(ya,yp, rownames = ['Actual'],colnames = ['Predicted'],
                           margins = True)

print(df_confusion)

class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))

for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1, 28 * 28 * 5376, 24))
    #images = Variable(images.unsqueeze(0))
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for u in range(8):
    acc = 100*class_correct[u] / class_total[u]
    miss_class = 100* (class_total[u] - class_correct[u])/ class_total[u]
    print('Accuracy of %5s : %2d %%' % (classes[u], acc))
    print('Miss Class of %5s : %2d %%' % (classes[u], miss_class))

plt.figure(2)
plt.plot(test_interval * np.arange(len(acc)), acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')
plt.savefig('accu1.png')
'''
for u in range(8):
    acc1[u] = 100*correct[u]/total[u]
    miss_class[u] = 100*(correct[u]- correct[u]/ total[u])
    print('Accuracy of %5s : %2d %%' % (classes[u], acc1))
    print('Miss Class of %5s : %2d %%' % (classes[u], miss_class))
#----------------------------------------

def evaluate(model, test_loader, criterion):
    classes = []
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            output = model(data)

            for pred, true in zip(output, labels):
                _, pred = pred.unsqueeze(0).topk(1)
                correct = pred.eq(true.unsqueeze(0))
                acc_results[i] = correct.cpu()
                classes.append(model.idx_to_class[true.item()+1])
                i += 1

    results = pd.DataFrame({
        'class': classes,
        'results': acc_results
    })
    results = results.groupby(classes).mean()

    return results

evaluate(cnn, test_loader, criterion)

#for it in range(niter):
    #trainiter = iter(train_loader)
    #images, labels = next(trainiter)
    #print(images.shape, labels.shape)
    #train_loss[it] = trainiter
    #solver.step(1)  # SGD by Caffe

    # store the train loss
    #train_loss[it] = solver.net.blobs['loss'].data
    #solver.test_nets[0].forward(start='conv1')

    #if it % test_interval == 0:
        #acc=solver.test_nets[0].blobs['accuracy'].data
        #print 'Iteration', it, 'testing...','accuracy:',acc
        #test_acc[it // test_interval] = acc

#plt.figure(1)
#plt.semilogy(np.arange(niter), train_loss)
#plt.xlabel('Number of Iteration')
#plt.ylabel('Training Loss Values')
#plt.title('Training Loss')

#plt.figure(2)
#plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
#plt.xlabel('Number of Iteration')
#plt.ylabel('Test Accuracy Values')
#plt.title('Test Accuracy')


#the_model = TheModelClass(*args, **kwargs)
#cnn1 = torch.load('./cnn.pkl')
#cnn1.eval()
'''