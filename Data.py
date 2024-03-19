import os
import pickle
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from sklearn import model_selection
from torch.utils.data import TensorDataset, Dataset

path = r"C:\Users\kiran.kammili\Downloads\vehicle_classification"
objects = ["Bicycle", "Bus", "Car", "Motorcycle", "NonVehicles", "Taxi", "Truck", "Van"]

data = []


# This will format the data in a way the NN will understand #
# We know have all the data stored inside an array #
# The data is stored in this format -> [pixels], class represented by int #

def create_data():
    for object in objects:

        new_path = os.path.join(path, object)
        class_num = objects.index(object)

        for img in os.listdir(new_path):
            # We want the image to be in gray scale because it won't make a difference #
            # The different objects come in all colors so that's why we set it to grayscale #
            img_array = cv2.imread(os.path.join(new_path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (64, 64))
            data.append([new_array, class_num])
            plt.show()


create_data()
x = []
y = []

random.shuffle(data)

# Reshape tensors for neural network input #
for pixels, label in data:
    # This allows us to split the data into inputs and outputs #
    x.append(pixels)
    y.append(label)

# We do -1 because it encompasses all the instances of data #
# 64 by 64 because of the count of pixels #
# We finally do 1 because we are in gray scale #
x = np.array(x).reshape(-1, 64, 64, 1)
y = np.array(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

test_loader = [x_test, y_test]
train_loader = [x_train, y_train]

classes = ["Bicycle", "Bus", "Car", "Motorcycle", "NonVehicles", "Taxi", "Truck", "Van"]

epochs = 10
batchSize = 32
lr = 0.001


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
])


model = CNN()

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train_tensor = torch.from_numpy(x_train).long()
y_train_tensor = torch.from_numpy(y_train).long()

# Create a DataLoader if needed
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

i = 0
for epoch in range(10):

    for i in range(len(train_dataset)):

        image, label = train_dataset[i]

        image = transformer(image)

        result = model(image)
        loss = lossFunction(result, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{10}], Loss: {loss.item():.4f}')

    i += 1

print("Done!")
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batchSize):
            label = labels[i]
            pred = predicted[i]

            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the network: {acc} %')

for i in range(8):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc} %')
