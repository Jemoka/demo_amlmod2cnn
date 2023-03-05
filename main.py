"""
main.py
An example CNN implementation for the CIFAR-10 task
"""

#### Imports ####
# we begin by importing the prerequisite tools
# PyTorch!
from operator import xor
from os import access
import torch
import torch.nn as nn
# Data Ops!
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
# Images!
from PIL import Image
# Loading bar. See below where it is used
from tqdm import tqdm

#### Constants ####
# It is generally a good idea to leave the constants up top so you
# can tune the model easily.
# Usually, Python's conventions is that the constants are ALL_CAPS

TRAIN_SPLIT = 0.9
EPOCHS = 3
BATCH_SIZE = 16
LR = 3e-3

SAMPLE_COUNT = 50000 # this is exogenously known by checking the ./data folder

#### Data Ops ####
# The folder structure is as expected from the original worksheet
# barring the mishap we had a while ago.
#
# We have ./data/1.png ... ./data/50000.png, which are the raw input images
# We also have ./y.csv, which are the labels of shape (|index|label|)

# let's load the x data first, because that's a little trickier
x = []
for i in range(1, SAMPLE_COUNT+1): # remember we are 1 indexed
    # open the image
    img = Image.open(f"./data/{i}.png")
    # convert into numpy array
    arr = np.array(img)
    # PyTorch expects things channel first, so we swap the axes
    arr = arr.swapaxes(0, -1)
    # append array to the larger list
    x.append(arr)
    # close image
    img.close()
x = np.array(x) # this array should be of shape (50000, 3, 32, 32)
                # because (SAMPLE_COUNT, RGB, x_size, y_size)
# we now normalize the input by dividing everything by 255
x = x/255

# let's then load the y data, ostensibly much simpler
y = pd.read_csv("./y.csv", index_col=0) # we set index_col=0 because the first column is a sequential index

# the data is already pre-shuffled, so we will not shuffle it again
# but if you are shuffling it, here's where you can do it

#### Encoding ####
# next up, we will one-hot encode the labels to yield an array of (50000, 10) 
# to do this, we will enlist the help of `collections.defaultdict`
# the way that the defaultdict works is that, instead of failing when a new
# key is indexed, it will call a function that is in its constructor to
# initialize the value of that key
# we will use this technique to assign a unique number to each label

label_ids = defaultdict(lambda:len(label_ids))

# this defaultdict will generate a number that counts up for every new
# key it discovers, and return the previous number it has already generated
# if it has already seen the number. i.e.:
#
# >>> label_ids["frog"] => 0
# >>> label_ids["truck"] => 1
# >>> label_ids["truck"] => 1
# >>> label_ids["deer"] => 2
# >>> label_ids["frog"] => 0
# 
# etc. etc. It will never crash on a new key, instead, it will just generate
# a new number that counts up.
#
# Don't worry about why this works if you don't want. You can also one-hot
# encode with `sklearn.preprocessing.OneHotEncoder` as we had done before.
#
# either way, for each element of the labels we will one-hot encode it by
# seeding an array of `0`s and setting the unique element that's activated
# to 1

y_enc = []

for row in y.index:
    # seed an zero array
    arr = np.zeros(10) # we have 10 classes
    # get what label it is 
    label_str = y.loc[row].label # => "deer", etc.
    # set the correct index of arr to 1 by making label_ids
    # compute the ID of the string label, then setting that
    # location to be 1
    arr[label_ids[label_str]] = 1
    # append
    y_enc.append(arr)

y_enc = np.array(y_enc) # as expected, this array should be of shape
                        # (50000, 10)

#### Splitting and Batching ####
# now we compute the train/val split
train_count = int(TRAIN_SPLIT*SAMPLE_COUNT)
val_count = SAMPLE_COUNT-train_count # validation split is just the sample count minues train count

# recall that our data is already pre-shuffled, so we can just
# parcel out the first n-values training, the rest for validation, etc.
train_x = x[:train_count] 
train_y = y_enc[:train_count] 

val_x = x[train_count:] 
val_y = y_enc[train_count:] 

# we will now cast everything into a PyTorch tensor of
# the float type, and group them into TensorDatasets 
train_dataset = TensorDataset(torch.tensor(train_x).float(),
                              torch.tensor(train_y).float())
val_dataset = TensorDataset(torch.tensor(val_x).float(),
                            torch.tensor(val_y).float())

# Great, we now create a DataLoader for the training data,
# batching the data as well.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

#### Model Definition ####
# It's time for the exciting part! Defining the model.

class CNNModel(nn.Module):

    # to define the network
    def __init__(self):
        # initialize superclass
        super().__init__()

        # CONVOLUTIONAL PART
        # remember the overall idea here is to
        # project small kernels into large space,
        # then project large kernels into smaller space

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2) # stride, if 1, can be omitted
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4) # we now up the kernel size slightly
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8)
        self.pool2 = nn.MaxPool2d(4) # then we pool to extract info again

        # then we flatten
        self.flat = nn.Flatten()

        # then, we parcel out the linear layers
        self.dense1 = nn.Linear(16, 64) # this value 16 comes from guessing-checking the output 
                                        # and seeing, after flattening, where PyTorch crashes
                                        # with MatMul errors
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 10)

        # activation funcitons
        self.relu = nn.ReLU() # learned with the class that ReLU can be recycled!
        self.softmax = nn.Softmax(1) # 1 is because we are activating the first (not batch) dimension

    # to pass input through the network
    def forward(self, x):
        net = self.relu(self.conv0(x))
        net = self.relu(self.conv1(net))
        net = self.pool1(net)

        net = self.relu(self.conv2(net))
        net = self.relu(self.conv3(net))
        net = self.pool2(net)

        net = self.flat(net)

        net = self.relu(self.dense1(net))
        net = self.relu(self.dense2(net))
        net = self.relu(self.dense3(net))
        net = self.softmax(self.dense4(net))

        return net

# initialize the model!
model = CNNModel()
# and the loss function
cross_entropy = nn.CrossEntropyLoss()
# and the optimizer
optim = torch.optim.Adam(model.parameters(), lr=LR)

#### Train Loop ####

for e in range(EPOCHS):
    print(f"Training epoch {e}!")

    for x,y in tqdm(iter(train_loader), total=len(train_loader)): # what's this tqdm business?
                                                                  # wrapping a loop with tqdm will print a fancy loading
                                                                  # bar on screen which you can use to track progress
                                                                  # of training a batch. It is transparent in all other respects
                                                                  # and will just dump the output of iter(train_loader)
                                                                  # so this loop is effectively just `for x,y in iter(train_loader):`
        # compute output
        output = model(x)

        # calculate loss
        loss = cross_entropy(output, y)

        # backprop operations
        loss.backward()
        optim.step()
        optim.zero_grad()

        # calculate accuracy
        # torch.argmax(*, axis=1) essentially un-onehot encodes
        # computing the *INDICIES* for predictions and targets
        predictions = torch.argmax(output, axis=1)
        targets = torch.argmax(y, axis=1)

        # accuracy, then, checks if the two are equal, then
        # add up all the "True" (corresponding to 1) in that list
        # Dividing this by the batch size will then give us the
        # mean accuracy in that batch.
        accuracy = (sum(predictions == targets)/BATCH_SIZE).item()

    print(f"Done with epoch {e}! Last accuracy {accuracy}")



