import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import cv2 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set(color_codes=True)

#read the image
image = cv2.imread('image1.jpg')
#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Original Image: ")

import numpy as np
filter_vals = np.array([[-1, -1, 1, 2], [-1, -1, 1, 0], [-1, -1, 1, 1], [-1, -1, 1, 1]])
print('Filter shape: ', filter_vals.shape)
# Neural network with one convolutional layer and four filters
class Net(nn.Module):
 
 def __init__(self, weight): #Declaring a constructor to initialize the class variables
 super(Net, self).__init__()
 # Initializes the weights of the convolutional layer to be the weights of the 4 defined filters
 k_height, k_width = weight.shape[2:]
 # Assumes there are 4 grayscale filters; We declare the CNN layer here. Size of the kernel equals size of the filter
 # Usually the Kernels are smaller in size
 self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
 self.conv.weight = torch.nn.Parameter(weight)
 
 def forward(self, x):
 # Calculates the output of a convolutional layer pre- and post-activation
 conv_x = self.conv(x)
 activated_x = fn.relu(conv_x)
# Returns both layers
 return conv_x, activated_x
# Instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
# Print out the layer in the network
print(model)
