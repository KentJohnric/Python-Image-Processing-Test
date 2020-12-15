import numpy as np
import cv2
import seaborn as sns 
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set(color_codes=True)
def visualization_layer(layer, n_filters= 4):
    
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # Grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
# Read the Image
image = cv2.imread('image1.jpg') #--read() helps in loading an image

#Convert image to grayscale; the second argument is a function of cv2 to convert
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Original Image:")
#Displat original Image
plt.imshow(gray, cmap='gray')
#Visualize all the filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left = 0, right = 1.5, bottom = 0.8, top = 1, hspace = 0.5, wspace = 0.05)

for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks = [], yticks = [])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filer %s' %str(i+1))

#convert the image into an input tensor
gray_img_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(1)
#print(type(gray_img_tensor))

#print(gray_img_tensor)

#Get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model.forward(gray_img_tensor.float())

#Visualize the output of a convolutional layer
visualization_layer(conv_layer)










#data = np.array(gray)
#flattened = data.flatten()
#np.savetxt('test.txt', data)
#flattened.shape

#as opencv loads in BGR format by default, we want to show it in RGB
#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#plt.show()
#gray.shape