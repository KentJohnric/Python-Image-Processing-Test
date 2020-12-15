import numpy as np
import cv2
import seaborn as sns 
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set(color_codes=True)

# Read the Image
image = cv2.imread('image1.jpg') #--read() helps in loading an image

#Convert image to grayscale; the second argument is a function of cv2 to convert
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Original Image:")

# 3x3 array for edge detection
mat_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])
mat_x = np.array([[ -1, 0, 1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])
  
#filtered_image = cv2.filter2D(gray, -1, mat_y)
#plt.imshow(filtered_image, cmap='gray')
filtered_image = cv2.filter2D(gray, -1, mat_x)
plt.imshow(filtered_image, cmap='gray')
plt.show()











#filtered_image = cv2.filter2D(gray, -1, mat_x)
#plt.imshow(filtered_image, cmap = 'gray')

#data = np.array(gray)
#flattened = data.flatten()
#np.savetxt('test.txt', data)
#flattened.shape

#as opencv loads in BGR format by default, we want to show it in RGB
#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#plt.show()
#gray.shape