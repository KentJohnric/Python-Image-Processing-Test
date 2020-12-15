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

data = np.array(gray)
flattened = data.flatten()
np.savetxt('test.txt', data)
flattened.shape

#as opencv loads in BGR format by default, we want to show it in RGB
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

gray.shape