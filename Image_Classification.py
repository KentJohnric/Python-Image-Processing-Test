import cv2
import seaborn as sns 
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set(color_codes=True)

# Read the Image
image = cv2.imread('image1.jpg') #--read() helps in loading an image

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#as opencv loads in BGR format by default, we want to show it in RGB
plt.show()

image.shape