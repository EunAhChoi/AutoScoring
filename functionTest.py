import Perspective
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

img_original = cv2.imread("/Users/hcy/Desktop/answerNumber.jpeg")
img = cv2.imread("/Users/hcy/Desktop/exex.png")

original_shape = img_original.shape[:2]
img_shape = img.shape[:2]
#print(original_shape)
#print(img_shape)
plt.imshow(img)
plt.show()


#original_shape = (1509, 2062)
#img_shape = (500, 500)

img2 = Perspective.point(img_original, original_shape)

print("#######################")

h, w = img2.shape[:2] 

cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 

#print(img2)
print(img2.shape)

#dst = cv2.resize(img2, dsize=(1519, 2062), interpolation=cv2.INTER_AREA)

#print(dst.shape)

#plt.subplot(1, 2, 1), plt.imshow(img2), plt.title('image') 
#plt.subplot(1, 2, 2), plt.imshow(dst), plt.title('perspective') 

plt.imshow(img2)
plt.show()