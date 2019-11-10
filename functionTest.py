import Perspective
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

img = cv2.imread("/Users/hcy/Desktop/exex.png")

print(img)

plt.imshow(img)
plt.show()

img2 = Perspective.point(img)

print("#######################")

h, w = img2.shape[:2] 

cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 

#print(img2)
print(img2.shape)

dst = cv2.resize(img2, dsize=(1519, 2062), interpolation=cv2.INTER_AREA)

print(dst.shape)

plt.subplot(1, 2, 1), plt.imshow(img2), plt.title('image') 
plt.subplot(1, 2, 2), plt.imshow(dst), plt.title('perspective') 
plt.show()
