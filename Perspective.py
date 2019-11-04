import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
def transform(): 
    img = cv2.imread("/Users/hcy/Desktop/ex.jpeg") 
    h, w = img.shape[:2] 
    pts1 = np.float32([[455, 350], [540, 350], [300, h], [730, h]]) 
    pts2 = np.float32([[0, 0], [1024, 0], [0, h], [1024, h]]) 
    M = cv2.getPerspectiveTransform(pts1, pts2) 
    img2 = cv2.warpPerspective(img, M, (w, h)) 
    cv2.circle(img, (455, 350), 20, (255, 0, 0), -1) 
    cv2.circle(img, (540, 350), 20, (0, 255, 0), -1) 
    cv2.circle(img, (300, h), 20, (0, 0, 255), -1) 
    cv2.circle(img, (730, h), 20, (0, 0, 0), -1) 
    cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
    cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
    cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
    cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
    plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
    plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
    plt.show() 

def findContour():
    img = cv2.imread("/Users/hcy/Desktop/ex.jpeg", cv2.COLOR_BGR2GRAY)

    # Call cv2.findContours
    contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CV_CHAIN_APPROX_NONE)

    lst_intensities = []

    for i in range(len(contours)):
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        pts = np.where(cimg == 255)
        list_intensities.append(img[pts[0], pts[1]]) 

#transform()
#findContour()

def convex():
    img = cv2.imread("/Users/hcy/Desktop/ex.jpeg")
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    print(defects)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(img,start,end,[0,255,0],2)
        cv2.circle(img,far,5,[0,0,255],-1)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#convex()

def point():
    
