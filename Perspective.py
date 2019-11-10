import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

def point():
    img = cv2.imread("/Users/hcy/Desktop/ex4.jpeg")
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #print(img)
    
    #cv2.imshow('imgray', imgray)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 경계선을 그리고
    cnt = contours[4]
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    maxX = 0
    maxY = 0
    minX = 10000
    minY = 10000
    # 꼭지점 저장되는 곳
    topX = 0
    topY = 0
    bottomX = 0
    bottomY = 0
    leftX = 0
    leftY = 0
    rightX = 0
    rightY = 0

    # 경계선 그린 거에서 꼭지점 좌표 찾는 부분
    for i in cnt:
        if maxX < i[0][0]:
            maxX = i[0][0]
            rightX = i[0][0]
            rightY = i[0][1]
        if maxY < i[0][1]:
            maxY = i[0][1]
            bottomX = i[0][0]
            bottomY = i[0][1]
        if minX > i[0][0]:
            minX = i[0][0]
            leftX = i[0][0]
            leftY = i[0][1]
        if minY > i[0][1]:
            minY = i[0][1]
            topX = i[0][0]
            topY = i[0][1]
        #print(i[0][1])

    # 320, 175 : top
    # 116, 575 : left
    # 857, 447 : right
    # 680, 839 : bottom
    print(maxX)
    print(maxY)
    print(minX)
    print(minY)
    
    right = [rightX, rightY]
    left = [leftX, leftY]
    top = [topX, topY]
    bottom = [bottomX, bottomY]

    print(top)
    print(left)
    print(right)
    print(bottom)

    # 이제 해야될 건 이 좌표로 perspective 해보는 거!!

    h, w = img.shape[:2] 
    pts1 = np.float32([top, right, left, bottom]) 
    pts2 = np.float32([[0, 0], [1024, 0], [0, h], [1024, h]]) 
    M = cv2.getPerspectiveTransform(pts1, pts2) 
    img2 = cv2.warpPerspective(img, M, (w, h)) 
    cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
    cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
    cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
    cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
    cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
    cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
    cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
    cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
    plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
    plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
    plt.show() 
    
    #print(img)
    
    #for
    #cnt = contours[0]

    #leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    #rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    #topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    #cv2.circle(img, leftmost, 10, (0,0,255), -1)
    #cv2.circle(img, rightmost, 10, (0,0,255), -1)
    #cv2.circle(img, topmost, 10, (0,0,255), -1)
    #cv2.circle(img, bottommost, 5, (0,255,255), -1)

    #cv2.imshow('img', img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

point()
