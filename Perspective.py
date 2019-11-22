import numpy as np 
import cv2  
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def point(img, shape):
    
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    #img = cv2.line(img, (0,0), (shape[0],0), (0, 0, 0), 120)
    #img = cv2.line(img, (0,0), (0,shape[1]), (0, 0, 0), 120)
    #img = cv2.line(img, (shape[0],0), (shape[0],shape[1]), (0, 0, 0), 120)
    #img = cv2.line(img, (0,shape[1]), (shape[0],shape[1]), (0, 0, 0), 120)

    #img = cv2.line(img, (60,60), (shape[0]-60,60), (255, 255, 255), 100)
    #img = cv2.line(img, (60,60), (60,shape[1]-60), (255, 255, 255), 100)
    #img = cv2.line(img, (shape[0]-60,60), (shape[0]-60,shape[1]-60), (255, 255, 255), 100)
    #img = cv2.line(img, (60,shape[1]-60), (shape[0]-60,shape[1]-60), (255, 255, 255), 100)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #print(img)
    print("초반")

    #mask = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    #ret,img_binary=cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY)

    #cv2.imshow('imgray', img_binary)
    ret,thresh = cv2.threshold(imgray,207,255,cv2.THRESH_BINARY)

    plt.imshow(thresh)
    plt.show()

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print(contours)

    # 경계선을 그리고
    cnt = contours[0]
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    plt.imshow(img)
    plt.show()

    print("중반")

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

    '''
    아주 좋고
    일단 그림 90 돌리고
    원본 코드랑 합치는 작업
    다음 시간에...
    '''
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
    
    if(maxX - minX >500 and maxY - minY>500):
        print("if문!!!!!!!!!!!!!!!!!!!!!")
        right = [rightX, rightY]
        left = [leftX, leftY]
        top = [topX, topY]
        bottom = [bottomX, bottomY]

        print(top)
        print(left)
        print(right)
        print(bottom)

        # 이제 해야될 건 이 좌표로 perspective 해보는 거!!

        print("후반")
        print("shape 는")
        print(img.shape)

        h, w = img.shape[:2] 
        pts1 = np.float32([left, top, bottom, right]) 
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
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
        #plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
        #plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        #plt.show() 
        
        print("리턴 이전")

        return img2

    else:
        print("else 문!!!!!!!!!!!!!!!!!!!!!")
        img = cv2.line(img, (0,0), (shape[0],0), (0, 0, 0), 120)
        img = cv2.line(img, (0,0), (0,shape[1]), (0, 0, 0), 120)
        img = cv2.line(img, (shape[0],0), (shape[0],shape[1]), (0, 0, 0), 120)
        img = cv2.line(img, (0,shape[1]), (shape[0],shape[1]), (0, 0, 0), 120)

        img = cv2.line(img, (60,60), (shape[0]-60,60), (255, 255, 255), 100)
        img = cv2.line(img, (60,60), (60,shape[1]-60), (255, 255, 255), 100)
        img = cv2.line(img, (shape[0]-60,60), (shape[0]-60,shape[1]-60), (255, 255, 255), 100)
        img = cv2.line(img, (60,shape[1]-60), (shape[0]-60,shape[1]-60), (255, 255, 255), 100)

        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        #print(img)
        print("초반")

        #mask = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        #ret,img_binary=cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY)

        #cv2.imshow('imgray', img_binary)
        ret,thresh = cv2.threshold(imgray,207,255,cv2.THRESH_BINARY)

        plt.imshow(thresh)
        plt.show()

        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        print(contours)

        # 경계선을 그리고
        cnt = contours[0]
        img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
        
        plt.imshow(img)
        plt.show()

        print("중반")


        right = [shape[0]-60, shape[1]-60]
        left = [60, 60]
        top = [shape[0]-60, 60]
        bottom = [60, shape[1]-60]

        print(top)
        print(left)
        print(right)
        print(bottom)

        # 이제 해야될 건 이 좌표로 perspective 해보는 거!!

        print("후반")
        print("shape 는")
        print(img.shape)

        h, w = img.shape[:2] 
        pts1 = np.float32([left, top, bottom, right]) 
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
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
        #plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
        #plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        #plt.show() 
        
        print("리턴 이전")

        return img2

#point()