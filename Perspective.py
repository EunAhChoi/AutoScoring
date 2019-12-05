import numpy as np 
import cv2  
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import math
from numpy import linalg

def point(img, shape):
    
    shape = (shape[1], shape[0])
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_AREA)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    print("img 쉐잎은?")
    print(img.shape)  
    print("gray 쉐잎은?")
    print(imgray.shape)

    #mask = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    #ret,img_binary=cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY)

    #cv2.imshow('imgray', img_binary)
    ret,thresh = cv2.threshold(imgray,207,255,cv2.THRESH_BINARY)

    #plt.imshow(thresh)
    #plt.show()

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)

    # 경계선을 그리고
    cnt = contours[0]
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    #plt.imshow(img)
    #plt.show()

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
        #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
        #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
        #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
        #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
        #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
        #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
        #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
        #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
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

        #plt.imshow(thresh)
        #plt.show()

        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #print(contours)

        # 경계선을 그리고
        cnt = contours[0]
        img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
        
        #plt.imshow(img)
        #plt.show()

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
        #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
        #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
        #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
        #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
        #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
        #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
        #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
        #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
        #plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
        #plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        #plt.show() 
        

        return img2

#point()


def distance(X,Y,X2,Y2):
    return math.sqrt(pow(X-X2,2) + pow(Y-Y2,2))


def transform(img, empty):

    empty = cv2.resize(empty, (900,900), interpolation=cv2.INTER_AREA)
    imgBlur = cv2.medianBlur(img, 5)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    
    imgray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

    # 원본 
    shape = imgray.shape
    print("img 쉐잎은?")
    print(img.shape)  
    print("gray 쉐잎은?")
    print(imgray.shape)

    #mask = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    #ret,img_binary=cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY)

    #cv2.imshow('imgray', img_binary)
    #plt.imshow(imgray)
    #plt.show()

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', imgray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret,thresh = cv2.threshold(imgray,140,255,cv2.THRESH_BINARY)

    #plt.imshow(thresh)
    #plt.show()

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)
    print(len(contours))

    # 경계선을 그리고
    cnt = contours[0]
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    #plt.imshow(img)
    #plt.show()

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("중반")

    
    minX = 10000
    # 꼭지점 저장되는 곳

    temp_X = 0
    temp_Y = 0
    # 임의의 점 1개 잡고, 가장 먼 곳을 찾는다. 요기가 최초의 꼭지점
    # 그 구해진 꼭지점에서 가장 먼 곳을 또 찾는다. 이게 두번쨰 꼭지점
    # 이제 그 둘의 중앙값을 찾고 거기서 젤 먼점을 구한다. 여기가 세번째
    # 마지막으로 면적을 통해 4번째 꼭지점을 찾는다. 끝
    
    #임의의 점은 그냥 X가 작은 곳을 잡겠다.
    for i in cnt:
        if minX > i[0][0]:
            minX = i[0][0]
            temp_X = i[0][0]
            temp_Y = i[0][1]

    dis = 0
    point1_X = 0
    point1_Y = 0
    
    # 임의의 좌표와 거리가 가장 먼 곳을 저장 이곳이 첫 번째 꼭지점이 됨
    for i in cnt:
        if dis < distance(temp_X, temp_Y, i[0][0], i[0][1]):
            dis = distance(temp_X, temp_Y, i[0][0], i[0][1])
            point1_X = i[0][0]
            point1_Y = i[0][1]
    

    print("point1_X ", point1_X)
    print("point1_Y ", point1_Y)

    dis = 0
    point2_X = 0
    point2_Y = 0

    for i in cnt:
        if dis < distance(point1_X, point1_Y, i[0][0], i[0][1]):
            dis = distance(point1_X, point1_Y, i[0][0], i[0][1])
            point2_X = i[0][0]
            point2_Y = i[0][1]

    print("point2_X", point2_X)
    print("point2_Y", point2_Y)

    # 다음은 두 점의 중앙값을 가지고 이 중앙값과 가장 거리가 먼 점을 포인트 3으로 잡는다
    center_X = (point1_X + point2_X)/2
    center_Y = (point1_Y + point2_Y)/2


    dis = 0
    point3_X = 0
    point3_Y = 0

    for i in cnt:
        if dis < distance(center_X, center_Y, i[0][0], i[0][1]):
            dis = distance(center_X, center_Y, i[0][0], i[0][1])
            point3_X = i[0][0]
            point3_Y = i[0][1]

    print("point3_X ", point3_X)
    print("point3_Y ", point3_Y)

    # 마지막으로 면적을 통해서 마지막 꼭지점을 찾는다.
    area = 0
    point4_X = 0
    point4_Y = 0
    
    for i in cnt:
        tempArea1 = abs((point1_X * point2_Y + point2_X * i[0][1] + i[0][0] * point1_Y) - (point2_X * point1_Y + i[0][0] * point2_Y + point1_X * i[0][1]))
        tempArea2 = abs((point1_X * i[0][1] + i[0][0] * point3_Y + point3_X * point1_Y) - (i[0][0] * point1_Y + point3_X * i[0][1] + point1_X * point3_Y))
        tempArea3 = abs((i[0][0] * point2_Y + point2_X * point3_Y + point3_X * i[0][1]) - (point2_X * i[0][1] + point3_X * point2_Y + i[0][0] * point3_Y))
        tempArea = tempArea1 + tempArea2 + tempArea3

        if area < tempArea:
            area = tempArea
            point4_X = i[0][0]
            point4_Y = i[0][1]
    
    print("point4_X", point4_X)
    print("point4_Y", point4_Y)
    
    points = []
    points.append(point1_X)
    points.append(point2_X)
    points.append(point3_X)
    points.append(point4_X)
    points.append(point1_Y)
    points.append(point2_Y)
    points.append(point3_Y)
    points.append(point4_Y)

    for i in range(len(points)):

        if(points[i] == 0):

            print("사진 자체에 선 긋고 짜르기!!!!!!!!!!!!!!!!!!!!!")
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

            #print(contours)

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
            # 이미지, 원의 중심, 원의 반지름, BGR값, -1은 주어진 색으로 도형을 채운다 라는 뜻
            #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
            #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
            #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
            #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
            #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
            #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
            #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
            #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
            plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
            plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
            plt.show() 
            

            print("최종 결과 shape ", img2.shape)
            return img2


    if abs(point1_X - point2_X) > 50 and abs(point1_Y - point2_Y) > 50:    

        print("컨투어 대로 짜르기")
        top = [point1_X, point1_Y]
        bottom = [point2_X, point2_Y]
        right = [point3_X, point3_Y]
        left = [point4_X, point4_Y]

        print("Top " , top)
        print("Left " , left)
        print("Right " , right) 
        print("Botton " , bottom)

        # 이제 해야될 건 이 좌표로 perspective 해보는 거!!

        print("후반")
        print("shape 는")
        print(img.shape)

        h, w = img.shape[:2] 
        pts1 = np.float32([left, top, bottom, right]) 
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
        M = cv2.getPerspectiveTransform(pts1, pts2) 
        img2 = cv2.warpPerspective(img, M, (w, h)) 
        #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
        #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
        #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
        #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
        #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
        #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
        #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
        #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
        plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        plt.show() 
        
        print("M")
        print(M)

        print("최종 결과 shape ", img2.shape)

        # 역행렬 연산
        N = linalg.inv(M)
        print(N)

        tempEmpty = cv2.warpPerspective(empty, N, (900, 900))
        
        plt.subplot(1, 2, 1), plt.imshow(empty), plt.title('empty') 
        plt.subplot(1, 2, 2), plt.imshow(tempEmpty), plt.title('tempEmpty')
        plt.show()



        ###########################################################
        ###########################################################

        imgBlur = cv2.medianBlur(tempEmpty, 5)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)    
        imgray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

        # 원본 
        shape = imgray.shape
        print("img 쉐잎은?")
        print(img.shape)  
        print("gray 쉐잎은?")
        print(imgray.shape)

        #mask = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
        #ret,img_binary=cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY)

        #cv2.imshow('imgray', img_binary)
        #plt.imshow(imgray)
        #plt.show()

        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', imgray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ret,thresh = cv2.threshold(imgray,140,255,cv2.THRESH_BINARY)

        #plt.imshow(thresh)
        #plt.show()

        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #print(contours)
        print(len(contours))

        # 경계선을 그리고
        cnt = contours[0]
        tempEmpty = cv2.drawContours(tempEmpty, [cnt], 0, (0,255,0), 3)
        
        #plt.imshow(img)
        #plt.show()

        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', tempEmpty)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("중반")

        
        minX = 10000
        # 꼭지점 저장되는 곳

        temp_X = 0
        temp_Y = 0
        # 임의의 점 1개 잡고, 가장 먼 곳을 찾는다. 요기가 최초의 꼭지점
        # 그 구해진 꼭지점에서 가장 먼 곳을 또 찾는다. 이게 두번쨰 꼭지점
        # 이제 그 둘의 중앙값을 찾고 거기서 젤 먼점을 구한다. 여기가 세번째
        # 마지막으로 면적을 통해 4번째 꼭지점을 찾는다. 끝
        
        #임의의 점은 그냥 X가 작은 곳을 잡겠다.
        for i in cnt:
            if minX > i[0][0]:
                minX = i[0][0]
                temp_X = i[0][0]
                temp_Y = i[0][1]

        dis = 0
        point1_X = 0
        point1_Y = 0
        
        # 임의의 좌표와 거리가 가장 먼 곳을 저장 이곳이 첫 번째 꼭지점이 됨
        for i in cnt:
            if dis < distance(temp_X, temp_Y, i[0][0], i[0][1]):
                dis = distance(temp_X, temp_Y, i[0][0], i[0][1])
                point1_X = i[0][0]
                point1_Y = i[0][1]
        

        print("point1_X ", point1_X)
        print("point1_Y ", point1_Y)

        dis = 0
        point2_X = 0
        point2_Y = 0

        for i in cnt:
            if dis < distance(point1_X, point1_Y, i[0][0], i[0][1]):
                dis = distance(point1_X, point1_Y, i[0][0], i[0][1])
                point2_X = i[0][0]
                point2_Y = i[0][1]

        print("point2_X", point2_X)
        print("point2_Y", point2_Y)

        # 다음은 두 점의 중앙값을 가지고 이 중앙값과 가장 거리가 먼 점을 포인트 3으로 잡는다
        center_X = (point1_X + point2_X)/2
        center_Y = (point1_Y + point2_Y)/2


        dis = 0
        point3_X = 0
        point3_Y = 0

        for i in cnt:
            if dis < distance(center_X, center_Y, i[0][0], i[0][1]):
                dis = distance(center_X, center_Y, i[0][0], i[0][1])
                point3_X = i[0][0]
                point3_Y = i[0][1]

        print("point3_X ", point3_X)
        print("point3_Y ", point3_Y)

        # 마지막으로 면적을 통해서 마지막 꼭지점을 찾는다.
        area = 0
        point4_X = 0
        point4_Y = 0
        
        for i in cnt:
            tempArea1 = abs((point1_X * point2_Y + point2_X * i[0][1] + i[0][0] * point1_Y) - (point2_X * point1_Y + i[0][0] * point2_Y + point1_X * i[0][1]))
            tempArea2 = abs((point1_X * i[0][1] + i[0][0] * point3_Y + point3_X * point1_Y) - (i[0][0] * point1_Y + point3_X * i[0][1] + point1_X * point3_Y))
            tempArea3 = abs((i[0][0] * point2_Y + point2_X * point3_Y + point3_X * i[0][1]) - (point2_X * i[0][1] + point3_X * point2_Y + i[0][0] * point3_Y))
            tempArea = tempArea1 + tempArea2 + tempArea3

            if area < tempArea:
                area = tempArea
                point4_X = i[0][0]
                point4_Y = i[0][1]
        
        print("point4_X", point4_X)
        print("point4_Y", point4_Y)
        
        points = []
        points.append(point1_X)
        points.append(point2_X)
        points.append(point3_X)
        points.append(point4_X)
        points.append(point1_Y)
        points.append(point2_Y)
        points.append(point3_Y)
        points.append(point4_Y)


        print("컨투어 대로 짜르기")
        top = [point1_X, point1_Y]
        bottom = [point2_X, point2_Y]
        right = [point3_X, point3_Y]
        left = [point4_X, point4_Y]

        print("Top " , top)
        print("Left " , left)
        print("Right " , right) 
        print("Botton " , bottom)

        # 이제 해야될 건 이 좌표로 perspective 해보는 거!!

        h, w = tempEmpty.shape[:2] 
        pts1 = np.float32([left, top, bottom, right]) 
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) 
        M = cv2.getPerspectiveTransform(pts1, pts2) 
        img2 = cv2.warpPerspective(tempEmpty, M, (w, h)) 
        #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
        #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
        #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
        #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
        #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
        #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
        #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
        #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
        plt.subplot(1, 2, 1), plt.imshow(tempEmpty), plt.title('image') 
        plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        plt.show()


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

        #print(contours)

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
        #cv2.circle(img, (top[0], top[1]), 20, (255, 0, 0), -1) 
        #cv2.circle(img, (right[0], right[1]), 20, (0, 255, 0), -1) 
        #cv2.circle(img, (left[0], left[1]), 20, (0, 0, 255), -1) 
        #cv2.circle(img, (bottom[0], bottom[1]), 20, (0, 0, 0), -1) 
        #cv2.circle(img2, (0, 0), 20, (255, 0, 0), -1) 
        #cv2.circle(img2, (1024, 0), 20, (0, 255, 0), -1) 
        #cv2.circle(img2, (0, h), 20, (0, 0, 255), -1) 
        #cv2.circle(img2, (1024, h), 20, (0, 0, 0), -1) 
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title('image') 
        plt.subplot(1, 2, 2), plt.imshow(img2), plt.title('perspective') 
        plt.show() 
        
        print("최종 결과 shape ", img2.shape)

        return img2


def drawbox(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    COLOR_MIN = np.array([0, 50, 50],np.uint8)
    COLOR_MAX = np.array([20, 150, 150],np.uint8)
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)

    plt.imshow(frame_threshed)
    plt.show()
    imgray = frame_threshed
    ret,thresh = cv2.threshold(frame_threshed,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Show",img)
    cv2.waitKey()
    cv2.destroyAllWindows()


    