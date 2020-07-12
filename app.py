# Perspective function that i made
import Perspective
import sys
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
# Image Similarity compare
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.ndimage import imread
# Matplotlib and thkinter are no longer conflict each other
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# ToolKit Interface
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from skimage.measure import compare_ssim
import imutils 
import numpy as np
import os
import subprocess
import pytesseract
from operator import eq
import codecs
import warnings
warnings.filterwarnings(action='ignore')

from flask import Flask, render_template, flash
from flask import redirect
from flask import url_for, request
from flask import send_file
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)

testSheet = None
answerSheet = None
studentSheet = None
position = []
answerList = []
trueAnswer = []
studentAnswer = []
main_shape = ()

@app.route("/")
def index():
    return render_template('index.html')
    
# Determine test type
@app.route("/test_type", methods=['GET', 'POST'])
def testType():
    global state
    if request.method == "POST":
        state = request.form.get("myradio")
        return redirect('/')

## testSheet ##########################
# Page to be delivered when file is submitted
@app.route('/EmptySheet_uploaded', methods = ['GET','POST'])
def upload_EmptySheet():
    global testSheet, main_shape
    if request.method == 'POST':
        f = request.files['EmptySheet']
        # save at uploads directory
        f.save("./uploads/TestSheet/" + secure_filename(f.filename))
	
        testSheet = cv2.imread("./uploads/TestSheet/"+f.filename)
        
        main_shape = testSheet.shape
        print(main_shape)

        # 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
        #testSheet = cv2.cvtColor(testSheet, cv2.COLOR_GRAY2RGB)
        #testSheet = Perspective.point(testSheet, main_shape)
        testSheet = cv2.cvtColor(testSheet, cv2.COLOR_BGR2GRAY)

        return redirect('/')


## AnswerSheet ####################################
# Page to be delivered when file is submitted
@app.route('/AnswerSheet_uploaded', methods = ['GET','POST'])
def upload_AnswerSheet():
    global position, answerList, answerSheet
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['AnswerSheet']
        # save at uploads directory
        f.save("./uploads/AnswerSheet/" + secure_filename(f.filename))
	
        #answerSheet = cv2.imread("./uploads/AnswerSheet/"+f.filename)
        answerSheet = cv2.imread("/Users/hcy/Desktop/GP/Picture/studentstudent.jpeg")
        #answerSheet = Perspective.point(answerSheet, main_shape)
        answerSheet = cv2.cvtColor(answerSheet, cv2.COLOR_BGR2GRAY)

        # 이미지의 차이를 실제 png 파일로 만들어주는 코드 추가
        diff = cv2.absdiff(testSheet, answerSheet)
        mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)

        ret,img_binary=cv2.threshold(mask, 110,255,cv2.THRESH_BINARY)

        cnts= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        th = 1
        imask = mask>th

        canvas = np.zeros_like(answerSheet, np.uint8)
        canvas[imask] = answerSheet[imask]

        # Vertical answer sheet
        # Now that we have the outlines stored in the list, we will draw a rectangle around the different areas of each image.
        x1,y1,h1 = 0,0,0
        num=0
        count=0
        startX,startY=0,0
        img=[]
        row=[]
        row.append([])
        box=[]
        # Up to the threshold
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            # The function that the boundingRect function calculates the bounding box around the contour
            # Save the xy coordinates of the rectangle and the width/height of the rectangle
            (x,y,w,h) = cv2.boundingRect(c)
            box.append(c)
            if count==0:
                row[num].append(count)
                pass
            else:
                if abs(y1-y)<20:
                    row[num].append(count)
                else:
                    num = num+1
                    row.append([])
                    row[num].append(count)
            y1=y
            count+=1
        
        #Check if the correct answer sheet can be separated horizontally
        a,b,c,d=0,0,0,0
        for i in range(0,len(row)):
            minX=10000
            minY=10000
            maxX=0
            maxY=0

            for j in range(0,len(row[i])):
                a,b,c,d = cv2.boundingRect(box[row[i][j]])
                if minX>a:
                    minX=a
                if maxY < b + d:
                    maxY = b + d
                if maxX < a + c:
                    maxX = a + c
                if minY > b:
                    minY = b
            if abs(maxY - minY) < 5:
                pass
            else:
                # +- 5 to cut it wide. cognition well
                img.append(answerSheet[minY-5:maxY+5, minX-5:maxX+5])
                # img : list of images of final answer words
                
        print("state is :", state)
        if state == "1":
            for i in range(0, len(img)):
                    #cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])
                    cv2.imwrite("./result/trueAnswer/"+""+str(i) + ".jpg", img[i])

            for i in range(len(img)):
                result = pytesseract.image_to_string(img[i],config='--psm 6')
                trueAnswer.append(result)
            print("Extract answer...")
            print("----------------------------")
            for i in range(len(trueAnswer)):
                print(trueAnswer[i])
            print("----------------------------")
            print("Extract Complete!!!")
            #if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
            if not (os.path.isdir("./result/answerSheet")):
                #os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))
                os.makedirs(os.path.join("./result/answerSheet"))
            #f = open("/Users/hcy/Desktop/GP/answerSheet/answerNumberList.txt","w",-1,"utf-8")
            f = open("./result/answerSheet/answerNumberList.txt","w",-1,"utf-8")
            for i in range(len(trueAnswer)):
                f.write(trueAnswer[i]+"\n")
            f.close()
            
            for i in range(0, len(img)):
                #cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])
                cv2.imwrite("./result/trueAnswer/"+""+str(i) + ".jpg", img[i])


        if state == "2":
            print("#######################################")
            # Compare the test paper and answer sheet and cut the answer and save each
            for i in range(0, len(img)):
                #cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])
                cv2.imwrite("./result/trueAnswer/"+""+str(i) + ".jpg", img[i])

            #os.chdir("/Users/hcy/Desktop/GP/src/")
            os.chdir("./src/")
            os.system("python3 main_answer.py")

            # Save the correct answer as a list
            #with codecs.open('/Users/hcy/Desktop/GP/answerSheet/trueAnswerLists.txt','r') as r:
            with codecs.open('./result/answerSheet/trueAnswerLists.txt','r') as r:
                while(1):
                    line = r.readline()
                    try:escape=line.index('\n')
                    except:escape=len(line)

                    if line:
                        trueAnswer.append(line[0:escape].replace(" ",""))
                    else:
                        break
            r.close()

            print("Extract answer...")
            print("----------------------------")
            for i in range(len(trueAnswer)):
                print(trueAnswer[i])
            print("----------------------------")
            print("Extract Complete!!!")

        return redirect('/')


## StudentSheet ####################################
# Page to be delivered when file is submitted
@app.route('/StudentSheet_uploaded', methods = ['GET','POST'])
def upload_StudentSheet():
    global studentSheet
    if request.method == 'POST': # POST 방식으로 전달된 경우
	
        studentSheet = cv2.imread("/Users/hcy/Desktop/GP/Picture/studentstudent.jpeg")
        #color = cv2.imread("./uploads/StudentSheet/"+f.filename)
        color = studentSheet
        
        #if not (os.path.isdir("/Users/hcy/Desktop/GP/answer")):
        if not (os.path.isdir("./result/answer")):
            #os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answer"))
            os.makedirs(os.path.join("./result/answer"))

        # 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
        #studentSheet = Perspective.point(studentSheet, main_shape)
        studentSheet = cv2.cvtColor(studentSheet, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(testSheet, studentSheet)
        #mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)

        ret,img_binary=cv2.threshold(mask, 110,255,cv2.THRESH_BINARY)

        cnts= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # Vertical answer sheet
        x1, y1, h1 = 0, 0, 0
        num = 0
        count = 0
        startX, startY = 0, 0
        img2 = []
        row2 = []
        row2.append([])
        box2 = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            box2.append(c)
            if count == 0:
                row2[num].append(count)
                pass
            else:
                if abs(y1 - y) < 20:
                    row2[num].append(count)
                else:
                    num = num + 1
                    row2.append([])
                    row2[num].append(count)
            y1 = y
            count += 1
       
        # Check if the correct answer sheet can be separated horizontally
        a, b, c, d = 0, 0, 0, 0
        for i in range(0, len(row2)):
            minX = 10000
            minY = 10000
            maxX = 0
            maxY = 0

            for j in range(0, len(row2[i])):
                a, b, c, d = cv2.boundingRect(box2[row2[i][j]])
                if minX > a:
                    minX = a
                if maxY < b + d:
                    maxY = b + d
                if maxX < a + c:
                    maxX = a + c
                if minY > b:
                    minY = b
            if abs(maxY - minY) < 5:
                pass
            else:
                img2.append(studentSheet[minY-5:maxY+5, minX-5:maxX+5])
                pos = []
                pos.append(minY)
                pos.append(maxY)
                pos.append(minX)
                pos.append(maxX)
                position.append(pos)

        if state == "1":
                
            for i in range(len(img2)):
                result = pytesseract.image_to_string(img2[i],config='--psm 6')
                studentAnswer.append(result)

            for i in range(len(studentAnswer)):
                print(studentAnswer[i] + "확인")

            #if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
            if not (os.path.isdir("./result/answerSheet")):
                #os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))
                os.makedirs(os.path.join("./result/answerSheet"))
            #f = open("/Users/hcy/Desktop/GP/answerSheet/answerwordList.txt","w",-1,"utf-8")
            f = open("./result/answerSheet/answerwordList.txt","w",-1,"utf-8")
            for i in range(len(studentAnswer)):
                f.write(studentAnswer[i]+"\n")
            f.close()
            
            for i in range(0, len(img2)):
                #cv2.imwrite("/Users/hcy/Desktop/GP/answer/"+""+str(i) + ".jpg", img2[i])
                cv2.imwrite("./result/answer/"+""+str(i) + ".jpg", img2[i])

        if state == "2":
                
            for i in range(0, len(img2)):
                #cv2.imwrite("/Users/hcy/Desktop/GP/answer/"+""+str(i) + ".jpg", img2[i])
                cv2.imwrite("./result/answer/"+""+str(i) + ".jpg", img2[i])
            #os.chdir("/Users/hcy/Desktop/GP/src/")
            os.chdir("./src/")
            os.system("python3 main.py")           # main.py 실행하면 answerImage에 있는 폴더 모두 실행, txt파일에 정답 저장

            # 학생 답지에서 추출한 답들. main.py에서 끌어다 왔음.
            #with codecs.open('/Users/hcy/Desktop/GP/answerSheet/answerwordLists.txt','r') as r:
            with codecs.open('./result/answerSheet/answerwordLists.txt','r') as r:
                while(1):
                    line = r.readline()
                    try:escape=line.index('\n')
                    except:escape=len(line)

                    if line:
                        studentAnswer.append(line[0:escape].replace(" ",""))
                    else:
                        break
            r.close()

            print(studentAnswer)

        print("###########################################")

        score = len(studentAnswer)
        print("score : ",score)
        correctNum = np.zeros(len(trueAnswer))
        for i in range(0,len(studentAnswer)):
            print("true: %s, stud: %s" %(trueAnswer[i],studentAnswer[i]))
            if studentAnswer[i] == trueAnswer[i] :
                correctNum[i] = 1
                
            else :
                score = score - 1   		
            
        print(score)
        print(correctNum)
        #color = cv2.imread("./uploads/StudentSheet/"+f.filename)
        for i in range(0,len(correctNum)):
            if(correctNum[i] == 1):
                print("correct!")
                #print(img2[i][0],img2[i][1])
                cv2.circle(color,(int((position[i][3]+position[i][2])/2),int((position[i][0]+position[i][1])/2)),30,(0,0,255),5)
                #cv2.circle(studentSheet,(int(x+w/2),int(y-h/2)),30,(0,0,255),-1)
            else:
                cv2.putText(color," / ", (int((position[i][3]+position[i][2])/2)-70,int((position[i][0]+position[i][1])/2)+35), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.putText(color,str(score) + " / " + str(len(correctNum)), (1070,1950),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5,cv2.LINE_AA)
        color = cv2.resize(color,(850,850))
        cv2.imwrite("./result/Result.jpg",color)
        # cv2.circle(img,(447,63), 63, (0,0,255), -1)

        return redirect('/')


if __name__ == '__main__':
    # debug를 True로 세팅하면, 해당 서버 세팅 후에 코드가 바뀌어도 문제없이 실행됨. 
    app.run()
