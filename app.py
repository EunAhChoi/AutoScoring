# 내가 만든 perspective 함수
import Perspective


# 이미지 픽셀 차이 보기
import sys
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

# 이미지 유사성 보기
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.ndimage import imread

# 이미지 색깔 차이로 뽑기
# opencv 사용됨

# Matplotlib과 thkinter가 서로 충돌이 나지 않도록 조정하는 코드. 지렸다.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Tkinter는 GUI에 대한 표준 python 인터페이스이며 window 창을 생성할 수 있음.
from tkinter import * # Toolkit interface의 약자
from tkinter import ttk
from tkinter import messagebox
# Python에서 이미지를 처리하고 핸들링하기 위한 Pillow 패키지에서 이미지를 표현하는 Image class.
from PIL import Image # Python Image Library의 약자
# Tkinter의 bitmapImage나 photoImage를 만들거나 수정할 수 있는 모듈
from PIL import ImageTk 
# 파일 열기 및 저장 함수들이 있는 모듈
from tkinter import filedialog
# Open CV...컴퓨터 비전 및 이미지 처리, 기계학습 소프트웨어 라이브러리
import cv2
# 두 이미지 사이의 유사성 지수 계산 모듈 (두 개의 이미지 비교해서 차이를 보여주는 역할)
from skimage.measure import compare_ssim
# OpenCV 기능을 향상시키기 위해 사용된다고 함.
import imutils 
import numpy as np # 익숙한 넘파잉 (선형대수, 통계 패키지. 주로 행렬, 벡터연산!)
import os # 운영체제에서 제공되는 여러 기능을 파이썬에서 수행할 수 있게 해준다.
import subprocess # 파이썬 외 다른 언어로 만들어진 프로그램을 통합, 제어할 수 있게 하는 모듈
import pytesseract # 파이썬에서 OCR을 수행해주게 하는 모듈.. (이미지에서 텍스트 따오기)
from operator import eq # 산술연산 모듈, 항등 연산자
import codecs # 인코딩 관련 모듈
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
state = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/info")
def info():
    return render_template('info.html')

# URL에 값을 넘기기. 그리고 출력
@app.route('/user/<username>')
def show_username(username):
    return "your name is {}, right?".format(username)

# URL에 넘길 때 변수의 데이터 타입을 설정 할 수 있음
@app.route('/user/id/<int:userid>')
def show_user_id(userid):
    return "your id is {}, right?".format(userid)


# 넘겨받은 숫자만큼 별 표시.
# 이런식으로 페이지를 재미있게 만들 수 있을 것 같음
@app.route('/draw/<number>')
def draw_start(number):
    r_str = ""
    for i in range(1, int(number)+1):
        r_str+="*"*i
        r_str+='<br>'  #html에서의 줄바꿈

    return r_str

# redirect를 통해 '/' 로 재연결 된다. (메인화면)
@app.route('/aaa')
def aaa():
    return redirect('/')


## 아래처럼 두 개를 겹쳐 놓으면, 두 경우에 대해서 모두 수행됨. 
## 단, 이 경우에 argument의 초기값이 정해져 있는 것을 확인
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name = 'hcy')


# file을 submit하는 페이지
@app.route('/upload')
def render_file():
    return render_template('upload.html')


# file이 submit되면 전달되는 페이지
# upload.html에서 form이 제출되면 /file_uploaded로 옴ㄹ겨지게 되어 있음
@app.route('/file_uploaded', methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['file1']
        # 파일 객체 혹은 파일 스트림을 가져오고, html 파일에서 넘겨지는 값의 이름을
        # file1으로 했기 때문에 file1임
        f.save(f'uploads/{secure_filename(f.filename)}')
        df_to_html = pd.read_csv(f'uploads/{secure_filename(f.filename)}').to_html() 
        # html로 전환해서 보여줌
        return df_to_html


#bootstrap 화면 켜기
@app.route("/mainframe")
def main_frame():
    return render_template('mainframe.html')


# 라디오 버튼. 테스트 타입 정하기
@app.route("/test_type", methods=['GET', 'POST'])
def testType():
    if request.method == "POST":
        state = request.form.get("myradio")
        return render_template('mainframe.html')


## testSheet ##########################
# file이 submit되면 전달되는 페이지
# mainframe.html에서 form이 제출되면 testSheet_uploaded로 옮겨지게 되어 있음
@app.route('/EmptySheet_uploaded', methods = ['GET','POST'])
def upload_EmptySheet():
    global testSheet, main_shape
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['EmptySheet']
        # uploads 폴더에 저장
        f.save("./uploads/TestSheet/" + secure_filename(f.filename))
	
        testSheet = cv2.imread("./uploads/TestSheet/"+f.filename) # 이미지 파일을 읽기 위한 객체를 리턴해주는 함수.
        # 0은 gray로 읽겠다는 의미 (cv2.IMREAD_GRAYSCALE)
        
        main_shape = testSheet.shape
        print(main_shape)

        # 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
        #testSheet = cv2.cvtColor(testSheet, cv2.COLOR_GRAY2RGB)

        testSheet = Perspective.point(testSheet, main_shape)

        testSheet = cv2.cvtColor(testSheet, cv2.COLOR_RGB2GRAY)

        return render_template('mainframe.html')


## AnswerSheet ####################################
# file이 submit되면 전달되는 페이지
# mainframe.html에서 form이 제출되면 testSheet_uploaded로 옮겨지게 되어 있음
@app.route('/AnswerSheet_uploaded', methods = ['GET','POST'])
def upload_AnswerSheet():
    global position, answerList, answerSheet
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['AnswerSheet']
        # uploads 폴더에 저장
        f.save("./uploads/AnswerSheet/" + secure_filename(f.filename))
	
        answerSheet = cv2.imread("./uploads/AnswerSheet/"+f.filename)

        answerSheet = Perspective.point(answerSheet, main_shape)

        answerSheet = cv2.cvtColor(answerSheet, cv2.COLOR_RGB2GRAY)

        # 이미지의 차이를 실제 png 파일로 만들어주는 코드 추가
        diff = cv2.absdiff(testSheet, answerSheet)
        mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)

        ret,img_binary=cv2.threshold(mask, 110,255,cv2.THRESH_BINARY)
        cv2.imwrite("/Users/hcy/Desktop/result6.png", img_binary)

        cnts= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]


        th = 1
        imask = mask>th

        canvas = np.zeros_like(answerSheet, np.uint8)
        canvas[imask] = answerSheet[imask]

        #이미지 잡음 제거 코드임
        #dst = cv2.fastNlMeansDenoising(canvas, None, 10, 7, 21)

        #ret,dst=cv2.threshold(dst,20,255,cv2.THRESH_BINARY_INV)


        cv2.imwrite("/Users/hcy/Desktop/result.png", canvas)

        # 정답지 세로로 분리
        # 이제 윤곽선을 list에 저장 했으므로, 각 이미지의 다른 영역 주위에 사각형을 그리겠습니다.
        x1,y1,h1 = 0,0,0
        num=0
        count=0
        startX,startY=0,0
        img=[]
        row=[]
        # print(row)하면 [[]] 상황
        row.append([])
        box=[]
        # 경계값 까지
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            # 윤곽선의 경계 상자를 계산한 다음 두 입력 이미지 (시험지,답지) 모두에 경계상자를 그려서
            # 두 이미지가 서로 다른 부분을 나타낸다
            # boundingRect 함수가 윤곽선 주변 경계상자를 계산하는 함수
            # 사각형의 xy 좌표와 사각형의 width/height 저장
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

        #정답지 세로로 분리된 것 중에 가로로 분리할 수 있는지 확인
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
                # +- 5 해서 좀 넓게 짤리게 했음. 인식 잘되게
                img.append(answerSheet[minY-5:maxY+5, minX-5:maxX+5])    # img == 최종 답안 단어들의 이미지를 저장한 리스트


        if state == 1:

            for i in range(0, len(img)):
                    cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])

            for i in range(len(img)):
                result = pytesseract.image_to_string(img[i],config='--psm 6')
                trueAnswer.append(result)
            print("Extract answer...")
            print("----------------------------")
            for i in range(len(trueAnswer)):
                print(trueAnswer[i])
            print("----------------------------")
            print("Extract Complete!!!")
            if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
                os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))
            f = open("/Users/hcy/Desktop/GP/answerSheet/answerNumberList.txt","w",-1,"utf-8")
            for i in range(len(trueAnswer)):
                f.write(trueAnswer[i]+"\n")
            f.close()
            
            for i in range(0, len(img)):
                cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])


        if state == 2:
            print("#######################################")
            # 시험지와 답안지 비교해서 답을 짤라서 각각 저장
            for i in range(0, len(img)):
                cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])

            os.chdir("/Users/hcy/Desktop/GP/src/")
            os.system("python3 main_answer.py")

            # 정답을 리스트로 저장해봄
            with codecs.open('/Users/hcy/Desktop/GP/answerSheet/trueAnswerLists.txt','r') as r:
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

        return render_template('mainframe.html')  


## StudentSheet ####################################
# file이 submit되면 전달되는 페이지
# mainframe.html에서 form이 제출되면 testSheet_uploaded로 옮겨지게 되어 있음
@app.route('/AnswerSheet_uploaded', methods = ['GET','POST'])
def upload_AnswerSheet():
    global position, answerList, answerSheet
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['AnswerSheet']
        # uploads 폴더에 저장
        f.save("./uploads/AnswerSheet/" + secure_filename(f.filename))



if __name__ == '__main__':
    # debug를 True로 세팅하면, 해당 서버 세팅 후에 코드가 바뀌어도 문제없이 실행됨. 
    app.run(debug = True)