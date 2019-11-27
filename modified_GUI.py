# -*- coding: utf-8 -*-
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


def changeDimension(arr):
   
    # 2차원 -> 3차원 변경 
    if(arr.ndim ==2):
        d3 = arr.ravel()
        d3 = np.hstack((d3,d3,d3))
        d3 = d3.reshape(a.shape[0], a.shape[1], 3)
        return d3
    
    # 3차원 -> 2차원 변경
    if(arr.ndim ==3):
        d2 = arr.ravel()
        slice_range = arr.shape[0] * arr.shape[1]
        d2 = d2[0:slice_range]
        d2 = d2.reshape(arr.shape[0], arr.shape[1])
        return d2


def dif_color():
	# 색상 범위 설정

	#print(type(i1))
	#print("##########################################")
	lower_orange = (100, 200, 200)
	upper_orange = (140, 255, 255)

	lower_green = (30, 80, 80)
	upper_green = (70, 255, 255)

	lower_blue = (0, 180, 55)
	upper_blue = (20, 255, 200)

	lower_red = (-2, 100, 100)
	upper_red = (2, 255, 255)

	# 이미지 파일을 읽어온다
	img = cv2.imread('/Users/hcy/Desktop/GP/예제시험지/answer.jpeg')
	#img = mpimg.imread('/Users/hcy/Desktop/GP/예제시험지/answer.jpeg', cv2.IMREAD_COLOR)
	cv2.imwrite("/Users/hcy/Desktop/temp.png",img)
	# BGR to HSV 변환
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# 색상 범위를 제한하여 mask 생성
	img_mask = cv2.inRange(img_hsv, lower_red, upper_red)
	# 원본 이미지를 가지고 Object 추출 이미지로 생성
	img_result = cv2.bitwise_and(img, img, mask=img_mask)
	# 다시 HSV -> BGR
	#img_result = cv2.cvtColor(img_result,cv2.COLOR_HSV2BGR)
	# 글씨에 점 같은 노이즈 없애보기
	kernel = np.ones((5,5), np.uint8)
	closing = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)

	# 글씨 굵게. 노이즈 적게 만들기
	kernel = np.ones((3,3), np.uint8)
	dilation = cv2.dilate(closing, kernel, iterations = 1)
	# 결과 이미지 생성
	cv2.imwrite("/Users/hcy/Desktop/result2.png", dilation)


# 이미지 픽셀 나타내보기
def pixel(i1, i2):
	# read images as 2D arrays (convert to grayscale for simplicity)
	img1 = to_grayscale(i1.astype(float))
	img2 = to_grayscale(i2.astype(float))

	# compare
	n_m, n_0 = compare_images(img1, img2)
	print("Manhattan norm:", n_m, "/per pixel", n_m/img1.size)
	print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def show_score(img1, img2):
	img1 = resize(img1, (2**10, 2**10))
	img2 = resize(img2, (2**10, 2**10))

	score, diff = compare_ssim(img1, img2, full = True)
	print(score)


def select1():        # 시험지 선택 함수

	global testSheet, main_shape  # 전역변수선언. 시험지를 저장하게 된다.
	path = filedialog.askopenfilename()
	
	testSheet = cv2.imread(path, 0) # 이미지 파일을 읽기 위한 객체를 리턴해주는 함수.
	# 0은 gray로 읽겠다는 의미 (cv2.IMREAD_GRAYSCALE)

	main_shape = testSheet.shape

	# 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
	#testSheet = cv2.cvtColor(testSheet, cv2.COLOR_GRAY2RGB)

	testSheet = cv2.cvtColor(testSheet, cv2.COLOR_GRAY2RGB)

	testSheet = Perspective.point(testSheet, main_shape)

	testSheet = cv2.cvtColor(testSheet, cv2.COLOR_RGB2GRAY)

	

##############################################################################

# 인덴트 탭
def select2():         # 정답 and 좌표찾기
	# answerSheet : 정답지 저장하는 곳. position : ??, answerList : 정답 추출?
	global position, answerList, answerSheet
	path = filedialog.askopenfilename() # 파일 열기 모듈 method 사용. path에 경로 저장.
	answerSheet = cv2.imread(path,0) # 답지 경로 찾아서 이미지 파일 객체 생성

	#answerSheet = changeDimension(answerSheet)

	print("main_shape :", main_shape)
	print(answerSheet.shape)

	# 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
	answerSheet = cv2.cvtColor(answerSheet, cv2.COLOR_GRAY2RGB)

	answerSheet = Perspective.point(answerSheet, main_shape)

	answerSheet = cv2.cvtColor(answerSheet, cv2.COLOR_RGB2GRAY)

	# 이미지의 차이를 실제 png 파일로 만들어주는 코드 추가
	diff = cv2.absdiff(testSheet, answerSheet)
	mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)
	#diff = cv2.GaussianBlur(diff,(3,3),0)
	# "펄스펙티브에 있던 녀석임 밑이 ㅇㅋ?"
	#mask = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

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

#################################################################################

def select3():         # 학생들 정답 찾기 & 정답과 비교, 채점해서 출력
	global studentSheet
	path = filedialog.askopenfilename()
	studentSheet = cv2.imread(path,0)
	#studentSheet = cv2.imread("/Users/hcy/Desktop/answerNumber.jpeg")

	#studentSheet = Perspective.point(studentSheet, studentSheet.shape)

	if not (os.path.isdir("/Users/hcy/Desktop/GP/answer")):
		os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answer"))

	# 2차원 이미지 (흑백) -> 3차원 이미지 (컬러)
	studentSheet = cv2.cvtColor(studentSheet, cv2.COLOR_GRAY2RGB)

	studentSheet = Perspective.point(studentSheet, main_shape)

	studentSheet = cv2.cvtColor(studentSheet, cv2.COLOR_RGB2GRAY)

	diff = cv2.absdiff(testSheet, studentSheet)
	#mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)

	ret,img_binary=cv2.threshold(mask, 110,255,cv2.THRESH_BINARY)
	cv2.imwrite("/Users/hcy/Desktop/result7.png", img_binary)

	cnts= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# 정답지 세로로 분리
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

	# 정답지 세로로 분리된 것 중에 가로로 분리할 수 있는지 확인
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
			img2.append(studentSheet[minY-5:maxY+5, minX-5:maxX+5])  # img == 최종 답안 단어들의 이미지를 저장한 리스트
			pos = []
			pos.append(minY)
			pos.append(maxY)
			pos.append(minX)
			pos.append(maxX)
			position.append(pos)

	if state == 1:
    		
		for i in range(len(img2)):
			result = pytesseract.image_to_string(img2[i],config='--psm 6')
			studentAnswer.append(result)

		for i in range(len(studentAnswer)):
			print(studentAnswer[i] + "확인")

		if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
			os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))
		f = open("/Users/hcy/Desktop/GP/answerSheet/answerwordList.txt","w",-1,"utf-8")
		for i in range(len(studentAnswer)):
			f.write(studentAnswer[i]+"\n")
		f.close()
		
		for i in range(0, len(img2)):
			cv2.imwrite("/Users/hcy/Desktop/GP/answer/"+""+str(i) + ".jpg", img2[i])

	if state == 2:
    		
		for i in range(0, len(img2)):
			cv2.imwrite("/Users/hcy/Desktop/GP/answer/"+""+str(i) + ".jpg", img2[i])
		os.chdir("/Users/hcy/Desktop/GP/src/")
		os.system("python3 main.py")           # main.py 실행하면 answerImage에 있는 폴더 모두 실행, txt파일에 정답 저장

		# 학생 답지에서 추출한 답들. main.py에서 끌어다 왔음.
		with codecs.open('/Users/hcy/Desktop/GP/answerSheet/answerwordLists.txt','r') as r:
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
	color = cv2.imread(path)
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
	cv2.imshow("/Users/hcy/Desktop/GP/Result",color)
	cv2.imwrite("/Users/hcy/Desktop/GP/Result.jpg",color)
	# cv2.circle(img,(447,63), 63, (0,0,255), -1)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

############################################################################
print("#####################################################################")

# initialize the window toolkit along with the two image panels
# 띄어쓰기가 indent
testSheet = None
answerSheet = None
studentSheet = None
position = []
answerList = []
trueAnswer = []
studentAnswer = []
main_shape = ()


# 숫자 시험지인지, 영어 시험지인지 선택하게 하는 버튼
# 숫자 시험지면 상태에 0, 영어 시험지면 상태에 1
out = 0
root = Tk()
root.title("Test type")
root.geometry('200x200+200+200')

def selectTypeOfTest():
    global state
    str = ''
    if radVar.get() == 1:
        str = str + '숫자 시험지가 선택되었습니다.'
        state = 1
    if radVar.get() == 2:
        str = str + '영어 시험지가 선택되었습니다.'
        state = 2
    if radVar.get() == False:
        str = str + '아무것도 선택되지 않았습니다. 다시 선택하세여'
    messagebox.showinfo("Button clicked", str)
    str = ''
    if radVar.get() == 1 or radVar.get() == 2:
        root.destroy()

radVar = IntVar()
r1 = ttk.Radiobutton(root, text="Numeric", variable = radVar, value = 1)
r1.grid(column=0, row=0, padx = '10', pady = '10', ipadx = '10', ipady = '10')

r2 = ttk.Radiobutton(root, text="English", variable = radVar, value = 2)
r2.grid(column=0, row=1, padx = '10', pady = '10', ipadx = '10', ipady = '10')

action = ttk.Button(root, text = "Select type of Test", command = selectTypeOfTest)
action.grid(column = 0, row =2, padx = '10', pady = '10', ipadx = '10', ipady = '10')
    
root.mainloop()

#####################################################################
# 시험지를 넣는 UI 창
sheet = Tk()
sheet.title("Auto Scoring")
sheet.geometry('230x250+200+100')

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(sheet, text="Input Test Sheet", command=select1)     # button누르면 select3 실행됨
#btn.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn.grid(column = 0, row = 0, padx = '15', pady = '15', ipadx = '15', ipady = '15')

btn1 = Button(sheet, text="Input Answer Sheet", command=select2)
#btn1.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn1.grid(column = 0, row = 1, padx = '15', pady = '15', ipadx = '15', ipady = '15')


btn2 = Button(sheet, text="Input Student Test Sheet", command=select3)
#btn2.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn2.grid(column = 0, row = 2, padx = '15', pady = '15', ipadx = '15', ipady = '15')

# kick off the GUI
sheet.mainloop()

