# -*- coding: utf-8 -*-
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

    global testSheet  # 전역변수선언. 시험지를 저장하게 된다.
    path = filedialog.askopenfilename() # 파일 열기 모듈 method 사용. path에 경로 저장. 
	                                	# GUI 창이 열리고 거기에다가 입력하게 된다!
    testSheet = cv2.imread(path,0) # 이미지 파일을 읽기 위한 객체를 리턴해주는 함수.
	# 0은 gray로 읽겠다는 의미 (cv2.IMREAD_GRAYSCALE)

	# 사진 픽셀 가져옴
    testSheet_shape = testSheet.shape
    print(testSheet_shape)
	# 명암 조절
    testSheet2 = testSheet +100
    testSheet3 = testSheet -100
    #cv2.imshow("test", testSheet)
    #cv2.imshow("-100", testSheet2)
    #cv2.imshow("+100", testSheet3)


def select2():         # 정답 and 좌표찾기
	# answerSheet : 정답지 저장하는 곳. position : ??, answerList : 정답 추출?
	global answerSheet, position, answerList 
	path = filedialog.askopenfilename() # 파일 열기 모듈 method 사용. path에 경로 저장.
	answerSheet = cv2.imread(path,0) # 답지 경로 찾아서 이미지 파일 객체 생성

	print(answerSheet.shape)
	
	#answerSheet2 = answerSheet + 200
	#cv2.imshow("original", answerSheet)
	#cv2.imshow("modification", answerSheet2)

    # 이미지 노이즈 제거. (http://www.gisdeveloper.co.kr/?p=7168)
	#dst = cv2.fastNlMeansDenoising(answerSheet, None, 10, 7, 21)
	# 픽셀 차이 보기
	pixel(testSheet, answerSheet)

	# 이미치 차이 score
	show_score(testSheet, answerSheet)

	# 색으로 구별
	dif_color()

	
	# 이미지 서로 다른 부분 찾는 코드    
	# 정확히 두 이미지간의 다른 부분의 (x, y)-coordinate location을 찾아줌.
	# compare_ssim : 두 이미지 사이의 구조적 유사성 지수를 계산하여 different 이미지가 반환되도록 한다.
	# 시험지가 저장된 객체와 답지가 저장된 객체를 파라미터로
	# https://ng1004.tistory.com/89 <-- 여기서 퍼온듯
	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	
	(score, diff) = compare_ssim(testSheet,answerSheet, full=True)
	#diff = cv2.absdiff(testSheet, answerSheet)
	
	# score는 두 이미지의 Structural Similarity index를 저장. 범위는 -1~1까지. 1은 perfect match를 뜻함.
	# diff는 실제 차이 이미지를 저장한다. floating point data 로 저장되며 0~1까지 범위를 가짐
	# 우리는 이를 8bit unsigned integer (0~255)로 이루어진 array로 convert해야됨. (OpenCV를 이용하기 위해) 
	
	diff = (diff*255).astype("uint8")
	
	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	# OpenCV 이미지 프로세싱에서 thresholding을 적용하려면 grayscale이미지로 변환하여 적용해야한다.

	#thresh = cv2.threshold(diff, 130,255,cv2.THRESH_BINARY)
	#thresh = cv2.threshold(diff,130,255,cv2.THRESH_BINARY)
	thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)

	cv2.imwrite("/Users/hcy/Desktop/result3.png", thresh)
	# thresh 변수의 경계(윤곽선)를 찾음.
	
	#cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cnts = cnts[0] if imutils.is_cv2() else cnts[1]


	# 이미지의 차이를 실제 png 파일로 만들어주는 코드 추가
	diff = cv2.absdiff(testSheet, answerSheet)
	mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)

	ret,img_binary=cv2.threshold(mask, 130,255,cv2.THRESH_BINARY)
	cv2.imwrite("/Users/hcy/Desktop/result6.png", img_binary)

	img_temp = cv2.imread("/Users/hcy/Desktop/result6.png",cv2.IMREAD_COLOR)

	cnts= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	#result = pytesseract.image_to_string(Image.open("/Users/hcy/Desktop/result6.png"),lang='eng')
	#result = result.replace(" ","")
	#result = str(result)

	#print(result)

	#if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
	#	os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))
	#f = open("/Users/hcy/Desktop/GP/answerSheet/answerListTest.txt","w",-1,"utf-8")
	#f.write(result)
	#f.close()

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
			img.append(answerSheet[minY:maxY, minX:maxX])    # img == 최종 답안 단어들의 이미지를 저장한 리스트
	#print(len(answerList))
	answerList = []
	for i in range(0, len(img)):
		cv2.imwrite("/Users/hcy/Desktop/GP/trueAnswer/"+""+str(i) + ".jpg", img[i])

	# 이 코드가 실행하고 있는 위치에 answer이라는 폴더만들면 정답지가 answer폴더안에 저장       dir있는지 확인하고 없으면 만드는 코드로 수정
	for i in range(len(img)):
		#cv2.imshow(str(i), img[i])
		# resize = cv2.resize(img[i], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)

		'''gau = cv2.GaussianBlur(resize, (5, 5), 0)
		temp = cv2.addWeighted(resize, 1.5, gau, -0.5, 0)

		kernel = np.ones((2,2), np.uint8)
		er = cv2.erode(temp, kernel, iterations=1)'''

		#cv2.imshow("zzz", er)
		#tessdata_dir_config = r'--tessdata-dir "<C:\Program Files (x86)\Tesseract-OCR\tessdata>"'

		#cv2.imwrite("answer\\"+"ss"+str(i) + ".jpg", er)

		# OCR 기능을 위해 pytesseract 이용
		result = pytesseract.image_to_string(img[i],lang='eng')
		result = result.replace(" ","")
		result = str(result)
		answerList.append(result)
		#print(result)

	if not (os.path.isdir("/Users/hcy/Desktop/GP/answerSheet")):
		os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answerSheet"))

	f = open("/Users/hcy/Desktop/GP/answerSheet/answerList.txt","w",-1,"utf-8")
	for i in range(len(answerList)):
		f.write(answerList[i]+"\n")
	f.close()

	#cv2.waitKey(0)  # esc키 누르면 나온 답지 꺼짐
	#cv2.destroyAllWindows()

def select3():         # 학생들 정답 찾기 & 정답과 비교, 채점해서 출력
	global studentSheet
	studentAnswer = []
	path = filedialog.askopenfilename()
	studentSheet = cv2.imread(path,0)
	if not (os.path.isdir("/Users/hcy/Desktop/GP/answer")):
		os.makedirs(os.path.join("/Users/hcy/Desktop/GP/answer"))
	'''for i in range(0,len(position)):
		studentAnswer.append(studentSheet[position[i][0]:position[i][1],position[i][2]:position[i][3]])
		cv2.imwrite("answer\\"+str(i)+".jpg",studentAnswer[i])'''

	#cv2.imshow("test",testSheet)
	#cv2.imshow("answer",answerSheet)
	#cv2.imshow("student",studentSheet)

	# 이미지 서로 다른 부분 찾는 코드 위에 상세한 설명
	# 이미지 서로 다른 부분 찾는 코드    
	# 정확히 두 이미지간의 다른 부분의 (x, y)-coordinate location을 찾아줌.
	# compare_ssim : 두 이미지 사이의 구조적 유사성 지수를 계산하여 different 이미지가 반환되도록 한다.
	# 시험지가 저장된 객체와 답지가 저장된 객체를 파라미터로
	# https://ng1004.tistory.com/89 <-- 여기서 퍼온듯
	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(testSheet, studentSheet, full=True)
	# score는 두 이미지의 Structural Similarity index를 저장. 범위는 -1~1까지. 1은 perfect match를 뜻함.
	# diff는 실제 차이 이미지를 저장한다. floating point data 로 저장되며 0~1까지 범위를 가짐
	# 우리는 이를 8bit unsigned integer (0~255)로 이루어진 array로 convert해야됨. (OpenCV를 이용하기 위해)
	diff = (diff * 255).astype("uint8")
	# threshold (임계값)을 찾음... 뭔가 조정하는게 있는거 같음. (매우 이해하기 어려움)
	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	# thresh 변수의 경계(윤곽선)를 찾음.
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

	#zz=cv2.imread(path,1)
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
			img2.append(studentSheet[minY:maxY, minX:maxX])  # img == 최종 답안 단어들의 이미지를 저장한 리스트
			#cv2.rectangle(zz,(minX,minY),(maxX,maxY),(0,0,255),3)
			pos = []
			pos.append(minY)
			pos.append(maxY)
			pos.append(minX)
			pos.append(maxX)
			position.append(pos)
	#cv2.imwrite("/Users/hcy/Desktop/GP/myanswer.jpg",zz)
	for i in range(0, len(img2)):
		#print(img2[i]) #홍
		cv2.imwrite("/Users/hcy/Desktop/GP/answer/"+""+str(i) + ".jpg", img2[i])
	os.chdir("/Users/hcy/Desktop/GP/src/")
	os.system("python3 main.py")           # main.py 실행하면 answerImage에 있는 폴더 모두 실행, txt파일에 정답 저장

	# blank 처리 - len(answerList) - studentAnswerList
	#r = open('C:\\Users\yea\.spyder-py3\\answerSheet\\answerwordLists.txt','rt')
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
	answerList1 = []
	with codecs.open('/Users/hcy/Desktop/GP/answerSheet/answerListTest.txt','r',encoding='utf-8') as g:
		while(1):
			line = g.readline()
			try:escape=line.index('\r\n')
			except:
				escape = len(line)

			if line:
				answerList1.append(line[0:escape].replace(" ", ""))
			else:
				break
	g.close()

	print(answerList1)
	print(studentAnswer)

	score = len(studentAnswer)
	print("길이 ", len(studentAnswer)) # 홍
	correctNum = np.zeros(len(answerList1))
	for i in range(0,len(studentAnswer)):
		correct = 0
		for j in range(0,len(answerList)):              # answerList는 순서대로 저장돼 있으므로 j에 따라 채점
			if(studentAnswer[i] == answerList[j]):
				correct = 1
				correctNum[j]=1
				break
		if correct==0:
			print(studentAnswer[i])
			score-=1
	'''for item in studentAnswer:
		correct = 0
		for j in range(0,len(answerList1)):
			if eq(item,answerList1[j]):
				correct = 1
				correctNum[j] = 1
		if correct==0:
			print(item)
			score-=1'''
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

# initialize the window toolkit along with the two image panels
testSheet = None
answerSheet = None
studentSheet = None
position = []
answerList = []


# 숫자 시험지인지, 영어 시험지인지 선택하게 하는 버튼
root = Tk()
root.title("Test type")
root.geometry('250x300+200+200')

def selectTypeOfTest():
    str = ''
    if radVar.get() == 1:
        str = str + '숫자 시험지가 선택되었습니다.'
    if radVar.get() == 2:
        str = str + '영어 시험지가 선택되었습니다.'
    messagebox.showinfo("Button clicked", str)

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
sheet.geometry('270x500+200+100')

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(sheet, text="Input Test Sheet", command=select1)     # button누르면 select3 실행됨
#btn.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn.grid(column = 0, row = 0, padx = '25', pady = '25', ipadx = '25', ipady = '25')

btn1 = Button(sheet, text="Input Answer Sheet", command=select2)
#btn1.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn1.grid(column = 0, row = 1, padx = '25', pady = '25', ipadx = '25', ipady = '25')


btn2 = Button(sheet, text="Input Student Test Sheet", command=select3)
#btn2.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn2.grid(column = 0, row = 2, padx = '25', pady = '25', ipadx = '25', ipady = '25')

# kick off the GUI
sheet.mainloop()
