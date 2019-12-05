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

def onMouse(x):
    pass

def imgBlending(img1, img2):
	cv2.namedWindow('ImgPane')
	cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
	mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

	while True:
		img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
		cv2.imshow('ImgPane', img)

		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
		
		mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
	
	cv2.destroyAllWindows()

def select1():        # 시험지 선택 함수

	global testSheet, main_shape  # 전역변수선언. 시험지를 저장하게 된다.
	path = filedialog.askopenfilename()
	
	testSheet = cv2.imread(path, 0) # 이미지 파일을 읽기 위한 객체를 리턴해주는 함수.
	# 0은 gray로 읽겠다는 의미 (cv2.IMREAD_GRAYSCALE)
	#main_shape = testSheet.shape
	
	testSheet = cv2.imread("/Users/hcy/Desktop/GP/Examples/empty.jpeg")

	print(testSheet.shape)
	
	'''
	testSheet = cv2.cvtColor(testSheet, cv2.COLOR_BGR2GRAY)

	testShape = main_shape[:2]
	print(testShape)
	testSheet = cv2.resize(testSheet, dsize = testShape, interpolation=cv2.INTER_AREA)

	kernel = np.ones((3,3), np.uint8)

	testSheet = cv2.erode(testSheet, kernel, iterations = 1)

	'''



##############################################################################

# 인덴트 탭
def select2():         # 정답 and 좌표찾기
	# answerSheet : 정답지 저장하는 곳. position : ??, answerList : 정답 추출?
	global position, answerList, answerSheet, answer_shape
	path = filedialog.askopenfilename() # 파일 열기 모듈 method 사용. path에 경로 저장.
	answerSheet = cv2.imread(path,0) # 답지 경로 찾아서 이미지 파일 객체 생성

	answerSheet = cv2.imread("/Users/hcy/Desktop/GP/Examples/ClearNumber.jpeg")

	#answerSheet = cv2.cvtColor(answerSheet,cv2.COLOR_BGR2RGB)

	##shape = (10000, 10000)
	##answerSheet = cv2.resize(answerSheet, dsize=shape, interpolation=cv2.INTER_AREA)
	#plt.imshow(answerSheet)
	#plt.show()

	# 팽창 해보자. Dilation
	kernel = np.ones((3,3), np.uint8)

	dilation = cv2.erode(answerSheet, kernel, iterations = 1)

	cv2.namedWindow('test', cv2.WINDOW_NORMAL)
	cv2.imshow('test', dilation)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# 노이즈 제거하는 코드
	'''
	noise = cv2.medianBlur(answerSheet, 5)
	hsv_noise = cv2.cvtColor(noise, cv2.COLOR_RGB2HSV)

	plt.imshow(noise)
	plt.show()

	plt.imshow(hsv_noise)
	plt.show()
	'''
	# 흑백을 좀 더 선명하게 바꾸는 코드 
	'''
	kernel = np.array([[0, -1, 0],
						[-1, 5, -1],
						[0, -1, 0]])

	gray = cv2.imread("/Users/hcy/Desktop/GP/예제시험지/Real.jpeg", cv2.IMREAD_GRAYSCALE)

	image_sharp = cv2.filter2D(gray, -1, kernel)

	plt.imshow(image_sharp, cmap="gray"), plt.axis("off")
	plt.show()

	'''

	#imgBlending(answerSheet, answerSheet)

	# 두 이미지 더하기
	
	#answerSheet = cv2.cvtColor(answerSheet,cv2.COLOR_RGB2GRAY)
	#temp = answerSheet

	#result = answerSheet + temp

	
	##plt.imshow(answerSheet)
	#plt.show()
	
	answerSheet = Perspective.transform(dilation, testSheet)



	'''

	answerSheet = cv2.cvtColor(answerSheet, cv2.COLOR_BGR2GRAY)

	# 이미지의 차이를 실제 png 파일로 만들어주는 코드 추가
	diff = cv2.absdiff(testSheet, answerSheet)
	mask = cv2.cvtColor(diff, cv2.COLOR_BAYER_BG2GRAY)
	#diff = cv2.GaussianBlur(diff,(3,3),0)
	# "펄스펙티브에 있던 녀석임 밑이 ㅇㅋ?"
	#mask = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

	ret,img_binary=cv2.threshold(mask, 110,255,cv2.THRESH_BINARY)
	cv2.imwrite("/Users/hcy/Desktop/두이미지차이.png", img_binary)

	_, cnts, _= cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cnts = cnts[0] if imutils.is_cv2() else cnts[1]


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

	'''

	#Perspective.drawbox(answerSheet)
	#answer_shape = answerSheet.shape
	
	#
	# too many values to unpack
	# 설정한 변수의 개수와 리턴해 주는 변수의 개수가 차이날 때 발생한다.
	#





#################################################################################

def select3():         # 학생들 정답 찾기 & 정답과 비교, 채점해서 출력
	global studentSheet
	path = filedialog.askopenfilename()
	studentSheet = cv2.imread(path,0)
	

def select4():
	print("kkk")
	global main_shape
	temp = cv2.imread("/Users/hcy/Desktop/GP/Examples/ClearNumber.jpeg")
	main_shape = temp.shape
	print(main_shape)


	

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
answer_shape = ()
student_shape = ()


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
sheet.geometry('230x350+200+100')

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

btn3 = Button(sheet, text="AutoScoring", command=select4)
#btn2.pack(side="bottom", fill="both", expand="True", padx="50", pady="50", ipadx="50", ipady="50")
btn3.grid(column = 0, row = 3, padx = '15', pady = '15', ipadx = '15', ipady = '15')

# kick off the GUI
sheet.mainloop()

