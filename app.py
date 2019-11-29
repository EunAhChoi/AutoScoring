import Perspective

from flask import Flask, render_template, flash
from flask import redirect
from flask import url_for, request
from flask import send_file
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
app = Flask(__name__)

#testSheet = None
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


## testSheet ##########################
# file이 submit되면 전달되는 페이지
# upload.html에서 form이 제출되면 testSheet_uploaded로 옮겨지게 되어 있음
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
# upload.html에서 form이 제출되면 testSheet_uploaded로 옮겨지게 되어 있음
@app.route('/AnswerSheet_uploaded', methods = ['GET','POST'])
def upload_AnswerSheet():
    global testSheet, main_shape
    if request.method == 'POST': # POST 방식으로 전달된 경우
        f = request.files['EmptySheet']
        # uploads 폴더에 저장
        f.save("./uploads/TestSheet/" + secure_filename(f.filename))
	
        testSheet = cv2.imread("./uploads/TestSheet/"+f.filename)


if __name__ == '__main__':
    # debug를 True로 세팅하면, 해당 서버 세팅 후에 코드가 바뀌어도 문제없이 실행됨. 
    app.run(debug = True)