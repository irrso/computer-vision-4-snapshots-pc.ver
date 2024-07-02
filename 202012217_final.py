import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

class WindowClass(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('인생네컷 PC.ver')
        self.setGeometry(200, 200, 880, 500)
        
        self.startBtn = QPushButton('시작', self)
        self.startBtn.setGeometry(10, 10, 100, 30)
        self.startBtn.clicked.connect(self.startFunction)
        
        self.photoBtn = QPushButton('촬영', self)
        self.photoBtn.setGeometry(10, 45, 100, 30)
        self.photoBtn.clicked.connect(self.photoFunction)
        self.photoBtn.setEnabled(False)
        
        self.rephotoBtn = QPushButton('재촬영', self)
        self.rephotoBtn.setGeometry(10, 80, 100, 30)
        self.rephotoBtn.clicked.connect(self.rephotoFunction)
        self.rephotoBtn.setEnabled(False)
        
        self.saveBtn = QPushButton('저장', self)
        self.saveBtn.setGeometry(10, 115, 100, 30)
        self.saveBtn.clicked.connect(self.savcloseFunction)
        self.saveBtn.setEnabled(False)
        
        self.closeBtn = QPushButton('나가기', self)
        self.closeBtn.setGeometry(10, 150, 100, 30)
        self.closeBtn.clicked.connect(self.closeFunction)
        
        self.filter1 = QRadioButton('필터1', self)
        self.filter1.setGeometry(130, 10, 100, 30)
        self.filter1.clicked.connect(self.sunglassFunction)
        self.filter1.setEnabled(False)
        
        self.filter2 = QRadioButton('필터2', self)
        self.filter2.setGeometry(130, 40, 100, 30)
        self.filter2.clicked.connect(self.blushFunction)
        self.filter2.setEnabled(False)
        
        self.filter3 = QRadioButton('필터3', self)
        self.filter3.setGeometry(130, 70, 100, 30)
        self.filter3.clicked.connect(self.heartFunction)
        self.filter3.setEnabled(False)
        
        self.filter4 = QRadioButton('필터4', self)
        self.filter4.setGeometry(130, 100, 100, 30)
        self.filter4.clicked.connect(self.cloudFunction)
        self.filter4.setEnabled(False)
        
        self.filter5 = QRadioButton('필터5', self)
        self.filter5.setGeometry(130, 130, 100, 30)
        self.filter5.clicked.connect(self.macheartFunction)
        self.filter5.setEnabled(False)
        
        self.nofilter = QRadioButton('필터x', self)
        self.nofilter.setGeometry(130, 160, 100, 30)
        self.nofilter.clicked.connect(self.nofilterFunction)
        self.nofilter.setEnabled(False)
        
        self.label = QLabel('0/4', self)
        self.label.setGeometry(150, 450, 100, 30)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setVisible(False)
        
        self.img_label = QLabel('', self)
        self.img_label.setGeometry(230, 10, 640, 480)
    
    def startVideo(self):
        while True:
            ret, self.frame = cap.read()
            if not ret:
                print('프레임 획득 실패')
                break
            
            self.showVideo()
            
            
    def showVideo(self):
        img = cv.cvtColor(cv.flip(self.frame, 1), cv.COLOR_BGR2RGB)
        img = QImage(img.data, self.frame.shape[1], self.frame.shape[0], self.frame.shape[1]*self.frame.shape[2],
                     QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.img_label.setPixmap(pixmap)
        
        cv.waitKey(1)
            
    
    def startFunction(self):
        #시작
        self.startBtn.setEnabled(False)
        self.photoBtn.setEnabled(True)
        self.filter1.setEnabled(True)
        self.filter2.setEnabled(True)
        self.filter3.setEnabled(True)
        self.filter4.setEnabled(True)
        self.filter5.setEnabled(True)
        self.nofilter.setEnabled(True)
        self.label.setVisible(True)
        
        global cap
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.imgs=[]
        
        global mesh
        mp_mesh = mp.solutions.face_mesh 
        mesh = mp_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
        
        self.startVideo()
                
    
    def photoFunction(self):
        # 촬영
        self.imgs.append(cv.flip(self.frame, 1))
        self.label.setText(str(len(self.imgs))+'/4')
        
        if len(self.imgs) > 3:
            self.photoBtn.setEnabled(False)
            self.rephotoBtn.setEnabled(True)
            self.saveBtn.setEnabled(True)
            self.saveBtn.setEnabled(True)
            
            self.stack = cv.resize(self.imgs[0], dsize=(0,0), fx=0.35, fy=0.35)
            for i in range(1, len(self.imgs)):
                self.stack = np.vstack((self.stack, cv.resize(self.imgs[i], dsize=(0,0), fx=0.35, fy=0.35)))
                
                cv.namedWindow(' ')
                cv.moveWindow(' ', 1080, 155)
                cv.imshow(' ', self.stack)
        
   
    def rephotoFunction(self):
        # 재촬영
        self.photoBtn.setEnabled(True)
        self.rephotoBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.imgs=[]
        self.label.setText('0/4')
        cv.destroyWindow(' ')
        
        
    def savcloseFunction(self):
        # 사진 저장
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './', 'Image files (*.jpeg)')
        cv.imwrite(fname[0], self.stack)
        
        
    def closeFunction(self):
        # 나가기
        cap.release()
        cv.destroyAllWindows()
        self.close()

        
    def sunglassFunction(self, state):
        # 선글라스 필터
        while True:
            ret, self.frame = cap.read()
            
            sunglass = cv.imread('sunglass.png', cv.IMREAD_UNCHANGED)
    
            res = mesh.process(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))

            if res.multi_face_landmarks:
                for landmarks in res.multi_face_landmarks:
                    eye_lx, eye_ly = 0, 0
                    eye_rx, eye_ry = 0, 0
                    eye_cx, eye_cy = 0, 0
                    
                    for id, p in enumerate(landmarks.landmark):
                        x, y = int(p.x*self.frame.shape[1]), int(p.y*self.frame.shape[0])
                        if id == 446:
                           eye_lx, eye_ly = x, y
                        if id == 226:
                            eye_rx, eye_ry = x, y
                        if id == 6:
                           eye_cx, eye_cy = x, y
                               
                    eye_w = eye_lx-eye_rx
                    eye_h = int(eye_w/1.8)
                        
                    x1, x2 = int(eye_cx - eye_w), int(eye_cx + eye_w)
                    y1, y2 = int(eye_cy - 1.25*eye_h/2), int(eye_cy + 2.75*eye_h/2)
                    
                    if eye_w > 0 and eye_h > 0 and x1 > 0 and y1 > 0 and x2 < self.frame.shape[1] and y2 < self.frame.shape[0]:
                        sunglasses = cv.resize(sunglass, dsize=(2*eye_w, 2*eye_h))
                        alpha = sunglasses[:, :, 3:]/255
                        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2]*(1-alpha) + sunglasses[:, :, :3]*alpha
                    
                
            self.showVideo()
            
            
    def blushFunction(self, state):
        # 볼터치 필터
       while True:
           ret, self.frame = cap.read()
           
           blush = cv.imread('blush.png', cv.IMREAD_UNCHANGED)
           
           res = mesh.process(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
   
           if res.multi_face_landmarks:
               for landmarks in res.multi_face_landmarks:
                   lcheek_lx, lcheek_ly = 0, 0
                   lcheek_rx, lcheek_ry = 0, 0
                   lcheek_cx, lcheek_cy = 0, 0
                   
                   rcheek_lx, rcheek_ly = 0, 0
                   rcheek_rx, rcheek_ry = 0, 0
                   rcheek_cx, rcheek_cy = 0, 0
                   
                   for id, p in enumerate(landmarks.landmark):
                       x, y = int(p.x*self.frame.shape[1]), int(p.y*self.frame.shape[0])
                       if id == 366:
                           lcheek_lx, lcheek_ly = x, y
                       if id == 358:
                           lcheek_rx, lcheek_ry = x, y
                       if id == 280:
                           lcheek_cx, lcheek_cy = x, y
                       if id == 129:
                           rcheek_lx, rcheek_ly = x, y
                       if id == 137:
                           rcheek_rx, rcheek_ry = x, y
                       if id == 50:
                           rcheek_cx, rcheek_cy = x, y
                           
                   lcheek_w, rcheek_w = lcheek_lx-lcheek_rx, rcheek_lx-rcheek_rx
                   lcheek_h, rcheek_h = lcheek_w, rcheek_w
                       
                   x1, x2 = int(lcheek_cx - lcheek_w), int(lcheek_cx + lcheek_w)
                   y1, y2 = int(lcheek_cy - lcheek_h), int(lcheek_cy + lcheek_h)
                   
                   if lcheek_w > 0 and lcheek_h > 0 and x1 > 0 and y1 > 0 and x2 < self.frame.shape[1] and y2 < self.frame.shape[0]:
                       lblush = cv.resize(blush, dsize=(2*lcheek_w, 2*lcheek_h))
                       alpha = lblush[:, :, 3:]/255
                       self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2]*(1-alpha) + lblush[:, :, :3]*alpha
                       
                   x3, x4 = int(rcheek_cx - rcheek_w), int(rcheek_cx + rcheek_w)
                   y3, y4 = int(rcheek_cy - rcheek_h), int(rcheek_cy + rcheek_h)
                   
                   if rcheek_w > 0 and rcheek_h > 0 and x3 > 0 and y3 > 0 and x4 < self.frame.shape[1] and y4 < self.frame.shape[0]:
                       rblush = cv.resize(blush, dsize=(2*rcheek_w, 2*rcheek_h))
                       alpha = rblush[:, :, 3:]/255
                       self.frame[y3:y4, x3:x4] = self.frame[y3:y4, x3:x4]*(1-alpha) + rblush[:, :, :3]*alpha
               
           self.showVideo()
            
            
    def heartFunction(self, state):
        # 볼하트 필터
        while True:
            ret, self.frame = cap.read()
            
            heart = cv.imread('heart.png', cv.IMREAD_UNCHANGED)
            
            res = mesh.process(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
    
            if res.multi_face_landmarks:
                for landmarks in res.multi_face_landmarks:
                    lcheek_lx, lcheek_ly = 0, 0
                    lcheek_rx, lcheek_ry = 0, 0
                    lcheek_cx, lcheek_cy = 0, 0
                    
                    rcheek_lx, rcheek_ly = 0, 0
                    rcheek_rx, rcheek_ry = 0, 0
                    rcheek_cx, rcheek_cy = 0, 0
                    
                    for id, p in enumerate(landmarks.landmark):
                        x, y = int(p.x*self.frame.shape[1]), int(p.y*self.frame.shape[0])
                        if id == 366:
                            lcheek_lx, lcheek_ly = x, y
                        if id == 358:
                            lcheek_rx, lcheek_ry = x, y
                        if id == 280:
                            lcheek_cx, lcheek_cy = x, y
                        if id == 129:
                            rcheek_lx, rcheek_ly = x, y
                        if id == 137:
                            rcheek_rx, rcheek_ry = x, y
                        if id == 50:
                            rcheek_cx, rcheek_cy = x, y
                            
                    lcheek_w, rcheek_w = lcheek_lx-lcheek_rx, rcheek_lx-rcheek_rx
                    lcheek_h, rcheek_h = lcheek_w, rcheek_w
                        
                    x1, x2 = int(lcheek_cx - lcheek_w/2), int(lcheek_cx + lcheek_w/2)
                    y1, y2 = int(lcheek_cy - lcheek_h/2), int(lcheek_cy + lcheek_h/2)
                    
                    if lcheek_w > 0 and lcheek_h > 0 and x1 > 0 and y1 > 0 and x2 < self.frame.shape[1] and y2 < self.frame.shape[0]:
                        lheart = cv.resize(heart, dsize=(lcheek_w, lcheek_h))
                        alpha = lheart[:, :, 3:]/255
                        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2]*(1-alpha) + lheart[:, :, :3]*alpha
                        
                    x3, x4 = int(rcheek_cx - rcheek_w/2), int(rcheek_cx + rcheek_w/2)
                    y3, y4 = int(rcheek_cy - rcheek_h/2), int(rcheek_cy + rcheek_h/2)
                    
                    if rcheek_w > 0 and rcheek_h > 0 and x3 > 0 and y3 > 0 and x4 < self.frame.shape[1] and y4 < self.frame.shape[0]:
                        rheart = cv.resize(heart, dsize=(rcheek_w, rcheek_h))
                        alpha = rheart[:, :, 3:]/255
                        self.frame[y3:y4, x3:x4] = self.frame[y3:y4, x3:x4]*(1-alpha) + rheart[:, :, :3]*alpha
                
            self.showVideo()
            
    
    def cloudFunction(self, state):
        # 구름 필터
        while True:
            ret, self.frame = cap.read()
            
            cloud = cv.imread('cloud.png', cv.IMREAD_UNCHANGED)
            
            res = mesh.process(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
    
            if res.multi_face_landmarks:
                for landmarks in res.multi_face_landmarks:
                    head_x, head_y = 0, 0
                    
                    for id, p in enumerate(landmarks.landmark):
                        x, y = int(p.x*self.frame.shape[1]), int(p.y*self.frame.shape[0])
                        if id == 67:
                            head_lx, head_ly = x, y
                        if id == 287:
                            head_rx, head_ry = x, y
                        if id == 10:
                            head_cx, head_cy = x, y
                               
                    head_w = head_rx-head_lx
                    head_h = int(head_w/1.54)
                    
                    x1, x2 = int(head_cx - 3*head_w/2), int(head_cx + 3*head_w/2)
                    y1, y2 = int(head_cy - 7*head_h/2), int(head_cy - head_h/2)
                    
                    if head_w > 0 and head_h > 0 and x1 > 0 and y1 > 0 and x2 < self.frame.shape[1] and y2 < self.frame.shape[0]:
                        clouds = cv.resize(cloud, dsize=(3*head_w, 3*head_h)) 
                        alpha = clouds[:, :, 3:]/255
                        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2]*(1-alpha) + clouds[:, :, :3]*alpha
                
            self.showVideo()
         
            
    def macheartFunction(self, state):
        # 맥하트 필터
        while True:
            ret, self.frame = cap.read()
            
            heart = cv.imread('mac_heart.png', cv.IMREAD_UNCHANGED)
            
            res = mesh.process(cv.cvtColor(self.frame, cv.COLOR_BGR2RGB))
    
            if res.multi_face_landmarks:
                for landmarks in res.multi_face_landmarks:
                    head_x, head_y = 0, 0
                    
                    for id, p in enumerate(landmarks.landmark):
                        x, y = int(p.x*self.frame.shape[1]), int(p.y*self.frame.shape[0])
                        if id == 67:
                            head_lx, head_ly = x, y
                        if id == 287:
                            head_rx, head_ry = x, y
                        if id == 10:
                            head_cx, head_cy = x, y
                               
                    head_w = head_rx-head_lx
                    head_h = int(head_w/2.53)
                    
                    x1, x2 = int(head_cx - 3*head_w/2), int(head_cx + 3*head_w/2)
                    y1, y2 = int(head_cy - 7*head_h/2), int(head_cy - head_h/2)
                    
                    if head_w > 0 and head_h > 0 and x1 > 0 and y1 > 0 and x2 < self.frame.shape[1] and y2 < self.frame.shape[0]:
                        hearts = cv.resize(heart, dsize=(3*head_w, 3*head_h)) 
                        alpha = hearts[:, :, 3:]/255
                        self.frame[y1:y2, x1:x2] = self.frame[y1:y2, x1:x2]*(1-alpha) + hearts[:, :, :3]*alpha
                
            self.showVideo()
    
            
    def nofilterFunction(self):
        # 필터없음
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.startVideo()
        self.showVideo()
 
    
app = QApplication(sys.argv)
win = WindowClass()
win.show()
app.exec_()