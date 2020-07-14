import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
mixer.init()
class Camera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		self.model = load_model('models/cnncat2.h5')
		self.sound = mixer.Sound('alarm.wav')
		self.lbl = ['Close','Open']
		self.count = 0
		self.score = 0
		self.thicc=2
		self.rpred=[99]
		self.lpred=[99]
		
	def __del__(self):
		self.video.release()

	def get_frame(self):
		
		face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
		leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
		reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
		while True:
			ret,frame = self.video.read()
			height,width = frame.shape[:2]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		
			faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
			left_eye = leye.detectMultiScale(gray)
			right_eye =  reye.detectMultiScale(gray)
		
			cv2.rectangle(frame, (0,height-50), (200,height), (0,0,0), thickness=cv2.FILLED)
			for (x,y,w,h) in faces:
				cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
		
			for (x,y,w,h) in right_eye:
				r_eye=frame[y:y+h,x:x+w]
				self.count=self.count+1
				r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
				r_eye = cv2.resize(r_eye,(24,24))
				r_eye= r_eye/255
				r_eye=  r_eye.reshape(24,24,-1)
				r_eye = np.expand_dims(r_eye,axis=0)
				self.rpred = self.model.predict_classes(r_eye)
				if(self.rpred[0]==1):
					self.lbl='Open'
				if(self.rpred[0]==0):
					self.lbl='Closed'
				break

			for (x,y,w,h) in left_eye:
				l_eye=frame[y:y+h,x:x+w]
				self.count=self.count+1
				l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
				l_eye = cv2.resize(l_eye,(24,24))
				l_eye= l_eye/255
				l_eye=l_eye.reshape(24,24,-1)
				l_eye = np.expand_dims(l_eye,axis=0)
				self.lpred = self.model.predict_classes(l_eye)
				if(self.lpred[0]==1):
					self.lbl='Open'
				if(self.lpred[0]==0):
					self.lbl='Closed'
				break
			print(self.score)
			if(self.rpred[0]==0 and self.lpred[0]==0):
				self.score=self.score+1
				
				cv2.putText(frame,"Closed",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
			else:
				self.score=self.score-1
				
				cv2.putText(frame,"Open",(10,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL , 1,(255,255,255),1,cv2.LINE_AA)
			if(self.score<0):
				self.score=0
			cv2.putText(frame,'Score:'+str(self.score),(100,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
			if(self.score>15):
				cv2.imwrite(os.path.join(os.getcwd(),'image.jpg'),frame)
				try:
					self.sound.play()
				except:
					pass
				if(self.thicc<16):
					self.thicc= self.thicc+2
				else:
					self.thicc=self.thicc-2
					if(self.thicc<2):
						self.thicc=2
				cv2.rectangle(frame,(0,0),(width,height),(0,0,255),self.thicc)
			ret,jpeg = cv2.imencode('.jpg',frame)
		
			return jpeg.tobytes()





		
		