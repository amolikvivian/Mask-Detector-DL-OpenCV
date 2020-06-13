import cv2
import sys
import numpy as np 
from keras.models import load_model

#Loading pre-trained model
model = load_model('model/mod-009.model')

#Loading Face Classifier
cascade = 'cascadeClassifier/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascade)

#Initializing Colors and Labels
colors = {0:(0, 255, 0), 1:(0, 0, 255)}
labels = {0:'Mask', 1:'No Mask'}

#Change source to 0 for in-built cam
cam = cv2.VideoCapture(1)

while(True):

	_, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Passinng input image through Haar cascade classifier
	faces = faceCascade.detectMultiScale(gray, 1.3, 5)

	#Extracting Region of Interest - Face 
	for (x, y, w, h) in faces:

	    #Converting Face (RoI) to grayscale image
	    face = gray[y:y+w, x:x+h]
	    #Resizing image to 100x100
	    resized = cv2.resize(face, (100, 100))
	    #Normalizing pixel density
	    normal = resized / 255.0
	    #Reshaping image to pass through network
	    reshaped = np.reshape(normal, (1, 100, 100, 1))

	    #Passing image for predictions
	    result = model.predict(reshaped)
	    #Extracting prediction with maximum confidence
	    label = np.argmax(result, axis = 1)[0]

	    #Drawing rectangle with label
	    cv2.rectangle(img, (x, y), (x+w, y+h), colors[label], 2)
	    cv2.rectangle(img, (x, y-40), (x+w, y), colors[label], -1)
	    cv2.putText(img, labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
	    	(255, 255, 255), 2)

	cv2.imshow('Frame', img)
	
	#Escape Key
	key = cv2.waitKey(1)
	if key == 27 or key == ord('q'):
	    print('[Info] Interrupted')
	    break

#Cleaning Up
cv2.destroyAllWindows()
cam.release()
