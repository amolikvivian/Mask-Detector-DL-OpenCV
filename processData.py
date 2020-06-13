import os
import cv2
import numpy as np 

from keras.utils import np_utils

dataPath = 'dataset'

categories = os.listdir(dataPath)
labels = [i for i in range(len(categories))]

labelDict = dict(zip(categories, labels))

imgSize = 100
data = []
label = []

#Extracting image names
for category in categories:
	folderPath = os.path.join(dataPath, category)
	imgNames = os.listdir(folderPath)

	#Reading individual image
	for imgName in imgNames:
		imgpPath = os.path.join(folderPath, imgName)
		img = cv2.imread(imgpPath)

		#Grayscaling and Resizing image 100x100
		try:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized = cv2.resize(gray, (imgSize, imgSize))

			#Appending data and label list
			data.append(resized)
			label.append(labelDict[category])

		except Exception as e:
			print('Exception:', e)

#Converting images to numpy array
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], imgSize, imgSize, 1))

#One-Hot Encoding Labels
label = np.array(label)
labelOneHot = np_utils.to_categorical(label)

#Saving numpy arrays
np.save('savedData/data', data)
np.save('savedData/target', labelOneHot)