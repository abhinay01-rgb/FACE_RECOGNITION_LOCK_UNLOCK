import cv2
import numpy as np      # used for any numerical calculation 

from os import listdir      #from OS module importing listdir library :used for fetching DATA from the file

from os.path import isfile,join   


data_path ='C:/Users/A_B_H_I_N_A_Y/Desktop/My_CV2/faces/' #fetching images from file,, last / --> to FETCH image

onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
#Now above we will get data in the form of list which will be store in onlyfiles 

Training_Data,Labels = [],[]  #Training_Data -->Images that captured previously


#Calling Data from the locaation

for i,files in enumerate(onlyfiles):   #enumerate will provide itteration till all images fetches 
	image_path=data_path + onlyfiles[i]    #IT WILL TELL WHICH IMAGE WE ARE TALKING ABOUT BY GIVING INDEX TO IT.
	images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)   #NOW WE WILL READ IMAGES WHICH ARE GRAY IN COLOR
	
	Training_Data.append(np.asarray(images,dtype=np.uint8))  #Now appending all images in the form of array having data type "uint8" 8 bits

	Labels.append(i)  #appending i in Labels


Labels=np.asarray(Labels,dtype=np.int32)

#NOW BUILDING MODEL

model=cv2.face.LBPHFaceRecognizer_create()  #LocalBinaryPatternHistogram(LBPH)


model.train(np.asarray(Training_Data),np.asarray(Labels))  #passing images and index to model for training

#Model Training Done

print("Model Training Done")



#Haarcascade files for detection of a feature
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#FACE DEECTOR 

def face_detector(img,size=0.5):
	#Now we will conver every image into GRAYScale
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#calling faces with obect face_classifier  with the help of detctMultiScale fn
	faces=face_classifier.detectMultiScale(gray,1.3,5)

	if faces is(): #if there is no image
		return img,[]

	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
		#NOW DRWAING RECTANGLE OVER THE FACE WITH COLOR : 0,255,255 with width 2
		roi =img[y:y+h,x:x+w] #roi - region of interest  means BOX
		roi=cv2.resize(roi,(200,200))


	return img,roi




cap=cv2.VideoCapture(0)  #NOW opening camera

while True:
	ret,frame=cap.read()  #reading image 

	image,face=face_detector(frame)  #passing image and getting new image with roi

	try:
		face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
		result=model.predict(face)  #here face will match.

#NOW WE ARE CHECKING HOW MUCH FACE IS MATCHED in percentage
		if result[1]<500:  #psedo value must lie b/t confidence
			confidence=int(100*(1-(result[1])/300))
			display_string=str(confidence)+'% Confidence it is User'
		
		cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,120,255)) #putting text over frame	

		#NOW CHECKING FACE WITH CONFIDENCE VALUE
		if confidence >75:
			cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0))
			cv2.imshow('Face Cropper',image)

		else:
			cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
			cv2.imshow('Face Cropper',image)

				



	except: #if the face is not present then we will handle it in EXCEPT 
		cv2.putText(image,"Face Not Found",(300,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
		cv2.imshow('Face Cropper',image) 
		pass


	if cv2.waitKey(1)==13:
		break


cap.release()
cv2.destroyAllWindows()		








