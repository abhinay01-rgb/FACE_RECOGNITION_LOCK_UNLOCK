import cv2
import numpy as np 

#Haarcascade files for detection of a feature
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img we will get is in RGB but we have to work on GRAYSCALE
def face_extractor(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=face_classifier.detectMultiScale(gray,1.3,5) #fixing values coordinates for better rsult
	#if face is not present
	if faces is():
		return None
	#if face is present then we have to crop it for better result
	for(x,y,w,h) in faces: #taking coordinate x,y,z,h for rectangle cropping
		cropped_face=img[y:y+h,x:x+w] #cropping from y--> y+h ;x-->x+w

	return cropped_face #returning cropped faces




cap=cv2.VideoCapture(0)
count=0

while True:
	ret,frame=cap.read()
	if face_extractor(frame) is not None: #frame ->>CAMERA ;if frame is not empty then
		count+=1
		face=cv2.resize(face_extractor(frame),(200,200)) #resizing face according to camera in 200,200 pixels


		face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)



		file_name_path='faces/user'+str(count)+'.jpg'  #user-->NAME ; count-->1,2,3.....so on   ; .jpg --> image extension that is going to be saved


		cv2.imwrite(file_name_path,face)

		cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		#PUTTING TEXT OVER FACE (image name,user number,50 50 -->displaying coordinte ,font style,font scale,font color,size)

		cv2.imshow('Face Cropper',face)   #displaying faces in window


	else:  #if face not found else part will work
		print('FACE NOT FOUND')
		pass

	if cv2.waitKey(1)==13 or count==100: # To stop press ENTER key
		break

	
cap.release()  #To stop playing camera
cv2.destroyAllWindows()
print('Collecting Sample Complete')		




