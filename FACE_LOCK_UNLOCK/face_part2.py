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




