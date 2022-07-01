import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime as dt
from dateutil.tz import gettz
from os.path import exists
import matplotlib.pyplot as plt
from log import attendanceLog



# Preparing the Class
path = 'ImagesAttendance/'
images = []
personNames = []
listOfPerson = os.listdir(path)
for person in listOfPerson:
    current_image = cv2.imread(f'{path}/{person}')
    images.append(current_image)
    personNames.append(os.path.splitext(person)[0])

# print(personNames)


# Function for Encoding the faces of persons
def encodePerson(images):
    encodeList = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

# Getting encoding of all the person
encodedPersons = encodePerson(images)

# from face_recognition.api import face_locations
capture = cv2.VideoCapture(0)
while True:
    flag, image = capture.read()
    if flag == False:
        capture.release()
        print(f"no video/feed")
        #checking if photo is available:
        if exists('ImagesBasic/Vivek Anand.jpg'):
            image = cv2.imread('ImagesBasic/Vivek Anand.jpg')
        else:
            break
    # resize the image

    resized_image = cv2.resize(image,(0,0),None,0.25,0.25)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('Current',image)
    # fl = face_recognition.face_locations(image)
    # fl = face_recognition.face_locations(resized_image)
    # print(f"total faces {len(fl)}")
    # current_frame_faces = face_recognition.face_locations(image)
    current_frame_faces = face_recognition.face_locations(resized_image)
    current_frame_encoding = face_recognition.face_encodings(resized_image,current_frame_faces)
    print(f"total faces detected = {len(current_frame_faces)}")
    for face_encoding,face_location in zip(current_frame_encoding,current_frame_faces):
        matches = face_recognition.compare_faces(encodedPersons,face_encoding)
        face_distance = face_recognition.face_distance(encodedPersons,face_encoding)
        # print(face_distance)
        matchIndex = np.argmin(face_distance)
        # print(matchIndex)
        if face_distance[matchIndex] < 0.50 :
            name = personNames[matchIndex].upper()
            attendanceLog(name)
            print(f"{name} spotted...")
            y1,x2,y2,x1 = face_location
            y1,x2,y2,x1 = 4*y1, 4*x2, 4*y2, 4*x1
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            # cv2.rectangle(image,(x1,y2-30),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(image,name,(x1-10,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            # cv2_imshow(image)
            cv2.imshow('current',image)
        else:
            print(f"Spotted person is not in class")
            name = 'Unknown'
            y1,x2,y2,x1 = face_location
            y1,x2,y2,x1 = 4*y1, 4*x2, 4*y2, 4*x1
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            # cv2.rectangle(image,(x1,y1-10),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(image,name,(x1-10,y2+20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            # cv2_imshow(image)
            cv2.imshow('current', image)
    if flag == False:
        break
    else:
        k = cv2.waitKey(17) & 0xFF

        if k == 27:
            print('exiting....... ')
            cv2.destroyAllWindows()
            break


# print(personNames[0])
