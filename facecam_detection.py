import numpy as np
from PIL import Image
import cv2
import os
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

from keras.models import model_from_json
from keras_vggface.vggface import VGGFace

from tensorflow.python.client import device_lib


face_cascade = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(face_cascade)

with open('model.json','r') as f:
    json = f.read()
loaded_model = model_from_json(json)
loaded_model.load_weights("weights.h5", by_name=True)

names = ['Mohammad', 'Yining', 'Stranger']

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv2.CASCADE_SCALE_IMAGE
    )
    #save frame as an image to be opened later
    cv2.imwrite("frame.jpg", frame)
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        im = Image.open("frame.jpg")
        center_x = x+w/2
        center_y = y+h/2

        #Create dimensions for face to be cropped
        b_dim = min(max(w,h)*1.2,im.width, im.height)
        box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)

        crpim = im.crop(box).resize((64,64)) #crops to 64 x 64 image
        imarr = np.array(crpim).astype(np.float32) #turns image into numpy array

        imarr = np.expand_dims(imarr, axis=0)
        out = loaded_model.predict(imarr)
        print(out) #print outputted percentages
        best_index = np.argmax(out, axis=1)[0]
        print(names[best_index])
        texty = y - 10 if y - 10 > 10 else y + 10 
        cv2.putText(frame, names[best_index], (x, texty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        

        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
