# import the necessary packages
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

image_path = 'images'

with open('model.json','r') as f:
    json = f.read()
loaded_model = model_from_json(json)
loaded_model.load_weights("weights.h5", by_name=True)

names = ['Mohammad', 'Yining', 'Stranger']
 
# evaluate loaded model on test data
loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img)):
        x, y, w, h = face

        im = Image.open(os.path.join(image_path, imgname))
        center_x = x+w/2
        center_y = y+h/2

        #create dimensions for face to be cropped
        b_dim = min(max(w,h)*1.2,im.width, im.height)
        box = (center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2)

        crpim = im.crop(box).resize((64,64)) #crops to 64 x 64 image
        imarr = np.array(crpim).astype(np.float32) #turns image into numpy array

        imarr = np.expand_dims(imarr, axis=0)
        out = loaded_model.predict(imarr) #predict who is in the image

        print(out)

        #figure out index of highest percentage
        best_index = np.argmax(out, axis=1)[0] 

        print(names[best_index])

        #draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        texty = y - 10 if y - 10 > 10 else y + 10
        #drawtext of name of face
        cv2.putText(img, names[best_index], (x, texty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.imshow('Image', img)
        cv2.waitKey(0)

face_cascade = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
    print(f)
    save_faces(cascade, f)



