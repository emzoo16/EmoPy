from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
import cv2
import numpy as np
from keras.models import model_from_json
import json

# Choose a set of emotions to identify
# target_emotions = ['calm', 'anger', 'happiness']
# target_emotions = ['calm', 'anger', 'happiness', 'surprise',
#                    'disgust', 'fear', 'sadness']
# target_emotions = ['anger', 'fear', 'surprise', 'calm']
# target_emotions = ['happiness', 'disgust', 'surprise']
# target_emotions = ['anger', 'fear', 'surprise']
# target_emotions = ['anger', 'fear', 'calm']
# target_emotions = ['anger', 'happiness', 'calm']
# target_emotions = ['calm', 'disgust', 'surprise']
# target_emotions = ['sadness', 'disgust', 'surprise']
# target_emotions = ['anger', 'happiness']
target_emotions = ['anger', 'happiness', 'surprise']

#model = FERModel(target_emotions, verbose=True)

model = load_model("./output/fer2013_conv_dropout.h5")

face_detector = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    # Get a single frame from the input camera.
    ret, frame = cap.read()
    labels = []

    # Convert to greyscale to remove some noise.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Cut out the face region in the frame and resize it to the specified size.
        roi_gray = gray[y:y+h, x:x+w]

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi_gray)[0]
            prediction_position = (x, y)
            print(prediction)

            cv2.putText(frame, prediction, prediction_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Window should close when q is pressed, or ctrl+c if it doesn't work
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
