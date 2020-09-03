from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json
import json

# The size of the image of the face used for classification:
# 48x48 for fer2013
# 100x100 for Cohn-Kanade
# 64x64 for fer2013_oarriaga.hdf5 and fer2013_omar178.hdf5
img_width = 64
img_height = 64

face_detector = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Load a model. Choose from /output folder
classifier = load_model(
    "./output/fer2013_oarriaga.hdf5")

# classifier = model_from_json(
#     open("./output/model_4layer_2_2_pool.json", "r").read())
# classifier.load_weights('./output/model_4layer_2_2_pool.h5')

# For fer_25 and fer_5, use this instead:
# classifier = model_from_json(
#     open("./models/model_4layer_2_2_pool.json", "r").read())
# classifier.load_weights('./models/model_4layer_2_2_pool.h5')

# These labels are used for models that use fer2013.
# class_labels = ['angry', 'disgust', 'fear',
#                 'happy', 'sad', 'surprise', 'neutral']

# class_labels = ['surprise', 'happiness', 'anger']
class_labels = ['Anger', 'Anger', 'Surprise',
                'Happy', 'Sad', 'Surprise', 'Neutral']
# These labels are used for "ck_aswinMatthewsAshok.h5" which use Cohn-Kanade.
# class_labels = ["Neutral", "Angry", "Contempt",
#                 "Disgust", "Fear", "Happy", "Sadness", "Surprise"]
cap = cv2.VideoCapture(0)

while True:
    # Get a single frame from the input camera.
    ret, frame = cap.read()
    labels = []

    # Convert to greyscale to remove some noise.
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Cut out the face region in the frame and resize it to the specified size.
        roi_gray = grey[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (img_width, img_height),
                              interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        # make a prediction on the region of interest, print the corresponding class
            predictions = classifier.predict(roi)[0]
            print(predictions)
            label = class_labels[predictions.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
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
