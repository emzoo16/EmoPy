from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json
import json
from EmoPy.src.customLayer import SliceLayer, ChannelShuffle, PadZeros

target_dimensions = (48, 48)
channels = 1

take_picture = True

face_detector = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# Load a model. Choose from /output folder
classifier = load_model("./output/custom.h5", custom_objects={'SliceLayer': SliceLayer,
                                                              'ChannelShuffle': ChannelShuffle, 'PadZeros': PadZeros})

# classifier = model_from_json(
#     open("./output/customtest.json", "r").read(), custom_objects={'SliceLayer': SliceLayer, 'ChannelShuffle': ChannelShuffle,
#                                                                   'PadZeros': PadZeros})
# classifier.load_weights('./output/custom_weights.h5')

emotion_map = json.loads(
    open("./output/custom_emotion_map_0123456.json").read())


def process_image_for_prediction(image, take_picture):
    resized_image = cv2.resize(
        image, target_dimensions, interpolation=cv2.INTER_LINEAR)
    final_image = np.array([np.array([resized_image]).reshape(
        list(target_dimensions)+[channels])])
    return final_image


def print_prediction(prediction):
    normalized_prediction = [x/sum(prediction) for x in prediction]
    for emotion in emotion_map.keys():
        print('%s: %.1f%%' %
              (emotion, normalized_prediction[emotion_map[emotion]]*100))
    print("\n")
    dominant_emotion_index = np.argmax(prediction)

    for emotion in emotion_map.keys():
        if dominant_emotion_index == emotion_map[emotion]:
            dominant_emotion = emotion
            break

    sadness_index = emotion_map["sadness"]
    if(normalized_prediction[sadness_index] > 0.2):
        dominant_emotion = "sadness"

    anger_index = emotion_map["anger"]
    if(normalized_prediction[anger_index] > 0.05):
        dominant_emotion = "anger"

    return dominant_emotion


cap = cv2.VideoCapture(0)

while True:
    # Get a single frame from the input camera.
    ret, frame = cap.read()

    # Convert to greyscale to remove some noise.
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.3, 5)

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Cut out the face region in the frame and resize it to the specified size.
        roi_gray = grey[y:y+h, x:x+w]

        if np.sum([roi_gray]) != 0:
            processed_image = process_image_for_prediction(
                roi_gray, take_picture)

            predictions = classifier.predict(processed_image)[0]

            label = print_prediction(predictions)
            label_position = (x, y)

            print("emotion: " + label)

            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            print("no face found")
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Window should close when q is pressed, or ctrl+c if it doesn't work
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
