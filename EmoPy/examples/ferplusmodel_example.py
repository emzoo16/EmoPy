from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2
import os

# target_emotions = ['anger', 'calm', 'happiness']
# target_emotions = ['anger', 'happiness']
# target_emotions = ['calm', 'anger', 'happiness',
#    'surprise', 'disgust', 'fear', 'sadness']
# target_emotions = ['anger', 'fear', 'surprise', 'calm']
# target_emotions = ["surprise", "sadness", "happiness", "anger"]
# target_emotions = ['angry', 'disgust', 'fear',
#                    'happy', 'sad', 'surprise', 'neutral']
# target_emotions = ['happiness', 'surprise', 'disgust']
# target_emotions = ['anger', 'fear', 'surprise']
# target_emotions = ['anger', 'calm', 'happiness']
target_emotions = ['Anger', 'Disgust', 'Fear',
                   'Happy', 'Sad', 'Surprise', 'Neutral']
# target_emotions = ['anger', 'happiness']
# target_emotions = ["anger", "happiness", "neutral", "sadness", "surprise"]

# model = FERModel(target_emotions, verbose=True)

model = model_from_json(
    open("./output/model_4layer_2_2_pool.json", "r").read())
model.load_weights('./output/model_4layer_2_2_pool.h5')

# model = load_model(
#     "./output/lightvgg2.h5")

count = 0

files = [file for file in os.listdir(
    "./image_data/ferplus_subset_test/sadness")if not file.startswith('.')]
print('Predicting on Neutral image...')
for file in files:
    print("\n" + file)

    image = cv2.imread(resource_filename(
        'EmoPy.examples', 'image_data/ferplus_subset_test/sadness' + '/' + file))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_resized = cv2.resize(image, (48, 48),
                               interpolation=cv2.INTER_AREA)
    image_resized = image_resized.astype('float')/255.0
    image_resized = img_to_array(image_resized)
    image_resized = np.expand_dims(image_resized, axis=0)

    predictions = model.predict(image_resized)[0]
    emotion = target_emotions[predictions.argmax()]
    print(emotion)

    # emotion = model.predict(resource_filename(
    #     'EmoPy.examples', 'image_data/ferplus_subset_test/happiness' + '/' + file))
    if(emotion == "Sad"):
        count = count + 1

print(count/len(files))
