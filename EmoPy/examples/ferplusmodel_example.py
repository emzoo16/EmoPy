from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2
import os
from EmoPy.src.customLayer import SliceLayer, ChannelShuffle, PadZeros

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
# target_emotions = ['Anger', 'Disgust', 'Fear',
#                    'Happy', 'Sad', 'Surprise', 'Neutral']
target_emotions = ['sadness', 'disgust', 'calm',
                   'fear', 'happiness', 'anger', 'surprise']
# target_emotions = ['anger', 'happiness']
# target_emotions = ["anger", "happiness", "neutral", "sadness", "surprise"]

model = FERModel(target_emotions, verbose=True)

# model = load_model(
#     "./output/custom.h5", custom_objects={'SliceLayer': SliceLayer, 'ChannelShuffle': ChannelShuffle,
#                                           'PadZeros': PadZeros})

# model = model_from_json(
#     open("./output/model_4layer_2_2_pool.json", "r").read())
# model.load_weights('./output/model_4layer_2_2_pool.h5')

# model = load_model(
#     "./output/lightvgg2.h5")

count = 0
target_emotion = "anger"

print('Predicting on Neutral image...')

files = [file for file in os.listdir(
    "./image_data/FER2013Test_Sorted/" + target_emotion)if not file.startswith('.')]

for file in files:
    print("\n" + file)
    emotion = model.predict(resource_filename(
        'EmoPy.examples', 'image_data/FER2013Test_Sorted/' + target_emotion + '/' + file))
    if(emotion == target_emotion):
        count = count + 1

print(count/len(files))
