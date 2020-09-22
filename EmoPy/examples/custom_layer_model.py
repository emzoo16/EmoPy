from EmoPy.src.fermodel import FERModel
from EmoPy.src.directory_data_loader import DirectoryDataLoader
from EmoPy.src.data_generator import DataGenerator
from EmoPy.src.neuralnets import CGP_CNN
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import pandas as pd

from pkg_resources import resource_filename, resource_exists

dataset = "ferplus_subset"
model_name = "custom"
batch_size = 64
epochs = 50

model_file_name = dataset + "_" + model_name + \
    str(epochs)

target_dimensions = (48, 48)
channels = 1
verbose = True

validation_split = 0.15

print('--------------- Custom Model -------------------')
print('Loading data...')
directory_path = resource_filename('EmoPy.examples', 'image_data/' + dataset)
data_loader = DirectoryDataLoader(
    datapath=directory_path, validation_split=validation_split)
dataset = data_loader.load_data()


if verbose:
    dataset.print_data_details()

print('Preparing training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator().fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator().fit(test_images, test_labels)

emotion_map = dataset.get_emotion_index_map()

print('Training net...')
model = CGP_CNN(emotion_map, verbose=True)
# Commented because loading existing weights instead of creating new model

# neuralnets.fit_generator not keras.models.fit_generator
# history = model.fit_generator(train_gen.generate(target_dimensions, batch_size=batch_size),
#                               test_gen.generate(
#                                   target_dimensions, batch_size=batch_size),
#                               epochs=epochs)

pretrained_model = model.load_weights(
    './output/checkpoints/custom_dropout_weights.hdf5')

history = model.continue_training_model(pretrained_model, train_gen.generate(target_dimensions, batch_size=batch_size),
                                        test_gen.generate(
    target_dimensions, batch_size=batch_size),
    epochs=epochs)

# neuralnets.fit_generator not keras.models.fit_generator
history = model.fit(train_images, train_labels, epochs=epochs)

hist_df = pd.DataFrame(history.history)

# save to json file
hist_json_file = "output/" + model_file_name + ".json"
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save model configuration (src.nueralnets.save_model)
model.save_model("output/" + model_file_name + ".h5", "output/" +
                 model_file_name + "_emotion_map.json", emotion_map)

print("model successfully saved")
