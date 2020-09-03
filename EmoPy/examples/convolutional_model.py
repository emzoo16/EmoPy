from EmoPy.src.fermodel import FERModel
from EmoPy.src.directory_data_loader import DirectoryDataLoader
from EmoPy.src.data_generator import DataGenerator
from EmoPy.src.neuralnets import ConvolutionalNN
import pandas as pd
import os
from pkg_resources import resource_filename, resource_exists

dataset = "ferplus_subset"
model_type = "conv"
batch_size = 64
epochs = 50

model_name = dataset + "_" + model_type + "_" + str(epochs)

validation_split = 0.15

target_dimensions = (48, 48)
channels = 1
verbose = True

print('--------------- Convolutional Model -------------------')
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
model = ConvolutionalNN(target_dimensions, channels,
                        emotion_map, verbose=True)

history = model.fit_generator(train_gen.generate(target_dimensions, batch_size=batch_size),
                              test_gen.generate(target_dimensions,
                                                batch_size=batch_size),
                              epochs=epochs)

hist_df = pd.DataFrame(history.history)

output_directory = "output/"
try:
    os.mkdir("output/" + model_name)
    output_directory = "output/" + model_name + "/"
except OSError:
    print("Creation of the directory failed. Saving to output directory instead")

# save history to json file
hist_json_file = output_directory + model_name + "_history.json"
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save model configuration (src.nueralnets.save_model)
model.save_model(output_directory + model_name + ".h5",
                 output_directory + model_name + "_emotion_map.json", emotion_map)

print("model successfully saved")
