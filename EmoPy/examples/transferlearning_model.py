
#from EmoPy.src.csv_data_loader import CSVDataLoader
from EmoPy.src.directory_data_loader import DirectoryDataLoader
from EmoPy.src.neuralnets import TransferLearningNN
from EmoPy.src.data_generator import DataGenerator
from keras.models import load_model
import pandas as pd

from pkg_resources import resource_filename

from keras import backend as K
K.set_image_data_format("channels_last")

validation_split = 0.15
verbose = True
model_name = 'vgg16'
dataset = "ferplus_subset"

epochs = 10
batch_size = 64

model_file_name = dataset + "_" + model_name + \
    "_" + str(epochs) + "_" + str(batch_size)

target_dimensions = (128, 128)
raw_dimensions = (48, 48)

print('--------------- ' + model_name + ' -------------------')
print('Loading data...')
# csv_file_path = resource_filename('EmoPy.examples', 'image_data/sample.csv')
# data_loader = CSVDataLoader(target_emotion_map=fer_dataset_label_map, datapath=csv_file_path,
#                             validation_split=validation_split, image_dimensions=raw_dimensions, csv_label_col=0, csv_image_col=1, out_channels=3)
# dataset = data_loader.load_data()

directory_path = resource_filename('EmoPy.examples', 'image_data/' + dataset)
data_loader = DirectoryDataLoader(
    datapath=directory_path, validation_split=validation_split)
dataset = data_loader.load_data()

emotion_map = dataset.get_emotion_index_map()

if verbose:
    dataset.print_data_details()

print('Creating training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator().fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator().fit(test_images, test_labels)

print('Initializing neural network with InceptionV3 base model...')
model = TransferLearningNN(model_name=model_name,
                           emotion_map=dataset.get_emotion_index_map())

print('Training model...')
history = model.fit_generator(train_gen.generate(target_dimensions, batch_size=batch_size),
                              test_gen.generate(
                                  target_dimensions, batch_size=batch_size),
                              epochs=epochs)

hist_df = pd.DataFrame(history.history)

# save history to json file
hist_json_file = "output/" + model_file_name + ".json"
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# Save model configuration
# model.export_model('output/transfer_learning_model.json','output/transfer_learning_weights.h5',"output/transfer_learning_emotion_map.json", emotion_map)

# Save model configuration (src.nueralnets.save_model)
model.save_model("output/" + model_file_name + ".h5", "output/" +
                 model_file_name + "_emotion_map.json", emotion_map)

print("model successfully saved")
