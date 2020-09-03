from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import pandas as pd
import json

file_name = "FER2013TrainValidation_Sorted_inception_v3_50_64"

with open('../examples/output/' + file_name + ".json", 'r') as f:
    history = json.load(f)
    df = pd.DataFrame(history)

accuracy_df = df[["acc", "val_acc"]]

accuracy_df.plot()
plt.title(file_name + " Accuracy")
plt.savefig("../examples/output/" +
            file_name + "_acc.png")
plt.close()

loss_df = df[["loss", "val_loss"]]

loss_df.plot()
plt.title(file_name + " Loss")
plt.savefig("../examples/output/" +
            file_name + "_loss.png")
plt.close()

print("successfully saved visualisations")
