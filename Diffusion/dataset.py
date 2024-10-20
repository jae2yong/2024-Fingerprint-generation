from datasets import load_dataset
import matplotlib.pyplot as plt
from config import *

config.dataset_name = "Dataset/Fingerprint_Optical/"
dataset = load_dataset(config.dataset_name, split="train")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()
plt.show()