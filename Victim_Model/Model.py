import os
import kagglehub
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH_CSV = "../Dataset/datasets/nih-chest-xrays/sample/versions/4/sample_labels.csv"
DATASET_PATH_CSV = "../Dataset/datasets/nih-chest-xrays/sample/versions/4/sample/images"
KAGGLE_DATASET_PATH = "nih-chest-xrays/sample"
os.environ['KAGGLEHUB_CACHE'] = KAGGLE_DATASET_PATH




# Download the Dataset
print("Downloading sample dataset...")
path = kagglehub.dataset_download(KAGGLE_DATASET_PATH)
print("Path to dataset files:", path)


# load the csv data
df = pd.read_csv(DATASET_PATH_CSV)

# Reformat the output labels from label1|label2|label3.. to ["label1", "label2", "label3", ...]
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
