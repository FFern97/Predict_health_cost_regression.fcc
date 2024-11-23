########################  1  ##########################

# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import sklearn
from sklearn.model_selection import train_test_split




########################  2  ##########################


# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()


########################  3  ##########################


# Categorical data to numerical.

df = pd.read_csv('insurance.csv')
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])


########################  4  ##########################

# Split data 80% train / 20% test.

train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)

print(train_dataset.shape)
print(test_dataset.shape)


########################  5  ##########################

# Pop off Expenses column.

train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

print(train_labels.shape)
print(test_labels.shape)

