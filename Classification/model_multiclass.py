# data import
import kagglehub
# data processing
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# modelling
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.dummy import DummyClassifier
# visualisation
import matplotlib.pyplot as plt
import seaborn as sns


#%% data import
path = kagglehub.dataset_download("developerghost/intrusion-detection-logs-normal-bot-scan")
print("Path to dataset files: ", path)

file_path = os.path.join(path, "Network_logs.csv")
df = pd.read_csv(file_path)

#%% drop features that are not useful for the analysis
df = df.drop(columns=["Source_IP", "Destination_IP", "Intrusion"])

#%% treat categories variables
df_cat = pd.get_dummies(df, columns=["Request_Type", "Protocol", "User_Agent", "Status"], drop_first=True, dtype=int)
print(df_cat.head())
print(df_cat.dtypes)

#%% seperate indeoendant and dependant variables
X = df_cat.drop(columns=["Scan_Type"]).astype(float)
y = pd.factorize(df_cat["Scan_Type"])[0].astype(float)

print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% split data into training, validation and testing sets
# First split off test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Split remaining data into train and validation