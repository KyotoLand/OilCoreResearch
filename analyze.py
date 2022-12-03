import glob
import re

import numpy as np
import os
import matplotlib.pyplot as plt
import shutil


import pandas as pd
# for loading/processing the images
import keras.preprocessing.image as keras_im
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors


def extract_features(file, model):
    # load the image as a 224x224 array
    img = keras_im.image_utils.load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


path = r"C:\Users\ivmi0322\PycharmProjects\OilCoreResearch\resources\wavelets"
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
scalograms = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
            # adds only the image files to the scalograms list
            scalograms.append(file.name)

# load model
model = VGG16()
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

data = {}
p = r"C:\Users\ivmi0322\PycharmProjects\OilCoreResearch\resources\features\scalogram_features.pkl"

# loop through each image in the dataset
for scalogram in scalograms:
    # try to extract the groups and update the dictionary
    feat = extract_features(scalogram, model)
    data[scalogram] = feat

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the groups
feat = np.array(list(data.values()))

# reshape so that there are 85 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=85)
pca.fit(feat)
x = pca.transform(feat)
print(x.shape)
neighbors = NearestNeighbors(n_neighbors=3)
neighbors_fit = neighbors.fit(x)
distances, indices = neighbors_fit.kneighbors(x)
distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.plot(distances)
plt.savefig("C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\neighbours.png")
plt.clf()
# cluster feature vectors
kmeans = KMeans(n_clusters=12, random_state=22)
# kmeans = DBSCAN(eps=8.5, min_samples=2)
label = kmeans.fit_predict(x)

# Getting unique labels

u_labels = np.unique(label)

# plotting the results:
pca = PCA(2)

# Transform the data
df = pca.fit_transform(feat)
for i in u_labels:
    plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
plt.legend()
plt.savefig("C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\clusters.png")
plt.clf()

cutted_filenames = []
for filename in filenames:
    temp = filename[4:-4].strip().upper()
    cutted_filenames.append(temp)
cutted_filenames = set(cutted_filenames)
print(cutted_filenames)

which_group = {}
for filename in cutted_filenames:
    which_group.update({filename: []})
# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

files = glob.glob('C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\groups\\wavelets\\*')
for f in files:
    os.remove(f)
files = glob.glob('C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\groups\\grafs\\*')
for f in files:
    os.remove(f)
for group in groups.keys():
    directory = str(group)

    # Parent Directories
    parent_wavelets_dir = "C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\groups\\wavelets\\"
    parent_grafs_dir = "C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\groups\\grafs\\"

    # Path
    path_wavelets = os.path.join(parent_wavelets_dir, directory)
    path_grafs = os.path.join(parent_grafs_dir, directory)

    # Create the directory
    os.makedirs(path_wavelets)
    os.makedirs(path_grafs)
    for image in groups[group]:
        shutil.copy2('C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\wavelets\\' + image, path_wavelets)
        shutil.copy2('C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resources\\grafs\\' + image, path_grafs)
        temp = image[4:-4].strip().upper()
        temp_ar = which_group.get(temp)
        temp_ar.append(group)
        which_group.update({temp: temp_ar})

print(which_group)
hist_platforms = {}
for plat, which_groups in which_group.items():
    temp_dict = {}
    for i in range(0, len(groups)):
        temp_dict.update({i: 0})
        for group in which_groups:
            if group == i:
                temp_counter = temp_dict.get(i)
                temp_counter += 1
                temp_dict.update({i: temp_counter})
    hist_platforms.update({plat: temp_dict})
print(hist_platforms)
hist_platforms_features = {}
for plat, hist in hist_platforms.items():
    hist_platforms_features.update({plat: list(hist.values())})

print(hist_platforms_features)

hist_platforms_features_normalized = {}

for plat, features in hist_platforms_features.items():
    norm = LA.norm(features)
    if norm == 0:
        hist_platforms_features_normalized.update({plat: list(features)})
    else:
        hist_platforms_features_normalized.update({plat: list(features/LA.norm(features))})

print(hist_platforms_features_normalized)
# plt.legend()
# plt.savefig("C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\hist.png")
# plt.clf()

sse = []
list_k = list(range(2, 30))
max_norm = 0
for ar in x:
    temp_norm = LA.norm(ar)
    if temp_norm > max_norm:
        max_norm = temp_norm

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    sse.append(np.sqrt(km.inertia_) / max_norm)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.savefig("C:\\Users\\ivmi0322\\PycharmProjects\\OilCoreResearch\\resourcesKMeans.png")
