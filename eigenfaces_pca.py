# -*- coding: utf-8 -*-
"""
Eigenfaces with PCA
Created on Wed Sep 30 20:22:02 2020

@author: ancarey
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

def fix_img(ima):
    im = np.asmatrix(np.rot90(np.reshape(np.atleast_2d(ima), (46, 56)), 1, (1,0)))
    return im

def mse(a, b):
    dif = np.subtract(a, b)
    square = np.square(dif)
    mse = square.mean()
    return mse

images = pd.read_csv('stacked_images.csv', sep=',', header=None)
p1I1 = pd.read_csv('person1image1.csv', sep=',', header=None)
   
# Get mean image
mean_image = np.sum(images, axis=1) / 10
mean_image = np.atleast_2d(mean_image)

# Normalize images
for j in range(9):
    images[j] = np.transpose(np.atleast_2d(images[j]) - mean_image)

''' Big matrix approach '''
# Calculate  Covariance Matrix
covariance = np.cov(images)
covariance = np.divide(covariance, 10.0)

# Get eigenvector and eigenvalues and sort
e_val, e_vec, = np.linalg.eig(covariance)
eig_pairs = [(e_val[index], e_vec[:,index]) for index in range(len(e_val))]
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(e_val))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(e_val))]
    
# # Plot
fig, axes = plt.subplots(4, 5, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    if i == 0:
        img = fix_img(mean_image)
    else:
        img = (eigvectors_sort[i-1]).astype(np.float64)
        img = fix_img(img)
    ax.imshow(img, cmap='gray')
plt.show()

    
'''Gram Matrix Approach'''
# Calculate Gram Covariance Matrix
gram_covariance = np.matrix(images.transpose()) * np.matrix(images)
gram_covariance = gram_covariance / 10

# Get eignevectors and eigenvalues, put in decreasing value
gram_e_val, gram_e_vec = np.linalg.eig(gram_covariance)
sort = gram_e_val.argsort()[::-1]
gram_e_val = gram_e_val[sort]
gram_e_vec = gram_e_vec[:, sort]

# Left multiply to get correct eigenvectors, find the norm of each, and then normalize
gram_e_vec = np.matmul(images, gram_e_vec)
norms = np.linalg.norm(gram_e_vec, axis=0)
gram_e_vec = gram_e_vec / norms

# Reshape all and plot
gram = [fix_img(mean_image), 
        fix_img(gram_e_vec[0]), 
        fix_img(gram_e_vec[1]),
        fix_img(gram_e_vec[2]),
        fix_img(gram_e_vec[3]),
        fix_img(gram_e_vec[4]),
        fix_img(gram_e_vec[5]),
        fix_img(gram_e_vec[6]),
        fix_img(gram_e_vec[7]),
        fix_img(gram_e_vec[8]),
        fix_img(gram_e_vec[9]),
        np.zeros((46,56))]
                                                                     

fig, axes = plt.subplots(2, 6, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(gram[i], cmap='gray')
fig.delaxes(axes[1,5])
plt.show()

'''Reconstruction of Person 1 Image 1''' 
# Compute the weights
origional = fix_img(p1I1.transpose())

img = p1I1.values.flatten() - mean_image
img = np.atleast_2d(img)
weights = gram_e_vec.transpose() * img
weights = weights.sum(axis=1)

# Do the projections
im = np.empty((2576, 10))

for i in range(9):   
    im[:, i] = np.atleast_2d(weights[i]*gram_e_vec[i])
    
proj0 = np.add(mean_image, im[:, 0])
proj1 = np.add(proj0, im[:, 1])
proj2 = np.add(proj1, im[:, 2])
proj3 = np.add(proj2, im[:, 3])
proj4 = np.add(proj3, im[:, 4])
proj5 = np.add(proj4, im[:, 5])
proj6 = np.add(proj5, im[:, 6])
proj7 = np.add(proj6, im[:, 7])
proj8 = np.add(proj7, im[:, 8])

projections = [proj0,
               proj1,
               proj2,
               proj3,
               proj4,
               proj5,
               proj6,
               proj7,
               proj8]

error = [mse(img, img),
         mse(img, proj0),
         mse(img, proj1),
         mse(img, proj2),
         mse(img, proj3),
         mse(img, proj4),
         mse(img, proj5),
         mse(img, proj6),
         mse(img, proj7),
         mse(img, proj8)]

    
projections = [origional,
               fix_img(proj0),
               fix_img(proj1),
               fix_img(proj2),
               fix_img(proj3),
               fix_img(proj4),
               fix_img(proj5),
               fix_img(proj6),
               fix_img(proj7),
               fix_img(proj8)]

fig, axes = plt.subplots(2, 5, figsize=(20,20))
for i, ax in enumerate(axes.flat):
    ax.imshow(projections[i], cmap='gray')
    ax.set_title(error[i])
plt.show()