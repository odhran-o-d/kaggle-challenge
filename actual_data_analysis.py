# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import datatable as dt
import time
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import VarianceThreshold

# %% [markdown]
# ## load the data

# %%
x_train = dt.fread('train.csv/train.csv')
y_train = dt.fread('train.labels.csv')
x_test = dt.fread('test.csv/test.csv')


# %%
# data normalization
# Imputation of the train data
imp = SimpleImputer(missing_values = -2, strategy = 'mean')
X_norm = imp.fit_transform(x_train)



# %%
print('The first 10 rows of the trainging dataset:')
x_train.head()


# %%
y_train.head()




# %%
y_train_pd = pd.read_csv('train.labels.csv')


categories = y_train_pd.Population.unique()




# %%
print('the training dataset has the size', x_train.shape)
print('the test dataset has the size', x_test.shape)

# %% [markdown]
# ## data processing



# %%
# dimension reduction 

time_start = time.time()
pca = PCA(n_components=2)
X_reduced=pca.fit_transform(X_norm)


# %%
# plotting the PCA

from kaggle_functions import plot_myReducedDim


# %%
plot_myReducedDim(X_reduced, y_train, 


# %%
#remove constant features:

from sklearn.feature_selection import VarianceThreshold

https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/

