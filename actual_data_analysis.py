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
import pandas as pd
import numpy as np
import datatable as dt
import time
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datatable as dt
import time
from sklearn.impute import SimpleImputer
import seaborn as sns

# %%

def plot_myReducedDim(X_reduced, y, no_of_labels, method):
    #restructure the data
    df = pd.DataFrame(X_reduced)
    df['categories'] = np.reshape(y, [y.shape[0],1])
    plt.figure(figsize=(12,8))
    
    if method == 'PCA':
        df['pca-one'] = X_reduced[:,0]
        df['pca-two'] = X_reduced[:,1]
        sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:,:],
        legend="full",
        alpha=0.3)
    elif method == 'tsne':
        df['tsne-one'] = X_reduced[:,0]
        df['tsne-two'] = X_reduced[:,1]
        sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:,:],
        legend="full",
        alpha=1)
    elif method == 'svd':
        df['svd-one'] = X_reduced[:,0]
        df['svd-two'] = X_reduced[:,1]
        sns.scatterplot(
        x="svd-one", y="svd-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:,:],
        legend="full",
        alpha=1)
    elif method == 'mds':
        df['mds-one'] = X_reduced[:,0]
        df['mds-two'] = X_reduced[:,1]
        sns.scatterplot(
        x="mds-one", y="mds-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:,:],
        legend="full",
        alpha=1)
    elif method == 'isomap':
        df['isomap-one'] = X_reduced[:,0]
        df['isomap-two'] = X_reduced[:,1]
        sns.scatterplot(
        x="isomap-one", y="isomap-two",
        hue="categories",
        palette=sns.color_palette("hls", no_of_labels),
        data=df.loc[:,:],
        legend="full",
        alpha=1)
    # you can write above function as just one statement instead of multiple if-elif

    plt.legend(loc='upper right', ncol=6) # horizontal legend upper right
    plt.show()


def code_to_numeric(labels):
    code_dict = {code:num for num, code in enumerate(np.unique(labels))}
    label_array = np.array(labels).reshape(labels.shape[0]).astype(str)
    numeric_list = [code_dict[label] for label in label_array]

    return numeric_list





# %%
# load the data 
x_train = dt.fread('train.csv/train.csv')
y_train = dt.fread('train.labels.csv')
x_test = dt.fread('test.csv/test.csv')
x_train = x_train.to_pandas()
y_train = y_train.to_pandas()
x_test = x_test.to_pandas()
# data normalization
# Imputation of the train data
imp = SimpleImputer(missing_values = -2, strategy = 'mean')
X_norm = imp.fit_transform(x_train)


y_numeric = code_to_numeric(y_train)




# %%
# see if you can make this function not shit


'''keys = y_train.Population.unique()
values = list(range(1,26))
dictionary = dict(zip(keys, values))

y_train_numeric = y_train["Population"].map(dictionary)
#y_train_numeric = y_train_numeric.values'''



# %%
print('The first 10 rows of the trainging dataset:')
x_train.head()



# %%
print('the training dataset has the size', x_train.shape)
print('the test dataset has the size', x_test.shape)

# %% [markdown]

# this cell should include the removal of some variables that are useless - see the code you put on slack! covariance matrices are too big so you can't do them tho :(

#remove constant features:

#from sklearn.feature_selection import VarianceThreshold

#https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/

# %%
# dimension reduction 
pca = PCA(n_components=0.9)
time_start = time.time()
X_reduced=pca.fit_transform(X_norm)

print(X_reduced.shape)
print('PCA time: {} seconds'.format(time.time()-time_start))


# %%

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train_numeric)
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.show()
# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score




k_array = list(range(2,20))

accList = []

for k in k_array:
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(clf_knn, X_reduced, y_numeric, cv=10)
    score=scores.mean()
    accList.append(score)

print(accList)

# %%

clf_knn.fit(X_reduced, y_numeric)



# %%

run a cheeky LDA