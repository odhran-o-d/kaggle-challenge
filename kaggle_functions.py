# 2D plotting PCA 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
no_of_labels=2
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