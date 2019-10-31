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


def code_to_numeric(labels):
    code_dict = {code:num for num, code in enumerate(np.unique(labels))}
    label_array = np.array(labels).reshape(labels.shape[0]).astype(str)
    numeric_list = [code_dict[label] for label in label_array]

    return numeric_list



from sklearn.metrics import fbeta_score, accuracy_score,confusion_matrix

def test_predict(clf, X_test, y_test): 
    results = {}
    predictions_test = clf.predict(X_test)
    accuracy=results['acc_test'] = accuracy_score(y_test, predictions_test)
    fbeta=results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
    
    print("accuracy for {} model is {}".format(clf.__class__.__name__,results['acc_test']))
    print("F beta for {} model is {}".format(clf.__class__.__name__,results['f_test']))
    print('------------------------------------------------------------------')
    
    return results


def save_to_csv(file_name,prediction_list):
    prediction_result = pd.DataFrame()
    prediction_result['ID'] = [i for i in range(1,1253)]
    prediction_result['Population'] = prediction_list
    prediction_result.to_csv(file_name,encoding='utf-8', index=False)
    return

# %%
# load the data 
x_train = dt.fread('train.csv/train.csv')
y_train = dt.fread('train.labels.csv')
x_test = dt.fread('test.csv/test.csv')
x_train = x_train.to_pandas()
y_train = y_train.to_pandas()
x_test = x_test.to_pandas()



# %%




# data normalization
# Imputation of the train data
imp = SimpleImputer(missing_values = -2, strategy = 'mean')
X_norm = imp.fit_transform(x_train)
X_test_norm = imp.fit_transform(x_test)
y_numeric = code_to_numeric(y_train)

# %%
# dimension reduction 
pca = PCA(n_components=1149)
time_start = time.time()
X_reduced=pca.fit_transform(X_norm)
X_test_reduced = pca.fit_transform(X_test_norm)




print(X_reduced.shape)
print('PCA time: {} seconds'.format(time.time()-time_start))




from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reduced,y_numeric, test_size = 0.3, random_state = 0)


print('the training dataset has the size', X_train1.shape)
print('the test dataset has the size', X_test1.shape)





# %% [markdown]

# this cell should include the removal of some variables that are useless - see the code you put on slack! covariance matrices are too big so you can't do them tho :(

#remove constant features:

#from sklearn.feature_selection import VarianceThreshold

#https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/

# %%



# %%

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_numeric)
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

from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB().fit(X_train1, y_train1)

predicted = nbclf.predict(X_test1)
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train1, y_train1)))
print('Accuracy of GaussianNB classifier on testing set: {:.2f}'.format(nbclf.score(X_test1, y_test1)))




# %%

preiction_values = test_predict(nbclf, X_test1, y_test1)


# %%
# support vectors 

from sklearn import svm
clf_svm = svm.SVC(kernel='sigmoid', gamma='scale', C=100) # Linear Kernel [linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable]
clf_svm.fit(X_train1, y_train1)

print('Accuracy of SVM on training set: {:.2f}'.format(clf_svm.score(X_train1, y_train1)))
print('Accuracy of SVM on test set: {:.2f}'.format(clf_svm.score(X_test1, y_test1)))

# %% 

predicted = clf_svm.predict(X_test_reduced)


# %%
# random forrest 

from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 26)
# Train the model on training data
rf.fit(X_train1, y_train1)
predicted = rf.predict(X_test1)

print('Accuracy of RF on training set: {:.2f}'.format(rf.score(X_train1, y_train1)))
print('Accuracy of RF on test set: {:.2f}'.format(rf.score(X_test1, y_test1)))




# %%

save_to_csv("attempt", predicted)

# %%
