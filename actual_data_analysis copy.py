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
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
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
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split



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
imp = SimpleImputer(missing_values = -2, strategy = 'most_frequent')
X_norm = imp.fit_transform(x_train)
X_test_norm = imp.transform(x_test)



#le = LabelEncoder()
#y_numeric = y_train.apply(le.fit_transform) 




# %%
# non-linear dimension reduction 
kpca = KernelPCA(kernel="rbf", n_components=900)
time_start = time.time()

X_reduced=kpca.fit_transform(X_norm)
X_test_reduced = kpca.transform(X_test_norm)

print(X_reduced.shape)
print('PCA time: {} seconds'.format(time.time()-time_start))

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reduced,y_numeric, test_size = 0.3, random_state = 0)


print('the training dataset has the size', X_train1.shape)
print('the test dataset has the size', X_test1.shape)



# %%
#XGBoost attempt

from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=200, learning_rate=0.25)
xgb_clf.fit(X_train1, y_train1)

score = xgb_clf.score(X_test1, y_test1)
print(score)



# %%

prediction = xgb_clf.predict(X_test_reduced)

# %%

code_dict = {code:num for num, code in enumerate(np.unique(y_train))}


def get_keys_by_value(dict, value):
    list_of_keys = []
    list_of_items = dict.items()
    list_of_items = dict.items()
    for item in list_of_items:
        if item[1] == value:
            list_of_keys.append(item[0])
    return list_of_keys


def numeric_to_code(predictions, code_dict):
    code_list = []
    for id, pred in enumerate(predictions):
        code_string = get_keys_by_value(code_dict, pred)
        code_list.append(code_string[0])

    return code_list


results = numeric_to_code(prediction, code_dict)



# %%
#gradient boosting classifiers 
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train1, y_train1)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train1, y_train1)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test1, y_test1)))



# %%
# linear dimension reduction 
pca = PCA(n_components=1149)
time_start = time.time()
X_reduced=pca.fit_transform(X_norm)
X_test_reduced = pca.transform(X_test_norm)


X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reduced,y_numeric, test_size = 0.3, random_state = 0)


print('the training dataset has the size', X_train1.shape)
print('the test dataset has the size', X_test1.shape)


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

save_to_csv("attempt_2", results)

# %%
