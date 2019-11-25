'''
Print confusion matrix, accuracy, precision and recall values for the classifier trained in svm_classifier.py
Copyright 2019, Pieropan Edoardo and Pavan Gianluca, All rights reserved.
'''
import numpy as np
import pandas as pd
import seaborn as sn
import joblib as joblib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from tabulate import tabulate


clf = joblib.load('../utils/svm_model_classes.sav')
x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')
y_class = np.load('../utils/class_labels_full.npy')
#pca = joblib.load('../utils/pca.sav')

# PCA
# print('Using PCA...')
# x = pca.transform(x)

print("Calculating cross validation score...")
y_pred = cross_val_predict(clf, x, y_class, cv=2)
conf_mat = confusion_matrix(y_class, y_pred)
import seaborn as sn
import pandas as pd
df_cm = pd.DataFrame(conf_mat, index = range(max(y_class)+1),
                  columns = range(max(y_class)+1))
print('Confusion matrix:')
print(tabulate(conf_mat))

accuracy = metrics.accuracy_score(y_class, y_pred)
prec, rec, fscore, support = precision_recall_fscore_support(y_class, y_pred, average='macro')
print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall", rec],["Fscore", fscore]]))

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
#plt.show()

for i in range(max(y_class)):
    print("Calculating confusion matrix for class "+str(i)+"...")
    clf = joblib.load('../utils/svm_model_'+str(i)+'.sav')
    x_temp = x[y_class == i]
    y_temp = y[y_class == i]
    y_pred = cross_val_predict(clf, x_temp, y_temp, cv=5)
    conf_mat = confusion_matrix(y_temp, y_pred)
    print("Confusion matrix class "+str(i)+":")
    print(tabulate(conf_mat))
    accuracy = metrics.accuracy_score(y_temp, y_pred)
    prec, rec, fscore, support = precision_recall_fscore_support(y_pred, y_pred, average='macro')
    print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall",rec],["Fscore", fscore]]))
