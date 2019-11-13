'''
Creazione, training e salvataggio di un classificatore SVM.
'''

import numpy as np
from tqdm import tqdm
import joblib as joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Carico labels e data
x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')
y_class = np.load('../utils/class_labels_full.npy')

# PCA
# print('Using PCA...')
# pca = PCA(n_components=1000)
# x = pca.fit_transform(x)
# joblib.dump(pca, '../utils/pca.sav')
# print('PCA instance saved...')

clf = SVC(kernel="rbf", C=100, gamma='auto', probability=False)
print('Training classifier...')
clf.fit(x, y_class)

joblib.dump(clf, '../utils/svm_model_classes.sav')
print('Model saved...')

print('Training sub-classifiers...')
for i in tqdm(range(max(y_class))):
    clf = SVC(kernel="rbf", C=100, gamma='auto', probability=False)
    x_temp = x[y_class == i]
    y_temp = y[y_class == i]
    clf.fit(x_temp, y_temp)
    joblib.dump(clf, '../utils/svm_model_'+str(i)+'.sav')

print('Models saved...')
print('Task terminated')