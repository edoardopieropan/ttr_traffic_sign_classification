'''
Creazione, training e salvataggio di un classificatore SVM.
'''

import numpy as np
import joblib as joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Carico labels e data
x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')

# PCA
print('Using PCA...')
pca = PCA(n_components=800)
x = pca.fit_transform(x)
joblib.dump(pca, '../utils/pca.sav')
print('PCA instance saved...')

clf = SVC(kernel="rbf", gamma='auto', C=100, probability=False)
print('Training classifier...')
clf.fit(x, y)

joblib.dump(clf, '../utils/svm_model.sav')
print('Model saved...')

# Stampa cross validation score (commenta per meno tempo)
print('Cross validation score: ')
score = cross_val_score(clf, x, y, cv=5)
print(score)

print('Task terminated')