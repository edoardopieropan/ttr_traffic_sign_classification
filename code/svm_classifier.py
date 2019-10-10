'''
Creazione, training e salvataggio di un classificatore SVM.
'''

import numpy as np
import joblib as joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Carico labels e data
x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')

print('Training classifier...')
clf = SVC(kernel="rbf", gamma='auto', C=100, probability=False)
clf.fit(x, y)

# Stampa cross validation score (commenta per ameno tempo)
print('Cross validation score: ')
score = cross_val_score(clf, x, y, cv=5)
print(score)

joblib.dump(clf, '../utils/svm_model.sav')
print('Model saved...')

print('Task terminated')