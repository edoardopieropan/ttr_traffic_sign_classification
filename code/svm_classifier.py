'''
Creazione, training e salvataggio di un classificatore SVM.
'''

import numpy as np
import joblib as joblib
from sklearn.svm import SVC

# Carico labels e data
x = np.load('../utils/features_full.npy')
y = np.load('../utils/labels_full.npy')

print('Training classifier...')
clf = SVC(kernel="rbf", gamma='auto', C=100, probability=True)
clf.fit(x, y)

joblib.dump(clf, '../utils/svm_model.sav')
print('Model saved...')

print('Task terminated')