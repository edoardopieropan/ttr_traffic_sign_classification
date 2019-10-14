'''
Creazione, training e salvataggio di un classificatore SVM.
'''

import numpy as np
import joblib as joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

x = np.load('../utils/images_full.npy', allow_pickle=True)

nobj = x.shape[0]
im_w = x.shape[1]
im_h = x.shape[2]
rgb  = x.shape[3]

#reshape
x = np.reshape(x, (nobj,im_w*im_h*rgb))

# Carico labels e data
#x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')

clf = SVC(kernel="rbf", gamma='auto', C=100, probability=False)
print('Training classifier...')
clf.fit(x, y)

# Stampa cross validation score (commenta per meno tempo)
print('Cross validation score: ')
score = cross_val_score(clf, x, y, cv=5)
print(score)

joblib.dump(clf, '../utils/svm_model.sav')
print('Model saved...')

print('Task terminated')