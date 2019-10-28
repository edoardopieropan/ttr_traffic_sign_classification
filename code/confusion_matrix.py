#TODO INSERIRE DATI TEST E TRAINING

import numpy as np
import pandas as pd
import seaborn as sn
import joblib as joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

clf = joblib.load('../utils/svm_model_classes.sav')
x = np.load('../utils/features_full.npy', allow_pickle=True)
y_class = np.load('../utils/class_labels_full.npy')

# PCA
print('Using PCA...')
pca = PCA(n_components=1000)
x = pca.fit_transform(x)
joblib.dump(pca, '../utils/pca.sav')
print('PCA instance saved...')

# Stampa cross validation score (commenta per meno tempo)
print("Calculating cross validation score...")
y_pred = cross_val_predict(clf, x, y_class)
conf_mat = confusion_matrix(y_class, y_pred)
print('Cross validation score: ')
import seaborn as sn
import pandas as pd
df_cm = pd.DataFrame(conf_mat, index = range(max(y_class)+1),
                  columns = range(max(y_class)+1))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()