import numpy as np
import pandas as pd
import seaborn as sn
import pandas as pd
import seaborn as sn
import os
import joblib as joblib
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tabulate import tabulate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

x = np.load('../utils/features_full.npy', allow_pickle=True)
y = np.load('../utils/labels_full.npy')
y_class = np.load('../utils/class_labels_full.npy')
cntr = 0
titles = ['SVC', 'KNN']
#clfs = [SVC(kernel="rbf", C=100, gamma='auto', probability=False), KNeighborsClassifier(n_neighbors=30) ]
clfs = [SVC(kernel="rbf", C=100, gamma='auto', probability=False)]
plt.figure(figsize = (10,7))

if not os.path.exists('../confusion_matrix/'):
    os.makedirs('../confusion_matrix/')

#SENZA PCA
for c in clfs:
    clf = c
    print('Training classifier...')
    clf.fit(x, y_class)

    print("Calculating cross validation score...")
    y_pred = cross_val_predict(clf, x, y_class, cv=5)

    
    conf_mat = confusion_matrix(y_class, y_pred)


    df_cm = pd.DataFrame(conf_mat, index = range(max(y_class)+1),
                    columns = range(max(y_class)+1))
    print('Confusion matrix:')
    print(tabulate(conf_mat))
    accuracy = metrics.accuracy_score(y_class, y_pred)
    prec, rec, fscore, support = precision_recall_fscore_support(y_class, y_pred, average='macro')
    print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall", rec],["Fscore", fscore]]))

    plt.clf()
    plt.title(titles[clfs.index(c)]+" classifier without PCA - superclasses")
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    #plt.show()
    plt.savefig('../confusion_matrix/'+str(cntr))
    cntr+=1

    for i in range(max(y_class)):
        x_temp = x[y_class == i]
        y_temp = y[y_class == i]
        clf = c
        print('Training classifier...')
        clf.fit(x_temp, y_temp)
        plt.title(titles[clfs.index(c)]+" classifier without PCA - class " + str(i))
        print("Calculating confusion matrix for class "+str(i)+"...")
        y_pred = cross_val_predict(clf, x_temp, y_temp, cv=5)


        conf_mat = confusion_matrix(y_temp, y_pred)

        df_cm = pd.DataFrame(conf_mat, index = range(max(y_temp)-min(y_temp)+1),
                    columns = range(max(y_temp)-min(y_temp)+1))

        print("Confusion matrix class "+str(i)+":")
        print(tabulate(conf_mat))
        accuracy = metrics.accuracy_score(y_temp, y_pred)
        prec, rec, fscore, support = precision_recall_fscore_support(y_temp, y_pred, average='macro')
        print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall", rec],["Fscore", fscore]]))

        plt.clf()
        plt.title(titles[clfs.index(c)]+" classifier without PCA - class " + str(i))
        sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        #plt.show()
        plt.savefig('../confusion_matrix/'+str(cntr))
        cntr+=1

# PCA
print('Using PCA...')
pca = PCA(n_components=100)
x = pca.fit_transform(x)

for c in clfs:
    clf = c
    print('Training classifier...')
    clf.fit(x, y_class)

    print("Calculating cross validation score...")
    y_pred = cross_val_predict(clf, x, y_class, cv=5)
    accuracy = metrics.accuracy_score(y_class, y_pred)
    print("Accuracy: " + str(accuracy))
    conf_mat = confusion_matrix(y_class, y_pred)
    df_cm = pd.DataFrame(conf_mat, index = range(max(y_class)+1),
                    columns = range(max(y_class)+1))
    print('Confusion matrix:')
    print(tabulate(conf_mat))
    accuracy = metrics.accuracy_score(y_class, y_pred)
    prec, rec, fscore, support = precision_recall_fscore_support(y_class, y_pred, average='macro')
    print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall", rec],["Fscore", fscore]]))

    plt.clf()
    plt.title(titles[clfs.index(c)]+" classifier PCA - superclasses")
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    #plt.show()
    plt.savefig('../confusion_matrix/'+str(cntr))
    cntr+=1

    for i in range(max(y_class)):
        x_temp = x[y_class == i]
        y_temp = y[y_class == i]
        clf = c
        print('Training classifier...')
        clf.fit(x_temp, y_temp)
        plt.title(titles[clfs.index(c)]+" classifier PCA - class " + str(i))
        print("Calculating confusion matrix for class "+str(i)+"...")
        y_pred = cross_val_predict(clf, x_temp, y_temp, cv=5)
        accuracy = metrics.accuracy_score(y_temp, y_pred)
        print("Accuracy: " + str(accuracy))
        conf_mat = confusion_matrix(y_temp, y_pred)
        df_cm = pd.DataFrame(conf_mat, index = range(max(y_temp)-min(y_temp)+1),
                    columns = range(max(y_temp)-min(y_temp)+1))
        print("Confusion matrix class "+str(i)+":")
        print(tabulate(conf_mat))
        accuracy = metrics.accuracy_score(y_temp, y_pred)
        prec, rec, fscore, support = precision_recall_fscore_support(y_temp, y_pred, average='macro')
        print(tabulate([["Accuracy", accuracy],["Precision", prec],["Recall", rec],["Fscore", fscore]]))

        plt.clf()
        plt.title(titles[clfs.index(c)]+" classifier PCA - class " + str(i))
        sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
        #plt.show()
        plt.savefig('../confusion_matrix/'+str(cntr))
        cntr+=1