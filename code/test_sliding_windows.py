import cv2
import numpy as np
from PIL import Image
import joblib as joblib
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_xception
from numpy.lib.stride_tricks import as_strided as ast
from itertools import product
from helpers import pyramid
from helpers import sliding_window
import argparse
import time
import math

model = Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')
clf_classes = joblib.load('../utils/svm_model_classes.sav')
# pca = joblib.load('../utils/pca.sav')
width, height = 71,71

def predict_signal(signal, p2, frame):
    im = Image.fromarray(signal)
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    cropped_image = im.resize((width, height), Image.NEAREST) #resize delle immagini
    cropped_frame_array = np.asarray(cropped_image)

    # Estrazione feature
    img_data = np.expand_dims(cropped_frame_array, axis=0)
    img_data = preprocess_xception(img_data)
    img_feature = model.predict(img_data).flatten()
    class_probabilities = clf_classes.predict_proba(img_feature.reshape(1,-1))
    print(class_probabilities)

    predicted_class_str = str(np.where(class_probabilities[0] == max(class_probabilities[0]))[0][0])

    if(predicted_class_str != '5'):
        clf = joblib.load('../utils/svm_model_'+predicted_class_str+'.sav')
        probabilities = clf.predict_proba(img_feature.reshape(1,-1))
        print(probabilities)
        predicted_labels_str = str(np.where(probabilities[0] == max(probabilities[0]))[0][0])

        sample_image = Image.open("../samples/"+predicted_class_str+"/"+predicted_labels_str+".jpg")
        sample_image = sample_image.convert("RGB")
        b, g, r = sample_image.split()
        sample_image = Image.merge("RGB", (r, g, b))
        basewidth = 50
        wpercent = (basewidth/float(sample_image.size[0]))
        hsize = int((float(sample_image.size[1])*float(wpercent)))
        sample_image = sample_image.resize((basewidth, hsize), Image.ANTIALIAS) #resize delle immagini
        frame = Image.fromarray(frame)
        frame.paste(sample_image , p2)
        f = True
    else:
        f = False
        sample_image = Image.open("../samples/bg.jpg")
        sample_image = sample_image.convert("RGB")
        b, g, r = sample_image.split()
        sample_image = Image.merge("RGB", (r, g, b))
        frame = Image.fromarray(frame)
        frame.paste(sample_image , p2)

    return np.array(frame), f

counter = 0

if not os.path.exists('../sliding_windows_results/'):
    os.makedirs('../sliding_windows_results/')

imagenames = sorted(os.listdir("../tests/"))

for image in imagenames:
    image = cv2.imread("../tests/"+image)
    size = int(math.floor(image.shape[1]/3))
    (winW, winH) = (size, size)
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=size, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if (window.shape[0] != winH) or (window.shape[1] != winW):
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)

        frame, f = predict_signal(np.asarray(window), (x + winW - 50, y + winH - 50), clone) 
        if f == True:   
            cv2.imshow("Window", frame)
            cv2.imwrite('../sliding_windows_results/res'+str(counter)+'.png',frame)
            cv2.waitKey(1)
            counter +=1
            time.sleep(2)
        else:
            cv2.imshow("Window", clone)
            cv2.waitKey(1)

print("Task terminated...")