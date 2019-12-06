'''
Feature extraction with Keras Xception application
Copyright 2019, Pieropan Edoardo and Pavan Gianluca, All rights reserved.
'''

import os
import numpy as np
from tqdm import tqdm
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_xception

x_train = np.load('../utils/images_full.npy', allow_pickle=True)

x_train_features = []
model = Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')

print("\nExtracting features...")
for i in tqdm(x_train):
    img_data = i
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_xception(img_data)

    # Estrazione feature
    x_train_features.append(model.predict(img_data).flatten())

np.save("../utils/features_full.npy", x_train_features) #esporto formato npy
print('Features saved...')

print('Task terminated')