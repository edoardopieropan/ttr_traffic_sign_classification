'''
Salvataggio del dataset in label e immagini in formato npy
'''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from data_augmentation import augment_data

labels = []
class_labels = []
images = []
width, height = 71, 71

class_dictionary = { '0':0, '4':0, '2':1, '5':1, '3':2, '8':2, '9':2, '6':3, '7':3, '1':4 }

print('\nReading dataset...')
foldernames= sorted(os.listdir("../dataset_t"))
for folder in tqdm(foldernames): # loop through all the files and folders
    current_label = folder
    imagenames = os.listdir("../dataset_t/"+folder)
    folder_label = class_dictionary[folder]
    for img in imagenames:
        image = Image.open("../dataset_t/"+folder+"/"+img)
        image = image.resize((width, height), Image.NEAREST) #resize delle immagini
        images.append(np.asarray(image))
        labels.append(current_label)
        class_labels.append(folder_label)
        augmented_images = augment_data(image, 2, True, True, True)
        images.extend(augmented_images)
        for ai in range(len(augmented_images)):
            labels.append(current_label)
            class_labels.append(folder_label)

if not os.path.exists('../utils/'):
    os.makedirs('../utils/')
    
#esporto formato npy
np.save("../utils/labels_full.npy", labels) 
np.save("../utils/class_labels_full.npy", class_labels)
print('labels saved...')

np.save("../utils/images_full.npy", images) #esporto formato npy        
print('images saved...')

print('Task terminated')