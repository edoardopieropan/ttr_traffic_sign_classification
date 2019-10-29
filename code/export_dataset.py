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

to_flip = ['06', '09', '12', '13']

howmuch_augment_class = { 
    '00':3, '01':4, '02':3, '03':4, '04':3,
    '05':3, '06':5, '07':2, '08':2, '09':5, 
    '10':4, '11':2, '12':9, '13':9, '14':2, 
    '15':2, '16':5
    }
    
class_dictionary = { 
    '00':4, '01':4, '02':4, '03':0, '04':0,
    '05':0, '06':3, '07':3, '08':3, '09':3, 
    '10':3, '11':1, '12':1, '13':1, '14':2, 
    '15':2, '16':2
    }

print('\nReading dataset...')
foldernames= sorted(os.listdir("../dataset"))
for folder in tqdm(foldernames): # loop through all the files and folders
    current_label = int(folder)
    imagenames = os.listdir("../dataset/"+folder)
    folder_label = class_dictionary[folder]
    for img in imagenames:
        image = Image.open("../dataset/"+folder+"/"+img)
        image = image.resize((width, height), Image.NEAREST) #resize delle immagini
        images.append(np.asarray(image))
        labels.append(current_label)
        class_labels.append(folder_label)
        augmented_images = augment_data(image, howmuch_augment_class[folder], True, True, (folder in to_flip))
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