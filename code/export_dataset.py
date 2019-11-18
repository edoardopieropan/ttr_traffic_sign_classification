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

howmuch_augment_class = { 
    '00':10, '01':10, '02':10, '03':10, '04':10,
    '05':10, '06':10, '07':10, '08':10, '09':10, 
    '10':10, '11':10, '12':10, '13':10, '14':10, 
    '15':0
    }

class_dictionary = { 
    '00':4, '01':4, '02':4, '03':0, '04':0,
    '05':0, '06':3, '07':3, '08':3, '09':1, 
    '10':1, '11':1, '12':2, '13':2, '14':2, 
    '15':5
    }

print('\nReading dataset...')
foldernames= sorted(os.listdir("../dataset"))
for folder in tqdm(foldernames): # loop through all the files and folders
    current_label = int(folder)
    imagenames = os.listdir("../dataset/"+folder)
    folder_label = class_dictionary[folder]
    for img in imagenames:
        image = Image.open("../dataset/"+folder+"/"+img)
        if(image.size[0]!=71 or image.size[1]!=71):
            image = image.resize((width, height), Image.NEAREST) #resize delle immagini
        if (np.shape(image)[2] != 3):
            image = image.convert('RGB')
        images.append(np.asarray(image))
        labels.append(current_label)
        class_labels.append(folder_label)
        
        augmented_images = augment_data(image, howmuch_augment_class[folder], True, True)
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