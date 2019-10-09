'''
Salvataggio del dataset in label e immagini in formato npy
'''

from PIL import Image
import numpy as np
from tqdm import tqdm
import os

labels = []
images = []
width, height = 60, 60

foldernames= sorted(os.listdir("../dataset"))
for folder in tqdm(foldernames): # loop through all the files and folders
    current_label = folder
    imagenames = os.listdir("../dataset/"+folder)

    for img in imagenames:
        image = Image.open("../dataset/"+folder+"/"+img)
        image = image.resize((width, height), Image.NEAREST) #resize delle immagini
        images.append(np.asarray(image))
        labels.append(current_label)

np.save("../utils/labels_full.npy", labels) #esporto formato npy
print('labels saved...')

np.save("../utils/images_full.npy", images) #esporto formato npy        
print('images saved...')

print('Task terminated')