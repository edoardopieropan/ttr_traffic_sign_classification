# DATA AUGMENTATION
import cv2
import numpy
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def augment_data(img, quantity, zoom, brightness, flip):
    return_array = []

    if zoom:
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range=[0.5,1.5])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(quantity):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # add image to array
            return_array.append(numpy.asarray(image))

    if brightness:
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range=[0.5,1.5])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(quantity):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # add image to array
            return_array.append(numpy.asarray(image))

    if flip:
        open_cv_image = numpy.array(img) 
        img = open_cv_image[:, :, ::-1].copy() 
        return_array.append(numpy.asarray(cv2.flip(img, 1)))

    return return_array