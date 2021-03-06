'''
Testing on video
Copyright 2019, Pieropan Edoardo and Pavan Gianluca, All rights reserved.
'''

import cv2
import numpy as np
from PIL import Image
import joblib as joblib
from sklearn.decomposition import PCA
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_xception

model = Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling='max')
clf_classes = joblib.load('../utils/svm_model_classes.sav')
#pca = joblib.load('../utils/pca.sav')

def create_tracker(n):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT']
    tracker_type = tracker_types[n]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker

def predict_signal(frame, bbox):
    im = Image.fromarray(frame)
    b, g, r = im.split()
    im = Image.merge("RGB", (r, g, b))
    cropped_image = im.crop ((int(bbox[0]),int(bbox[1]), int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))) #ritaglio con coordinate
    cropped_image = cropped_image.resize((width, height), Image.NEAREST) #resize delle immagini
    cropped_frame_array=np.asarray(cropped_image)

    # Estrazione feature
    img_data = np.expand_dims(cropped_frame_array, axis=0)
    img_data = preprocess_xception(img_data)
    img_feature = model.predict(img_data).flatten()
    #class_probabilities = clf.predict_proba(img_feature.reshape(1,-1))
    
    # img_feature = pca.transform(img_feature.reshape(1, -1))
    # predicted_class_labels = clf_classes.predict(img_feature)
    # predicted_class_str = str(predicted_class_labels).replace("[","")
    # predicted_class_str = predicted_class_str.replace("'","")
    # predicted_class_str = predicted_class_str.replace("]","")
    class_probabilities = clf_classes.predict_proba(img_feature.reshape(1,-1))
    print(class_probabilities)

    predicted_class_str = str(np.where(class_probabilities[0] == max(class_probabilities[0]))[0][0])

    if predicted_class_str != '5':
        clf = joblib.load('../utils/svm_model_'+predicted_class_str+'.sav')
        # predicted_labels = clf.predict(img_feature)
        # predicted_labels_str = str(predicted_labels).replace("[","")
        # predicted_labels_str = predicted_labels_str.replace("'","")
        # predicted_labels_str = predicted_labels_str.replace("]","")
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
    else:
        sample_image = Image.open("../samples/bg.jpg")
        sample_image = sample_image.convert("RGB")
        b, g, r = sample_image.split()
        sample_image = Image.merge("RGB", (r, g, b))
        frame = Image.fromarray(frame)
        frame.paste(sample_image , p2)
    return np.array(frame)

if __name__ == '__main__':
    START = 1 #starting frame
    width, height = 71, 71
    
    # Read video
    video = cv2.VideoCapture("../video/test_video.mp4")
 
    # Create new dictionary
    kv={}
    # variabili per eliminare tracker 
    c = np.zeros(24)
    c_count = 0
    c_del = 0

    counter = START-1
    trackers =[]
    images = []

    # Read bbox
    with open("../video/bbox.txt") as f:
        for line in f.readlines():
            splitted = line.split(" ", 1)
            splitted[1] = splitted[1].replace("\n","")
            splitted[1] = splitted[1].replace(" ","")
            kv[int(splitted[0])] = eval(splitted[1])

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    if not ok:
        print('Cannot read video file')
        sys.exit()
    

    video.set(cv2.CAP_PROP_POS_FRAMES, START)
    out = cv2.VideoWriter('../video/output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 29, (frame_width,frame_height))
    while (video.isOpened()):
        # Read a new frame
        ok, frame = video.read()
        counter+=1
        if not ok:
            break
        if( counter in kv.keys() ):
            tracker=create_tracker(4)
            tracker.init(frame, kv[counter])
            trackers.append(tracker)
            c[c_count]=5
            c_count+=1

        # Start timer
        timer = cv2.getTickCount()

        p1=(0,0)
        p2=(0,0)
        # Update trackers
        for t in trackers:
            ok, bbox = t.update(frame)
            if not ok:
                trackers.remove(t)
            else:           
                # Draw bounding box
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                frame = predict_signal(frame, bbox)

                cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
           
            try:
                i=c_del+trackers.index(t)
            except:
                print("An exception occurred")
                break

            if c[i] != 0:
                c[i]-=1
                if c[i] == 0:
                    trackers.remove(t)
                    c_del+=1
                    
        out.write(frame)
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    video.release()
    out.release()
    print('Video saved as video/output_video.mp4')
    cv2.destroyAllWindows()
    print('Task terminated')