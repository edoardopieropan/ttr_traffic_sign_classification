#!/bin/sh
# AUTORUN SCRIPT
# TRAFFIC SIGN CLASSIFICATION
# Copyright 2019, Pieropan Edoardo and Pavan Gianluca, All rights reserved.

cd code/
python3 export_dataset.py
python3 keras_features.py
python3 svm_classifier.py