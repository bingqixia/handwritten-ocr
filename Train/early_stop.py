import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import cv2
import base64
import pickle

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True