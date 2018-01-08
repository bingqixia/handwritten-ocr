# Mute tensorflow debugging information on console
import os
import sys
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

app = Flask(__name__)

def load_map():
    mapping = {}
    f = open('mapping.txt', 'r')
    lines = f.read().splitlines()
    for line in lines:
        columns = line.strip().split(' ')
        #print columns[0]
        #print columns[1]
        mapping[int(columns[0])] = chr(int(columns[1]))
    #print mapping
    return mapping

def load_inputdata(input_path):
    # get fileNames
    data = np.zeros((28, 28))
    f = open(input_path, 'r')
    lines = f.read().splitlines()
    flag = 1
    f.close()
    words = np.zeros(100)
    c = 0
    for img_path in lines:
        #print img_path
        if img_path == 'end':
            print ''
            words[c] = 1
            continue
        # load imgs
        c = c+1
        img = imread(img_path, mode='L')
        #print img
        #ret, img = cv2.threshold(img, 0,255,cv2.THRESH_BINARY)
            #x = np.invert(x)
        #imsave('tmp/'+img_path, img)
        resized_img = resized(img)
        resized_img = np.invert(resized_img)
        #imsave('res/'+img_path, resized_img)
        if flag == 1:
            data = resized_img
            flag = 0
            continue
        data = np.vstack((data, resized_img))
    return data, words

def resized(rawimg):
    blur = cv2.medianBlur(rawimg,3)
    #imsave('blur.bmp', blur)
    fx = 28.0 / blur.shape[0]
    fy = 28.0 / blur.shape[1]
    fx = fy = min(fx, fy)
    img = imresize(blur, fx, 'cubic', mode='L')
    #cv2.imwrite("1.png", img)
    #imsave('resized1.bmp', img)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite('tmp1.bmp', img)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = (int)((28 - w) / 2)
    y = (int) ((28 - h) / 2)
    outimg[y:y+h, x:x+w] = img
    #outimg = cv2.GaussianBlur(outimg,(5,5),0)
    #cv2.imwrite("out.bmp", outimg)
    #ret, retimg = cv2.threshold(outimg, 220,255,cv2.THRESH_BINARY) 
    return outimg

def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def predict(imgPath):
    data, words = load_inputdata(imgPath)
    #data = np.invert(data)
    data = data.reshape(-1, 28, 28, 1)
    data = data.astype('float32')
    #print data.shape[0]
    data /= 255

    # Predict from model
    outs = model.predict(data)
    return outs,  data.shape[0], words

if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(description='A webapp for testing models generated from training.py on the EMNIST dataset')
    parser.add_argument('--bin', type=str, default='bin', help='Directory to the bin containing the model yaml and model h5 files')
    parser.add_argument('--img', type=str, default='', help='imgPath')
    args = parser.parse_args()

    # Overhead
    model = load_model(args.bin)
    #mapping = pickle.load(open('%s/mapping.p' % args.bin, 'rb'))
    mapping = load_map()

    imgPath = args.img
    outs, cont, words = predict(imgPath)
    #print outs
    for i in range(cont):
        out = outs[i]
        #print out
        if (words[i]):
            print ('')
        #print np.argmax(out, axis=0)
        #print ('prediction: %c' % chr(mapping[(int(np.argmax(out, axis=0)))]), ends='')
        key = int(np.argmax(out, axis=0))
        #print mapping[key]
        sys.stdout.write('%c' % mapping[key])
print ''
