import warnings
import cv2, numpy as np
from keras.optimizers import SGD
from keras import backend
from vgg import vgg_16

backend.set_image_data_format('channels_first')
warnings.filterwarnings("ignore")

def ini_vgg_model(cnn_weight):
    vgg_mod = vgg_16(cnn_weight)
    vgg_mod.layers.pop()
    vgg_mod.layers.pop()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    vgg_mod.compile(optimizer=sgd, loss='categorical_crossentropy')
    return vgg_mod

def image_feature(image, vgg_mod):
    img_feat, img = np.zeros((1, 4096)), cv2.resize(cv2.imread(image), (224, 224))
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0) 
    img_feat[0,:] = vgg_mod.predict(img)[0]
    return img_feat
