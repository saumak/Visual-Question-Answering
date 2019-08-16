
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D
import numpy as np



def pop(model):
    '''Ref: https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.ib_nodes = []
            model.ob_nodes = []
        else:
            model.layers[-1].ob_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

def load_model_legacy(model, weight_path):
    ''' Training the model with legacy keras(old version)'''

    import h5py
    f = h5py.File(weight_path, mode='r')
    flattenLayer = model.layers

    nb_layers = f.attrs['nb_layers']

    for k in range(nb_layers):
        g = f['layer_{}'.format(k)]
        wT = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if not wT: continue
        if len(wT[0].shape) >2: 
            # swap conv axes
            wT[0] = np.swapaxes(wT[0],0,3)
            wT[0] = np.swapaxes(wT[0],0,2)
            wT[0] = np.swapaxes(wT[0],1,2)
        flattenLayer[k].set_weights(wT)

    f.close()

def vgg_16(wT_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,( 3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if wT_path:
        load_model_legacy(model, wT_path)

    #Removing the last two layers 
    model = pop(model)
    model = pop(model)
        

    return model

