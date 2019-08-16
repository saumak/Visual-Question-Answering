import warnings
from keras.models import model_from_json
from keras import backend

backend.set_image_data_format('channels_first')
warnings.filterwarnings("ignore")

def ini_lstm_mlp_model(lstm_mlp_model, lstm_mlp_weight):
    model = model_from_json(open(lstm_mlp_model).read())
    model.load_weights(lstm_mlp_weight)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
