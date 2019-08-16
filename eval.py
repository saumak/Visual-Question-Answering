import warnings
import spacy, numpy as np
from img_feat import ini_vgg_model, image_feature
from text_feat import text_feature
from mlp_model import ini_lstm_mlp_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")

image = 'COCO_val2014_000000000294.jpg'
question = u"Who is in the picture?"

spacy_lg = spacy.load('en_core_web_lg')

lstm_mlp_model = 'features/VQA_MODEL.json'
lstm_mlp_weight = 'features/VQA_MODEL_WEIGHTS.hdf5'
encoder_weights = 'features/FULL_labelencoder_trainval.pkl'
vgg_weights = 'features/vgg16_weights.h5'

img_model = ini_vgg_model(vgg_weights)
mlp_model = ini_lstm_mlp_model(lstm_mlp_model, lstm_mlp_weight)

img_feat = image_feature(image, img_model)
qus_feat = text_feature(spacy_lg, question)

ans_prob = mlp_model.predict([qus_feat, img_feat])

img=mpimg.imread(image)
imgplot = plt.imshow(img)
plt.show()

print(question)

encoder = joblib.load(encoder_weights)
for label in reversed(np.argsort(ans_prob)[0,-5:]): print(str(round(ans_prob[0,label]*100,2)).zfill(5), "% ", encoder.inverse_transform(label))
