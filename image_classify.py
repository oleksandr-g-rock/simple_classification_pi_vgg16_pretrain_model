#import modules
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#load imgenet vgg16 model
model = VGG16(weights='imagenet')

#load image and change size to 224*224
img_path = 'demo.jpg'
img = image.load_img(img_path, target_size=(224, 224))

#convert image to array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#predict class for image
preds = model.predict(x)
print('Result:', decode_predictions(preds, top=1)[0])
