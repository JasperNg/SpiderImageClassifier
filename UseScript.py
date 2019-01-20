from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('Spider_model2.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
img = image.load_img("D:\Desktop\Train Data\Ticks\A648x364_Lyme_Disease.jpg", target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
if classes == 0:
    print("Spider")
else:
    print("Ticks")




