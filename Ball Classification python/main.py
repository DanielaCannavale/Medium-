import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

data_train_path="train"
data_test_path="test"

batch_size=4
img_size=224
optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
loss="categorical_crossentropy"
epochs=10
path_onnx="Daniela\PycharmProjects\Ball\Ball-Classification"

image_generator_test = ImageDataGenerator(rescale=1./255,)
data_train = image_generator_test.flow_from_directory(batch_size=batch_size,
                                                  directory=data_train_path,
                                                  shuffle=True,
                                                  target_size=(img_size, img_size),
                                                  class_mode='categorical'

                                                  )
data_test = image_generator_test.flow_from_directory(batch_size=batch_size,
                                                  directory=data_test_path,
                                                  shuffle=True,
                                                  target_size=(img_size, img_size),
                                                  class_mode='categorical'

                                                  )
print("Train Data",data_train.class_indices)
print("Test Data",data_test.class_indices)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout

inputs=Input(shape=(224,224,3),name="input")

x = Conv2D(16,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

x=tf.keras.layers.GlobalMaxPool2D()(x)

x=tf.keras.layers.Dense(128,activation="relu")(x)
x=tf.keras.layers.Dense(len(data_train.class_indices),activation="softmax",name="output")(x)

model=tf.keras.Model(inputs=inputs,outputs=x)
model.summary()

model.compile(optimizer =optimizer,loss = loss,  metrics = ["accuracy"])

callbacks=[]
callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss',factor = 0.1,patience = 2,min=1e-6,verbose=1))
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="loss",patience=4,restore_best_weights=True))
model.fit(data_train,validation_data=data_test,  epochs = 10,callbacks=callbacks)

from PIL import Image
import numpy as np

img = Image.open("test/football/football22.png").convert('RGB').resize((img_size,img_size))
img=np.asarray(img)
predicted=model.predict(np.expand_dims(img,axis=0))
predicted
list(data_train.class_indices.keys())[np.argmax(predicted)]
print(predicted)
import os
save_model_path="save_dmodel"
if not os.path.isdir(save_model_path):
      os.makedirs(save_model_path)

model.save(save_model_path)


#import numpy as np
#import onnxruntime as ort

#img = Image.open("download.jpg").convert('RGB').resize((224,224))
#img=np.expand_dims(np.asarray(img, dtype="float32"),axis=0)
#img=img/255

#sess_ort = ort.InferenceSession("onnx_path",providers=ort.get_available_providers())

#outputs = sess_ort.run(None, {sess_ort.get_inputs()[0].name: img})
#print(outputs)
