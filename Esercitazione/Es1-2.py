import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32")

x_train =(x_train / 255) - 0.5 
x_test = (x_test / 255) - 0.5 
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=1, synchronized = True ))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(Dropout(0.05))


model.add(Conv2D(16, (3, 3), activation='relu'))  
model.add(BatchNormalization(axis=1, synchronized = True))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(Dropout(0.05))


model.add(Conv2D(16, (3, 3), activation='relu'))  
model.add(BatchNormalization(axis=1, synchronized = True))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
model.add(Dropout(0.05))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.build()
model.summary()

callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=3)
#Si può mettere la callback per uscire prima


model.fit(
    X_train, to_categorical(Y_train), epochs=1000, 
    batch_size=256, shuffle=True, 
    validation_data=(X_val,to_categorical(Y_val)),
    callbacks=callback
    )

results = model.evaluate(
  x_test,
  to_categorical(y_test)
)

print(results)
now = datetime.datetime.now()

note=" "
with open('results.txt','a') as f:
  f.write(f"\n  {str(now.hour)}:{ str(now.minute)}  -  note: {note},  {results}")

#model.save_weights('modelEs1-2.h5')



# [0.2595813274383545, 0.9093999862670898] 32-16-dropout(0.3)
# [0.27315813302993774, 0.9072999954223633] 32-16-dropout(0.1)
# [0.2753869593143463, 0.9021999835968018] 512-16-dropout(0.4)