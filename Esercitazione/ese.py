import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import datetime

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32")

x_train =(x_train / 255) - 0.5 
x_test = (x_test / 255) - 0.5 
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)



#x_train = x_train.reshape((-1, 784))
#x_test = x_test.reshape((-1, 784))

model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.summary()

callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=3)
#Si può mettere la callback per uscire prima

model.fit(
    X_train, to_categorical(Y_train), epochs=100, 
    batch_size=64, shuffle=True, 
    validation_data=(X_val,to_categorical(Y_val)),
    callbacks=callback
    )

results = model.evaluate(
  x_test,
  to_categorical(y_test)
)

print(results)
model.save_weights('modelEs1-1.h5')

now = datetime.datetime.now()

note=" "
with open('resultsEseRandom.txt','a') as f:
  f.write(f"\n  {str(now.hour)}:{ str(now.minute)}  -  note: {note},  {results}")