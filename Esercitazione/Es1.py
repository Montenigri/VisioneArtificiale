import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32")

x_train =(x_train / 255) - 0.5 
x_test = (x_test / 255) - 0.5 

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)



#x_train = x_train.reshape((-1, 784))
#x_test = x_test.reshape((-1, 784))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
#model.add(Conv2D(8, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.summary()

#callback = keras.callbacks.EarlyStopping(monitor= "val_loss", patience=3)
#Si può mettere la callback per uscire prima

model.fit(
    X_train, to_categorical(Y_train), epochs=10, 
    batch_size=128, shuffle=True, 
    validation_data=(X_val,to_categorical(Y_val))
    )

results = model.evaluate(
  x_test,
  to_categorical(y_test)
)

print(results)
model.save_weights('model.h5')

#batch: 64, epochs:10
#[0.3038882911205292, 0.8992000222206116] un layer conv2d, 32 
#[0.28579914569854736, 0.9077000021934509] due layer  conv2d, 32-16
#[0.29426005482673645, 0.9010000228881836] due layer conv2d, 32-8
#[0.295382022857666, 0.9010000228881836] tre layer conv2d, 32-16-8
#[0.2905367314815521, 0.8991000056266785] due layer, 64-8

#batch 128 , epochs:10
#[0.2953232526779175, 0.8974000215530396] due layer  conv2d, 32-16

#batch 256 , epochs:10
#[0.2862311899662018, 0.8986999988555908] due layer  conv2d, 32-16