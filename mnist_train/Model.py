import tensorflow as tf
import datetime
import numpy
from PIL import Image


class Model:
    mnist = tf.keras.datasets.mnist

    def create_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name='layers_flatten'),
            tf.keras.layers.Dense(512, activation='sigmoid', name='layers_dense'),
            tf.keras.layers.Dropout(0.2, name='layers_dropout'),
            tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
        ])

    def trainAndSave(self):
        model = self.create_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        path_checkpoint = "training_1/cp.ckpt"
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                           save_weights_only=True,
                                                           verbose=1)
        (x_train, y_train), (x_test, y_test) = self.mnist.load_data()

        image = Image.open('7281677746801_.pic.jpg').convert('L')
        image = image.resize((28, 28))
        image_array = numpy.array(image)
        image_array = image_array.reshape((1, 28, 28))
        x_train = numpy.append(x_train, image_array, axis=0)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model.fit(x=x_train,
                  y=y_train,
                  epochs=5,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback, save_callback])
        model.save('model/m')

    def predict(self, index):
        loaded_model = tf.keras.models.load_model('model/m')
        with numpy.load('/home/qinxizhou/work/git/mnist.npz') as data:
            train_examples = data['x_train']
            train_labels = data['y_train']
            test_examples = data['x_test']
            test_labels = data['y_test']
        predictions = loaded_model.predict(test_examples)
        print(numpy.argmax(predictions[index]))

    def predictMy(self, array):
        loaded_model = tf.keras.models.load_model('model/m')
        with numpy.load('/home/qinxizhou/work/git/mnist.npz') as data:
            train_examples = data['x_train']
            train_labels = data['y_train']
            test_examples = data['x_test']
            test_labels = data['y_test']
        predictions = loaded_model.predict(array)
        print(numpy.argmax(predictions))

    def LoadAllPics(self, path):
        ## TODO

if __name__ == "__main__":
    m = Model()
    m.trainAndSave()
