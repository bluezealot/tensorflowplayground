import numpy
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.datasets import fetch_openml


class Extractor:
    def run(self):
        numpy.load('/home/qinxizhou/work/git/mnist.npz')
        with numpy.load('/home/qinxizhou/work/git/mnist.npz') as data:
            train_examples = data['x_train']
            train_labels = data['y_train']
            test_examples = data['x_test']
            test_labels = data['y_test']
        print("Number of filtered training examples:", len(train_examples))
        print("Number of filtered test examples:", len(test_examples))
        print(train_labels[0])
        print(test_labels[5])
        train_imgs = numpy.asfarray(test_examples[:, 0:])
        img = train_imgs[178].reshape((28, 28))
        plt.imshow(img, cmap="Greys")
        plt.show()


if __name__ == "__main__":
    ext = Extractor()
    ext.run()
