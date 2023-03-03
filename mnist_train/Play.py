from Model import Model
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




if __name__ == "__main__":
    # 加载训练好的模型
    # model = tf.keras.models.load_model('E:\VsCode\my_model.h5')

    #加载图像文件并将其转换为模型所需的格式
    image = Image.open('mypics/3_1.png').convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.reshape((1, 28, 28, 1))
    image_array = image_array / 255.0
    img = image_array.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()

    # 进行预测
    # predictions = model.predict(image_array)

    # 打印预测结果
    # print(predictions)
    m = Model()
    # m.predict(178)
    m.predictMy(image_array)
