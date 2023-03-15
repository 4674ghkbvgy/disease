import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical

# 加载数据
train_data = np.load('/home/zty/project/disease/dataset/train.npy', allow_pickle=True).item()
test_data = np.load('/home/zty/project/disease/dataset/test.npy', allow_pickle=True).item()

# 数据预处理
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_data['images'], to_categorical(train_data['labels']), batch_size=32)
test_generator = test_datagen.flow(test_data['images'], to_categorical(test_data['labels']), batch_size=32)

# 构建模型
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 模型编译
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

# 模型训练
history = model.fit(train_generator, epochs=20, validation_data=test_generator)

# 模型评估
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

model.save('/home/zty/project/disease/modelVGG.h5')