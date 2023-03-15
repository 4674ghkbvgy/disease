
# import tensorflow as tf
# import numpy as np
# # 加载训练数据和测试数据
# # train_data = np.load('/home/zty/project/disease/dataset/train.npy')
# # test_data = np.load('/home/zty/project/disease/dataset/test.npy')
# train_data = np.load('/home/zty/project/disease/dataset/train.npy', allow_pickle=True)
# test_data = np.load('/home/zty/project/disease/dataset/test.npy', allow_pickle=True)

# # 构建模型
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_data.shape[1:]),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # 编译模型
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 训练模型
# model.fit(train_data, epochs=10, validation_data=test_data)


import tensorflow as tf
import numpy as np

# 加载数据
train_data = np.load('/home/zty/project/disease/dataset/train.npy', allow_pickle=True).item()
test_data = np.load('/home/zty/project/disease/dataset/test.npy', allow_pickle=True).item()

# 将图像数据的像素值范围缩放到 [0, 1] 之间
train_images = train_data['images'] / 255.0
test_images = test_data['images'] / 255.0

# 将标签数据转换为 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_data['labels'])
test_labels = tf.keras.utils.to_categorical(test_data['labels'])

# 定义模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 保存模型
model.save('/home/zty/project/disease/model1.h5')