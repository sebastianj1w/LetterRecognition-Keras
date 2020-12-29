import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cnn_input1

# tf.keras.backend.set_floatx('float64')

my_data = cnn_input1.Data_Control(
    './datapre1049/'
)
n_class = 26

train_x = my_data.traindata
train_x = train_x.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1)
train_y = my_data.trainlabel

Keep_p = 0.6
batch_size = 128

# 测试数据
test_x = my_data.testdata
test_x = test_x.reshape(-1, 252, 9, 1)
test_y = my_data.testlabel

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


class SplitInput(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SplitInput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SplitInput, self).build(input_shape)

    def call(self, inputs):
        x_image = tf.reshape(inputs, [-1, 252, 9, 1])
        x_image_split = tf.split(x_image, 3, 2)
        x_acc = [x_image_split[0]]
        x_gyr = [x_image_split[1]]
        x_gra = [x_image_split[2]]
        x_acc = tf.reshape(x_acc, [-1, 252, 3, 1])
        x_gyr = tf.reshape(x_gyr, [-1, 252, 3, 1])
        x_gra = tf.reshape(x_gra, [-1, 252, 3, 1])
        return x_acc, x_gyr, x_gra

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(SplitInput, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CnnBN(tf.keras.Model):
    def __init__(self, num_classes=26):
        super(CnnBN, self).__init__(name='CnnBN')
        self.num_classes = num_classes

        self.input_layer = SplitInput(input_shape=(None, 252, 9, 1), output_dim=1)

        self.acc_conv_layer_1 = layers.Conv2D(input_shape=(252, 3, 1), filters=16, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True, use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.acc_bn_layer_1 = layers.BatchNormalization(trainable=True, scale=True)
        self.acc_relu_layer_1 = layers.ReLU()
        self.acc_max_pool_layer_1 = layers.MaxPool2D(strides=[5, 1], padding='same')
        self.acc_conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True,  use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.acc_bn_layer_2 = layers.BatchNormalization(trainable=True, scale=True)
        self.acc_relu_layer_2 = layers.ReLU()
        self.acc_max_pool_layer_2 = layers.MaxPool2D(strides=[5, 1], padding='same')

        self.gyr_conv_layer_1 = layers.Conv2D(input_shape=(252, 3, 1), filters=16, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True,  use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.gyr_bn_layer_1 = layers.BatchNormalization(trainable=True, scale=True)
        self.gyr_relu_layer_1 = layers.ReLU()
        self.gyr_max_pool_layer_1 = layers.MaxPool2D(strides=[5, 1], padding='same')
        self.gyr_conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True,  use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.gyr_bn_layer_2 = layers.BatchNormalization(trainable=True, scale=True)
        self.gyr_relu_layer_2 = layers.ReLU()
        self.gyr_max_pool_layer_2 = layers.MaxPool2D(strides=[5, 1], padding='same')

        self.gra_conv_layer_1 = layers.Conv2D(input_shape=(252, 3, 1), filters=16, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True,  use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.gra_bn_layer_1 = layers.BatchNormalization(trainable=True, scale=True)
        self.gra_relu_layer_1 = layers.ReLU()
        self.gra_max_pool_layer_1 = layers.MaxPool2D(strides=[5, 1], padding='same')
        self.gra_conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(8, 3), strides=(1, 1),
                                              padding='same', trainable=True,  use_bias=True,
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.gra_bn_layer_2 = layers.BatchNormalization(trainable=True, scale=True)
        self.gra_relu_layer_2 = layers.ReLU()
        self.gra_max_pool_layer_2 = layers.MaxPool2D( trainable=True, strides=[5, 1], padding='same')

        self.sensor_conv_layer_1 = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                 padding='same', trainable=True,  use_bias=True,
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.sensor_bn_layer_1 = layers.BatchNormalization(trainable=True, scale=True)
        self.sensor_relu_layer_1 = layers.ReLU()
        self.sensor_conv_layer_2 = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                                 padding='same', trainable=True,  use_bias=True,
                                                 bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.sensor_bn_layer_2 = layers.BatchNormalization(trainable=True, scale=True)
        self.sensor_relu_layer_2 = layers.ReLU()
        self.sensor_max_pool_layer = layers.MaxPool3D( trainable=True, strides=[2, 1, 1], padding='same')

        self.flatten_layer = layers.Flatten()
        self.fc_layer = layers.Dense(128, activation='relu', trainable=True,  use_bias=True,
                                     bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.dropout_layer = layers.Dropout(0.1)

        self.dense_layer = layers.Dense(26, use_bias=True,
                                        bias_initializer=tf.keras.initializers.Constant(value=0.1))
        self.softmax = layers.Softmax()

    def call(self, inputs, training):
        # x_image = tf.reshape(inputs, [-1, 252, 9, 1])
        # x_image_split = tf.split(x_image, 3, 2)
        # x_acc = [x_image_split[0]]
        # x_gyr = [x_image_split[1]]
        # x_gra = [x_image_split[2]]
        # x_acc = tf.reshape(x_acc, [-1, my_data.traindata.shape[1], 3, 1])
        # x_gyr = tf.reshape(x_gyr, [-1, my_data.traindata.shape[1], 3, 1])
        # x_gra = tf.reshape(x_gra, [-1, my_data.traindata.shape[1], 3, 1])
        #
        # accinputs, gyrinputs, grainputs = x_acc, x_gyr, x_gra
        accinputs, gyrinputs, grainputs = self.input_layer(inputs)

        keep_prob = 0.9

        accconv1 = self.acc_conv_layer_1(accinputs)
        accBN_out1 = self.acc_bn_layer_1(accconv1, training=training)
        acch_conv1 = self.acc_relu_layer_1(accBN_out1)
        acch_pool1 = self.acc_max_pool_layer_1(acch_conv1)
        accconv2 = self.acc_conv_layer_2(acch_pool1)
        accBN_out2 = self.acc_bn_layer_2(accconv2, training=training)
        acch_conv2 = self.acc_relu_layer_2(accBN_out2)
        acch_pool2 = self.acc_max_pool_layer_2(acch_conv2)

        gyrconv1 = self.gyr_conv_layer_1(gyrinputs)
        gyrBN_out1 = self.gyr_bn_layer_1(gyrconv1, training=training)
        gyrh_conv1 = self.gyr_relu_layer_1(gyrBN_out1)
        gyrh_pool1 = self.gyr_max_pool_layer_1(gyrh_conv1)
        gyrconv2 = self.gyr_conv_layer_2(gyrh_pool1)
        gyrBN_out2 = self.gyr_bn_layer_2(gyrconv2, training=training)
        gyrh_conv2 = self.gyr_relu_layer_2(gyrBN_out2)
        gyrh_pool2 = self.gyr_max_pool_layer_2(gyrh_conv2)

        graconv1 = self.gra_conv_layer_1(grainputs)
        graBN_out1 = self.gra_bn_layer_1(graconv1, training=training)
        grah_conv1 = self.gra_relu_layer_1(graBN_out1)
        grah_pool1 = self.gra_max_pool_layer_1(grah_conv1)
        graconv2 = self.gra_conv_layer_2(grah_pool1)
        graBN_out2 = self.gra_bn_layer_2(graconv2, training=training)
        grah_conv2 = self.gra_relu_layer_2(graBN_out2)
        grah_pool2 = self.gra_max_pool_layer_2(grah_conv2)

        acch_pool3 = tf.expand_dims(acch_pool2, 3)
        gyrh_pool3 = tf.expand_dims(gyrh_pool2, 3)
        grah_pool3 = tf.expand_dims(grah_pool2, 3)
        sensor_conv = tf.concat([acch_pool3, gyrh_pool3, grah_pool3], 3)
        sensorconv1 = self.sensor_conv_layer_1(sensor_conv)
        sensorBN_out1 = self.sensor_bn_layer_1(sensorconv1, training=training)
        sensor_conv = self.sensor_relu_layer_1(sensorBN_out1)

        sensorconv2 = self.sensor_conv_layer_2(sensor_conv)
        sensorBN_out2 = self.sensor_bn_layer_2(sensorconv2, training=training)
        sensor_conv2 = self.sensor_relu_layer_2(sensorBN_out2)
        sensor_conv2 = self.sensor_max_pool_layer(sensor_conv2)
        h_conv = sensor_conv2

        h_flat = self.flatten_layer(h_conv)
        h_fc1 = self.fc_layer(h_flat)

        h_fc1_drop = self.dropout_layer(h_fc1, training=training)

        y_conv = self.dense_layer(h_fc1_drop)
        return y_conv
        # return self.softmax(y_conv)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


model = CnnBN(num_classes=26)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(input, labels):
    with tf.GradientTape() as tape:
        predictions = model(input, training=True)
        loss = tf.reduce_mean(loss_object(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(input, labels):
    predictions = model(input, training=False)
    t_loss = tf.reduce_mean(loss_object(labels, predictions))

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    return predictions


def generatebatch(X, step, batch_size):
    start = (step * batch_size) % len(X)
    if start + batch_size > len(X):
        start = ((step + 1) * batch_size) % len(X)
    end = min(start + batch_size, len(X))
    return start, end  # 生成每一个batch


logdir = "keras_logs"

summary_writer = tf.summary.create_file_writer(logdir)

STEPS = 30001

best_step = 0
best_acc = 0

with summary_writer.as_default():
    tf.summary.trace_on(graph=True, profiler=False)  # 开启Trace，可以记录图结构和profile信息

    for step in range(STEPS):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        random_ind = range(0, len(train_x))

        start, end = generatebatch(random_ind, step, batch_size)
        batch_index = random_ind[start:end]
        batch_x = train_x[batch_index]
        batch_y = train_y[batch_index]
        train_step(batch_x, batch_y)

        if step % 200 == 0:
            pre = test_step(test_x, test_y)

            tf.summary.scalar('loss', train_loss.result(), step=step)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=step)  # 还可以添加其他自定义的变量

            template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(step + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  test_loss.result(),
                                  test_accuracy.result() * 100))

            if step == 0:
                pred_best = pre
            if test_accuracy.result() >= best_acc:
                best_acc = test_accuracy.result()
                best_step = step
                pred_best = pre

                model.save('./keras_model_save/')
                tf.saved_model.save(model, './saved_model/')

                # converter = tf.lite.TFLiteConverter.from_keras_model(model)
                # converter.experimental_new_converter = True
                # # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                #
                # tflite_model = converter.convert()
                # with open('model.tflite', 'wb') as f:
                #     f.write(tflite_model)

        if step % 40 == 0:
            random_ind = list(range(train_x.shape[0]))
            np.random.shuffle(random_ind)

    tf.summary.trace_export(name="model_trace", step=3, profiler_outdir=None)  # 保存Trace信息到文件

print('Bast ACC: %f' % best_acc)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
#               loss=tf.keras.losses.sparse_categorical_crossentropy,
#               metrics=['sparse_categorical_accuracy'])
#
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs/", histogram_freq=1)
# model.fit(train_x, train_y, batch_size=batch_size, epochs=5, callbacks=[tensorboard_callback])
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = True
# # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#
# tflite_model = converter.convert()
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)
#
res = model.predict(train_x[:128])
print(res)
