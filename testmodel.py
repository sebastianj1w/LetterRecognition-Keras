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


class CnnBN(tf.keras.Model):
    def __init__(self, num_classes=26):
        super(CnnBN, self).__init__(name='CnnBN')
        self.num_classes = num_classes
        self.flatten_layer = layers.Flatten()
        self.fc_layer = layers.Dense(26)
        self.softmax_layer = layers.Softmax()

    def call(self, inputs, training):
        x_image = self.flatten_layer(inputs)
        h_fc1 = self.fc_layer(x_image)
        h_fc1 = self.softmax_layer(h_fc1)
        return h_fc1
        # return self.softmax(y_conv)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


model = CnnBN(num_classes=26)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(input, labels):
    with tf.GradientTape() as tape:
        predictions = model(input, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(input, labels):
    predictions = model(input, training=False)
    t_loss = loss_object(labels, predictions)

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

STEPS = 5001

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

                model.save('./test_model_save/')
                tf.saved_model.save(model, './test_saved_model/')

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
