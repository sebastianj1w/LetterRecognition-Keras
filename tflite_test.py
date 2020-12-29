import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import time

import tensorflow as tf

# model_path = "./model/quantize_frozen_graph.tflite"
model_path = "./model.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

# # with tf.Session( ) as sess:
# if 1:
#     file_list = os.listdir(test_image_dir)
#
#     model_interpreter_time = 0
#     start_time = time.time()
#     # 遍历文件
#     for file in file_list:
#         print('=========================')
#         full_path = os.path.join(test_image_dir, file)
#         print('full_path:{}'.format(full_path))
#
#         # 只要黑白的，大小控制在(28,28)
#         img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
#         res_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
#         # 变成长784的一维数据
#         new_img = res_img.reshape((784))
#
#         # 增加一个维度，变为 [1, 784]
#         image_np_expanded = np.expand_dims(new_img, axis=0)
#         image_np_expanded = image_np_expanded.astype('float32')  # 类型也要满足要求
#
#         # 填装数据
#         model_interpreter_start_time = time.time()
#         interpreter.set_tensor(input_details[0]['index'], image_np_expanded)
#
#         # 注意注意，我要调用模型了
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         model_interpreter_time += time.time() - model_interpreter_start_time
#
#         # 出来的结果去掉没用的维度
#         result = np.squeeze(output_data)
#         print('result:{}'.format(result))
#         # print('result:{}'.format(sess.run(output, feed_dict={newInput_X: image_np_expanded})))
#
#         # 输出结果是长度为10（对应0-9）的一维数据，最大值的下标就是预测的数字
#         print('result:{}'.format((np.where(result == np.max(result)))[0][0]))
#     used_time = time.time() - start_time
#     print('used_time:{}'.format(used_time))
#     print('model_interpreter_time:{}'.format(model_interpreter_time))