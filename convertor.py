import tensorflow as tf
import cnn_input1

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# my_data = cnn_input1.Data_Control(
#     './datapre1049/'
# )
# n_class = 26
#
# train_x = my_data.traindata
# train_x = train_x.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1)
# train_y = my_data.trainlabel
#
# Keep_p = 0.6
# batch_size = 128
#
# # 测试数据
# test_x = my_data.testdata
# test_x = test_x.reshape(-1, 252, 9, 1)
# test_y = my_data.testlabel

#
# def representative_dataset():
#     for data in tf.data.Dataset.from_tensor_slices(test_x).batch(1).take(100):
#         yield [data.astype(tf.float32)]
#

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./keras_model_save/')  # path to the SavedModel directory
converter.experimental_new_converter = True
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.representative_dataset = representative_dataset
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('LetterRecognitionModel.tflite', 'wb') as f:
    f.write(tflite_model)
    f.close()
