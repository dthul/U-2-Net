import tensorflow as tf

# Load the *.tflite model and get input details
model = tf.lite.Interpreter(model_path='u2netp_custom_quantized.tflite')
input_details = model.get_input_details()
print(input_details)
output_details = model.get_output_details()
print(output_details)

image_bytes = tf.io.read_file('test_data/test_images/0003.jpg')
image = tf.io.decode_image(image_bytes, dtype=tf.uint8)
image = tf.image.convert_image_dtype(image, dtype=input_details[0]['dtype'])
# TODO: this only works for floating point dtypes
# image = tf.stack([
#     (image[:, :, 0]-0.485)/0.229,
#     (image[:, :, 1]-0.456)/0.224,
#     (image[:, :, 2]-0.406)/0.225],
#     axis=2)

# Your network currently has an input shape (1, 128, 80 , 1),
# but suppose you need the input size to be (2, 128, 200, 1).
# model.resize_tensor_input(
#     input_details[0]['index'], (1, 400, 300, 3))
# model.allocate_tensors()
# input_details = model.get_input_details()
# print(input_details)
# output_details = model.get_output_details()
# print(output_details)
rescaled_images = tf.image.resize([image], input_details[0]['shape'][1:3])

model.allocate_tensors()
model.set_tensor(input_details[0]['index'], rescaled_images)
model.invoke()

output = model.get_tensor(output_details[0]['index'])
rescaled_output = tf.image.resize(output, image.shape[0:2])
mask_bytes = tf.image.encode_png(
    tf.image.convert_image_dtype(rescaled_output[0], dtype=tf.uint8))
tf.io.write_file('tflite_quantized_output.png', mask_bytes)
