import tensorflow as tf

# Load the *.tflite model and get input details
model = tf.lite.Interpreter(model_path='u2netp_custom.tflite')
input_details = model.get_input_details()
print(input_details)
output_details = model.get_output_details()
print(output_details)

image_bytes = tf.io.read_file('test_data/test_images/0003.jpg')
image = tf.io.decode_image(image_bytes, dtype=tf.uint8)
image = tf.image.convert_image_dtype(image, dtype=input_details[0]['dtype'])

# Your network currently has an input shape (1, 128, 80 , 1),
# but suppose you need the input size to be (2, 128, 200, 1).
model.resize_tensor_input(
    input_details[0]['index'], (1, 400, 300, 3))
# model.allocate_tensors()
input_details = model.get_input_details()
print(input_details)
output_details = model.get_output_details()
print(output_details)
rescaled_images = tf.image.resize([image], input_details[0]['shape'][1:3])

model.allocate_tensors()
model.set_tensor(input_details[0]['index'], rescaled_images)
model.invoke()

output = model.get_tensor(output_details[0]['index'])

mask_bytes = tf.image.encode_png(
    tf.image.convert_image_dtype(output[0], dtype=tf.uint8))
tf.io.write_file('tflite_output.png', mask_bytes)
