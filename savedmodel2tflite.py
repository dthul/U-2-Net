import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("u2netp_saved_model")
# The tflite converter can't handle non-fixed height and width yet, so we set it here
converter._funcs[0].inputs[0].set_shape((None, 3, 480, 480))
converter.experimental_new_converter = True
converter.allow_custom_ops = True
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open("u2netp_saved_model.tflite", 'wb') as file:
    file.write(tflite_quant_model)
