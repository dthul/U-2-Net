import tensorflow as tf
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    "u2netp_transformed.pb", ["input"], ["d0"], input_shapes={"input": (1, 3, 320, 320)})
converter.experimental_new_converter = True
converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("u2netp_transformed.tflite", "wb") as out:
    out.write(tflite_model)
