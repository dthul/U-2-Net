# import tensorflow as tf
# converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
#     "u2netp_transformed.pb", ["input"], ["d0"], input_shapes={"input": (1, 3, 320, 320)})
# converter.experimental_new_converter = True
# converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
# with open("u2netp_transformed.tflite", "wb") as out:
#     out.write(tflite_model)

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("u2netp_custom")
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open("u2netp_custom_quantized.tflite", "wb") as out:
    out.write(tflite_quant_model)
