# Step 1: PyTorch -> ONNX

`python u2net_export.py`

# Step 2: ONNX -> Tensorflow Freeze Graph (needs resize11 branch of onnx-tensorflow)

`onnx-tf convert -i "u2netp.onnx" -o  "u2netp.pb"`

# Step 2b: Visualize Tensorflow Model (useless, Tensorboard gives an unreadable graph)

Convert the Frozen Graph to a Saved Model

`python pb2savedmodel.py`

Convert the Saved Model to a Tensorboard Log

`python -m tensorflow.python.tools.import_pb_to_tensorboard --model_dir u2netp_saved_model --log_dir tensorboard_log`

Open Tensorboard

`tensorboard --logdir=tensorboard_log`

# Step 3: Transform the Tensorflow Graph

TODO: Install custom Tensorflow graph transform (install bazilisk/bazel, prepare and activate virtualenv, install pip dependencies, get Tensorflow source and build the graph transform tool, `./configure`, `bazel build tensorflow/tools/graph_transforms:transform_graph`)

`tensorflow-repo/bazel-bin/tensorflow/tools/graph_transforms/transform_graph --in_graph=u2netp.pb --out_graph=u2netp_transformed.pb --inputs='input' --outputs='d0' --transforms='fold_constants remove_noop_split dilation2d_to_maxpool2d remove_noop_padv2 swap_trans_mul_add swap_trans_relu fold_transposed_pads strip_unused_nodes'`

# Step 4: Tensorflow Freeze Graph -> Tensorflow Lite

`python pb2tflite.py`

or

`toco --enable_v1_converter --graph_def_file u2netp_transformed.pb --output_file u2netp_transformed.tflite --input_arrays input --output_arrays d0 --input_shapes 1,3,320,320`

If you want to do inference on sizes other than 320x320, do something like this at runtime (not so efficient though!):

```python
from tensorflow.contrib.lite.python import interpreter

# Load the *.tflite model and get input details
model = Interpreter(model_path='model.tflite')
input_details = model.get_input_details()

# Your network currently has an input shape (1, 128, 80 , 1),
# but suppose you need the input size to be (2, 128, 200, 1).
model.resize_tensor_input(
    input_details[0]['index'], (2, 128, 200, 1))
model.allocate_tensors()
```

(also works in C++ and Java APIs).

# Building onnxruntime AAR
Didn't build any aar files. Couldn't figure out why.

`./build.sh --config Release --android --android_sdk_path /Users/dthul/Library/Android/sdk --android_ndk_path /Users/dthul/Library/Android/sdk/ndk/21.2.6472646 --android_abi [armeabi-v7a|arm64-v8a|x86|x86_64] --android_api 21`