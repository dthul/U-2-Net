# Step 0: Install Dependencies

`pip install torch==1.5.0 TensorFlow==2.2.0 tflite-support==0.1.0a1`

# Step 1: Create the TensorFlow Model

Run the `u2net_export.py` script, which constructs the original PyTorch model as well as our TensorFlow reimplementation, transfers the original weights to the TensorFlow model, and saves the TensorFlow model to disk using the SavedModel format:

`python u2net_export.py`

The model will be exported to the `u2netp_custom/` folder.

# Step 2: TensorFlow SavedModel -> TensorFlow Lite

This step converts the TensorFlow model to a TensorFlow Lite model, applying optimizations (such as fusing convolution, bias, batch norm and activation) and quantization along the way.

`python pb2tflite.py`

The resulting TensorFlow Lite model will be saved as `u2netp_custom_quantized.tflite`.

# Step 3: Add Metadata to the TensorFlow Lite model

The bare TensorFlow Lite model can be enriched with metadata to describe the inputs and outputs. This allows for example Android Studio (>= 4.2) to automatically create binding code for the TensorFlow Lite model.

`python add_tflite_metadata.py`

This command modifies the `u2netp_custom_quantized.tflite` file in-place.

# Step 4: Optionally, Check that the Model Works

Run `python tflite_inference.py`.

The resulting segmentation mask will be stored in `tflite_quantized_output.png`.

# Step 5: Generate Android Wrapper Code

Either just import `u2netp_custom_quantized.tflite` into Android Studio (`New -> Other -> TensorFlow Lite Model`), or manually create Java binding code:

```
tflite_codegen --model=u2netp_custom_quantized.tflite \
    --package_name=com.weboloco.android.segmentation \
    --model_class_name=U2NetModel \
    --destination=./u2net_android_wrapper
```
