from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/support/metadata/metadata_schema.fbs
# for the schema

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "U^2 Net"
model_meta.description = ("Salient Object Detection")
model_meta.version = "v1"
model_meta.author = "Xuebin Qin"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be processed. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(320, 320))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [0.0]
input_normalization.options.std = [255.0]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "segmentation"
output_meta.description = "Greyscale segmentation. White is foreground, black is background."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
output_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.GRAYSCALE
output_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties
output_meta.processUnits = []
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats

# Creates subgraph info.
subgraph = _metadata_fb.SubGraphMetadataT()
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]
model_meta.subgraphMetadata = [subgraph]

b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

populator = _metadata.MetadataPopulator.with_model_file("u2netp_custom.tflite")
populator.load_metadata_buffer(metadata_buf)
populator.populate()
