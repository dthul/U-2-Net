import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './u2netp_saved_model'
graph_pb = 'u2netp.pb'

builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

with tf.io.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.compat.v1.get_default_graph()
    inp = g.get_tensor_by_name("input:0")
    out = g.get_tensor_by_name("d0:0")

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
