import tensorflow as tf
from model import U2NETP, u2netp_tf

net = U2NETP(3, 1)
input = tf.keras.Input(shape=(320, 320, 3), name="img")
output = u2netp_tf(input, 1)
net_tf = tf.keras.Model(input, output)

net_tf.variables

def transfer_weights(state_dict, tf_mod):
    pytorch_submods = list(net.named_modules())[1:]
    for name, mod in mods:

>>> vs[0].name
'stage1.rebnconvin.conv_s1/kernel:0'
>>> vs[1].name
'stage1.rebnconvin.conv_s1/bias:0'
>>> vs[2].name
'stage1.rebnconvin.bn_s1/gamma:0'
>>> vs[3].name
'stage1.rebnconvin.bn_s1/beta:0'
>>> vs[4].name
'stage1.rebnconvin.bn_s1/moving_mean:0'
>>> vs[5].name
'stage1.rebnconvin.bn_s1/moving_variance:0'

For Convolution ('.conv_' in name):
'.weight' -> '/kernel:0'
    [out_c, in_c, k0, k1] -> [k0, k1, in_c, out_c]
'.bias' -> '/bias:0'

For Batch Norm ('.bn_' in name):
'.running_mean' -> '/moving_mean:0'
'.running_var' -> '/moving_variance:0'
'.weight' -> '/gamma:0'
'.bias' -> '/beta:0'
'.num_batches_tracked' -> ignore