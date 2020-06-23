import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import tensorflow as tf


def rebnconv_tf(input, out_ch=3, dirate=1, name='rebnconv'):
    name_prefix = name + '.'
    x = tf.keras.layers.Conv2D(
        out_ch, 3, padding='same', dilation_rate=dirate, name=name_prefix + 'conv_s1')(input)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-05, momentum=0.9, name=name_prefix + 'bn_s1')(x)
    x = tf.keras.layers.ReLU(name=name_prefix + 'relu_s1')(x)
    return x


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

# upsample tensor 'src' to have the same spatial size with tensor 'tar'


def _upsample_like_tf(src, tar):
    # Should match PyTorch's F.interpolate(align_corners=False)
    # for a discussion, see: https://machinethink.net/blog/coreml-upsampling/
    if tar.shape[1] is not None and tar.shape[2] is not None \
            and tar.shape[1] // src.shape[1] == 2 \
            and tar.shape[1] % src.shape[1] == 0 \
            and tar.shape[2] // src.shape[2] == 2 \
            and tar.shape[2] % src.shape[2] == 0:
        # Exact upsampling by a factor of 2
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')(src)
    else:
        # Upsampling factor is either not exactly 2 or only known at runtime
        x = tf.image.resize(src, size=tf.shape(tar)[1:3], method='bilinear')
    return x


def _upsample_like(src, tar):

    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')

    return src


def _maxpool2d_tf(x):
    # U-2-Net uses nn.MaxPool2d(ceil_mode=True).
    # Tensorflow does not have such a ceil option for the output size.
    # Instead, we always pad the right and bottom side with 0
    # (I hope those are the correct sides, I didn't check the PyTorch source code yet).
    # Note that this only works properly for inputs which don't contain negative
    # values, which is fine in our case beause U-2-Net only uses MaxPool2D after ReLU.
    # Thus, for even sized images there is no change, while for odd sized images
    # the output size will be "ceiled".
    if x.shape[1] is None or x.shape[2] is None or x.shape[1] % 2 != 0 or x.shape[2] % 2 != 0:
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    return tf.keras.layers.MaxPool2D(strides=2, padding='valid')(x)


### RSU-7 ###


def rsu7_tf(input, mid_ch=12, out_ch=3, name='rsu7'):
    name_prefix = name + '.'
    hxin = rebnconv_tf(input, out_ch, name=name_prefix + 'rebnconvin')

    hx1 = rebnconv_tf(hxin, mid_ch, name=name_prefix + 'rebnconv1')
    # TODO: maybe some padding here, because Tensorflow does not have ceil_mode
    hx = _maxpool2d_tf(hx1)

    hx2 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv2')
    hx = _maxpool2d_tf(hx2)

    hx3 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv3')
    hx = _maxpool2d_tf(hx3)

    hx4 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv4')
    hx = _maxpool2d_tf(hx4)

    hx5 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv5')
    hx = _maxpool2d_tf(hx5)

    hx6 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv6')

    hx7 = rebnconv_tf(hx6, mid_ch, dirate=2, name=name_prefix + 'rebnconv7')

    hx6d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx7, hx6]), mid_ch, name=name_prefix + 'rebnconv6d')
    hx6dup = _upsample_like_tf(hx6d, hx5)

    hx5d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx6dup, hx5]), mid_ch, name=name_prefix + 'rebnconv5d')
    hx5dup = _upsample_like_tf(hx5d, hx4)

    hx4d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx5dup, hx4]), mid_ch, name=name_prefix + 'rebnconv4d')
    hx4dup = _upsample_like_tf(hx4d, hx3)

    hx3d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx4dup, hx3]), mid_ch, name=name_prefix + 'rebnconv3d')
    hx3dup = _upsample_like_tf(hx3d, hx2)

    hx2d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx3dup, hx2]), mid_ch, name=name_prefix + 'rebnconv2d')
    hx2dup = _upsample_like_tf(hx2d, hx1)

    hx1d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx2dup, hx1]), out_ch, name=name_prefix + 'rebnconv1d')

    return tf.keras.layers.add([hx1d, hxin])


class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-6 ###


def rsu6_tf(input, mid_ch=12, out_ch=3, name='rsu6'):
    name_prefix = name + '.'
    hxin = rebnconv_tf(input, out_ch, name=name_prefix + 'rebnconvin')

    hx1 = rebnconv_tf(hxin, mid_ch, name=name_prefix + 'rebnconv1')
    # TODO: maybe some padding here, because Tensorflow does not have ceil_mode
    hx = _maxpool2d_tf(hx1)

    hx2 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv2')
    hx = _maxpool2d_tf(hx2)

    hx3 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv3')
    hx = _maxpool2d_tf(hx3)

    hx4 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv4')
    hx = _maxpool2d_tf(hx4)

    hx5 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv5')

    hx6 = rebnconv_tf(hx5, mid_ch, dirate=2, name=name_prefix + 'rebnconv6')

    hx5d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx6, hx5]), mid_ch, name=name_prefix + 'rebnconv5d')
    hx5dup = _upsample_like_tf(hx5d, hx4)

    hx4d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx5dup, hx4]), mid_ch, name=name_prefix + 'rebnconv4d')
    hx4dup = _upsample_like_tf(hx4d, hx3)

    hx3d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx4dup, hx3]), mid_ch, name=name_prefix + 'rebnconv3d')
    hx3dup = _upsample_like_tf(hx3d, hx2)

    hx2d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx3dup, hx2]), mid_ch, name=name_prefix + 'rebnconv2d')
    hx2dup = _upsample_like_tf(hx2d, hx1)

    hx1d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx2dup, hx1]), out_ch, name=name_prefix + 'rebnconv1d')

    return tf.keras.layers.add([hx1d, hxin])


class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-5 ###


def rsu5_tf(input, mid_ch=12, out_ch=3, name='rsu5'):
    name_prefix = name + '.'
    hxin = rebnconv_tf(input, out_ch, name=name_prefix + 'rebnconvin')

    hx1 = rebnconv_tf(hxin, mid_ch, name=name_prefix + 'rebnconv1')
    # TODO: maybe some padding here, because Tensorflow does not have ceil_mode
    hx = _maxpool2d_tf(hx1)

    hx2 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv2')
    hx = _maxpool2d_tf(hx2)

    hx3 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv3')
    hx = _maxpool2d_tf(hx3)

    hx4 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv4')

    hx5 = rebnconv_tf(hx4, mid_ch, dirate=2, name=name_prefix + 'rebnconv5')

    hx4d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx5, hx4]), mid_ch, name=name_prefix + 'rebnconv4d')
    hx4dup = _upsample_like_tf(hx4d, hx3)

    hx3d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx4dup, hx3]), mid_ch, name=name_prefix + 'rebnconv3d')
    hx3dup = _upsample_like_tf(hx3d, hx2)

    hx2d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx3dup, hx2]), mid_ch, name=name_prefix + 'rebnconv2d')
    hx2dup = _upsample_like_tf(hx2d, hx1)

    hx1d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx2dup, hx1]), out_ch, name=name_prefix + 'rebnconv1d')

    return tf.keras.layers.add([hx1d, hxin])


class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4 ###


def rsu4_tf(input, mid_ch=12, out_ch=3, name='rsu4'):
    name_prefix = name + '.'
    hxin = rebnconv_tf(input, out_ch, name=name_prefix + 'rebnconvin')

    hx1 = rebnconv_tf(hxin, mid_ch, name=name_prefix + 'rebnconv1')
    # TODO: maybe some padding here, because Tensorflow does not have ceil_mode
    hx = _maxpool2d_tf(hx1)

    hx2 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv2')
    hx = _maxpool2d_tf(hx2)

    hx3 = rebnconv_tf(hx, mid_ch, name=name_prefix + 'rebnconv3')

    hx4 = rebnconv_tf(hx3, mid_ch, dirate=2, name=name_prefix + 'rebnconv4')

    hx3d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx4, hx3]), mid_ch, name=name_prefix + 'rebnconv3d')
    hx3dup = _upsample_like_tf(hx3d, hx2)

    hx2d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx3dup, hx2]), mid_ch, name=name_prefix + 'rebnconv2d')
    hx2dup = _upsample_like_tf(hx2d, hx1)

    hx1d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx2dup, hx1]), out_ch, name=name_prefix + 'rebnconv1d')

    return tf.keras.layers.add([hx1d, hxin])


class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4F ###


def rsu4f_tf(input, mid_ch=12, out_ch=3, name='rsu4f'):
    name_prefix = name + '.'
    hxin = rebnconv_tf(input, out_ch, name=name_prefix + 'rebnconvin')

    hx1 = rebnconv_tf(hxin, mid_ch, name=name_prefix + 'rebnconv1')
    hx2 = rebnconv_tf(hx1, mid_ch, dirate=2, name=name_prefix + 'rebnconv2')
    hx3 = rebnconv_tf(hx2, mid_ch, dirate=4, name=name_prefix + 'rebnconv3')
    hx4 = rebnconv_tf(hx3, mid_ch, dirate=8, name=name_prefix + 'rebnconv4')

    hx3d = rebnconv_tf(tf.keras.layers.Concatenate()
                       ([hx4, hx3]), mid_ch, dirate=4, name=name_prefix + 'rebnconv3d')
    hx2d = rebnconv_tf(tf.keras.layers.Concatenate()
                       ([hx3d, hx2]), mid_ch, dirate=2, name=name_prefix + 'rebnconv2d')
    hx1d = rebnconv_tf(tf.keras.layers.Concatenate()(
        [hx2d, hx1]), out_ch, name=name_prefix + 'rebnconv1d')

    return tf.keras.layers.add([hx1d, hxin])


class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

### U^2-Net small ###


def u2netp_tf(input, out_ch=1):
    # image pre-processing
    hx = tf.nn.bias_add(input, tf.constant([-0.485, -0.456, -0.406]))
    hx = tf.divide(hx, tf.constant([0.229, 0.224, 0.225]))

    # stage 1
    hx1 = rsu7_tf(hx, 16, 64, name='stage1')
    hx = _maxpool2d_tf(hx1)

    # stage 2
    hx2 = rsu6_tf(hx, 16, 64, name='stage2')
    hx = _maxpool2d_tf(hx2)

    # stage 3
    hx3 = rsu5_tf(hx, 16, 64, name='stage3')
    hx = _maxpool2d_tf(hx3)

    # stage 4
    hx4 = rsu4_tf(hx, 16, 64, name='stage4')
    hx = _maxpool2d_tf(hx4)

    # stage 5
    hx5 = rsu4f_tf(hx, 16, 64, name='stage5')
    hx = _maxpool2d_tf(hx5)

    # stage 6
    hx6 = rsu4f_tf(hx, 16, 64, name='stage6')
    hx6up = _upsample_like_tf(hx6, hx5)

    # decoder
    hx5d = rsu4f_tf(tf.keras.layers.concatenate(
        [hx6up, hx5]), 16, 64, name='stage5d')
    hx5dup = _upsample_like_tf(hx5d, hx4)

    hx4d = rsu4_tf(tf.keras.layers.concatenate(
        [hx5dup, hx4]), 16, 64, name='stage4d')
    hx4dup = _upsample_like_tf(hx4d, hx3)

    hx3d = rsu5_tf(tf.keras.layers.concatenate(
        [hx4dup, hx3]), 16, 64, name='stage3d')
    hx3dup = _upsample_like_tf(hx3d, hx2)

    hx2d = rsu6_tf(tf.keras.layers.concatenate(
        [hx3dup, hx2]), 16, 64, name='stage2d')
    hx2dup = _upsample_like_tf(hx2d, hx1)

    hx1d = rsu7_tf(tf.keras.layers.concatenate(
        [hx2dup, hx1]), 16, 64, name='stage1d')

    # side output
    d1 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side1')(hx1d)

    d2 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side2')(hx2d)
    d2 = _upsample_like_tf(d2, d1)

    d3 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side3')(hx3d)
    d3 = _upsample_like_tf(d3, d1)

    d4 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side4')(hx4d)
    d4 = _upsample_like_tf(d4, d1)

    d5 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side5')(hx5d)
    d5 = _upsample_like_tf(d5, d1)

    d6 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', name='side6')(hx6)
    d6 = _upsample_like_tf(d6, d1)

    d0 = tf.keras.layers.Conv2D(out_ch, 1, name='outconv')(
        tf.keras.layers.concatenate([d1, d2, d3, d4, d5, d6]))

    return tf.keras.activations.sigmoid(d0)


class U2NETP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
