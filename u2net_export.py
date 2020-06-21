import os
from skimage import io, transform
import tensorflow as tf
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP, u2netp_tf  # small version u2net 4.7 MB

# normalize the predicted SOD probability map

# Add NHWC checks:


def contains_cl(args):
    for t in args:
        if isinstance(t, torch.Tensor):
            if t.is_contiguous(memory_format=torch.channels_last) and not t.is_contiguous():
                return True
        elif isinstance(t, list) or isinstance(t, tuple):
            if contains_cl(list(t)):
                return True
    return False


def print_inputs(args, indent=''):
    for t in args:
        if isinstance(t, torch.Tensor):
            print(indent, t.stride(), t.shape, t.device, t.dtype)
        elif isinstance(t, list) or isinstance(t, tuple):
            print(indent, type(t))
            print_inputs(list(t), indent=indent + '    ')
        else:
            print(indent, t)


def check_wrapper(fn):
    name = fn.__name__

    def check_cl(*args, **kwargs):
        was_cl = contains_cl(args)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            print('-------------------')
            raise e
        failed = False
        if was_cl:
            if isinstance(result, torch.Tensor):
                if result.dim() == 4 and not result.is_contiguous(memory_format=torch.channels_last):
                    print("`{}` got channels_last input, but output is not channels_last:".format(name),
                          result.shape, result.stride(), result.device, result.dtype)
                    failed = True
        if failed and True:
            print("`{}` inputs are:".format(name))
            print_inputs(args)
            raise Exception(
                'Operator `{}` lost channels_last property'.format(name))
        return result
    return check_cl


def attribute(m):
    for i in dir(m):
        e = getattr(m, i)
        exclude_functions = ['is_cuda', 'has_names', 'numel',
                             'stride', 'Tensor', 'is_contiguous', '__class__']
        if i not in exclude_functions and not i.startswith('_') and '__call__' in dir(e):
            try:
                setattr(m, i, check_wrapper(e))
            except Exception as e:
                print(i)
                print(e)


attribute(torch.Tensor)
attribute(torch.nn.functional)
attribute(torch)


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def find_variable_by_name(name, net):
    for v in net.variables:
        if v.name == name:
            return v
    return None


def transfer_weights(state_dict, net):
    for name, value in state_dict.items():
        is_convolution = '.conv_' in name or name.startswith(
            'side') or name.startswith('outconv')
        is_batchnorm = '.bn_' in name
        assert(not (is_convolution and is_batchnorm))
        if not is_convolution and not is_batchnorm:
            print("Unknown state:", name)
            continue
        elif is_convolution:
            if name.endswith('.weight'):
                target_name = name.replace('.weight', '/kernel:0')
                target_value = value.permute(2, 3, 1, 0).numpy()
            elif name.endswith('.bias'):
                target_name = name.replace('.bias', '/bias:0')
                target_value = value.numpy()
            else:
                print("Unknown convolution state:", name)
                continue
        elif is_batchnorm:
            if name.endswith('.running_mean'):
                target_name = name.replace('.running_mean', '/moving_mean:0')
                target_value = value.numpy()
            elif name.endswith('.running_var'):
                target_name = name.replace(
                    '.running_var', '/moving_variance:0')
                target_value = value.numpy()
            elif name.endswith('.weight'):
                target_name = name.replace('.weight', '/gamma:0')
                target_value = value.numpy()
            elif name.endswith('.bias'):
                target_name = name.replace('.bias', '/beta:0')
                target_value = value.numpy()
            elif name.endswith('.num_batches_tracked'):
                # Ignore
                continue
            else:
                print("Unknown batch normalization state:", name)
                continue
        else:
            # This should be unreachable
            print("Weight transfer error")
            raise "Weight transfer error"
        target_variable = find_variable_by_name(target_name, net)
        if target_variable is None:
            print("Could not find state in Tensorflow model:",
                  name, "->", target_name)
            continue
        target_variable.assign(target_value)


def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2netp'  # u2net

    model_dir = './saved_models/' + model_name + '/' + model_name + '.pth'

    # --------- 2. model define ---------
    if(model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
        input = tf.keras.Input(shape=(320, 320, 3), name="img")
        output = u2netp_tf(input, 1)
        net_tf = tf.keras.Model(input, output)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    transfer_weights(net.state_dict(), net_tf)
    net_tf.save("u2netp_custom")
    net.eval()

    # --------- 3. export ---------
    # Channels last doesn't work yet with PyTorch 1.5.0 since conv2d doesn't
    # keep the memory order. PyTorch 1.5.1 might fix it (https://github.com/pytorch/pytorch/issues/37725)
    # net = net.to(memory_format=torch.channels_last)
    # dummy_input = torch.randn(1, 3, 320, 320).type(
    # torch.FloatTensor).to(memory_format=torch.channels_last)
    torch_out = net(dummy_input)

    output_name = "{}.onnx".format(model_name)
    variable_bwh = {0: 'batch_size', 2: 'width', 3: 'height'}
    torch.onnx.export(
        net,
        dummy_input,
        output_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6'],
        dynamic_axes={
            'input': variable_bwh,
            'd0': variable_bwh,
            'd1': variable_bwh,
            'd2': variable_bwh,
            'd3': variable_bwh,
            'd4': variable_bwh,
            'd5': variable_bwh,
            'd6': variable_bwh,
        })

    # --------- 3. validate ---------
    import onnx
    import onnxruntime
    onnx_model = onnx.load(output_name)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(output_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(torch_out[0]), ort_outputs[0], rtol=1e-3, atol=1e-5)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # for i_test, data_test in enumerate(test_salobj_dataloader):

    #     print("inferencing:",img_name_list[i_test].split("/")[-1])

    #     inputs_test = data_test['image']
    #     inputs_test = inputs_test.type(torch.FloatTensor)

    #     if torch.cuda.is_available():
    #         inputs_test = Variable(inputs_test.cuda())
    #     else:
    #         inputs_test = Variable(inputs_test)

    #     d1,d2,d3,d4,d5,d6,d7 = net(inputs_test)

    #     # normalization
    #     pred = d1[:,0,:,:]
    #     pred = normPRED(pred)

    #     # save results to test_results folder
    #     save_output(img_name_list[i_test],pred,prediction_dir)

    #     del d1,d2,d3,d4,d5,d6,d7


if __name__ == "__main__":
    main()
