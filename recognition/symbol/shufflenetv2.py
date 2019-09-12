import sys
import os
import mxnet as mx
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def channel_shuffle(data, groups):
    data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
    data = mx.sym.swapaxes(data, 1, 2)
    data = mx.sym.reshape(data, shape=(0, -3, -2))
    return data

def shuffleUnit(residual, in_channels, out_channels, split):
    # for guideline 1
    equal_channels = out_channels / 2

    if split==True:
        DWConv_stride = 1
        # split feature map
        branch1 = mx.sym.slice_axis(residual, axis=1, begin=0, end=in_channels / 2)
        branch2 = mx.sym.slice_axis(residual, axis=1, begin=in_channels / 2, end=in_channels)
    else:
        DWConv_stride = 2
        branch1 = residual
        branch2 = residual

        branch1 = mx.sym.Convolution(data=branch1, num_filter=in_channels, kernel=(3, 3),
                                     pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=in_channels, no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)

        branch1 = mx.sym.Convolution(data=branch1, num_filter=equal_channels,
                                     kernel=(1, 1), stride=(1, 1), no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)
        branch1 = mx.sym.Activation(data=branch1, act_type='relu')


    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels,
    	              kernel=(1, 1), stride=(1, 1), no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = mx.sym.Activation(data=branch2, act_type='relu')


    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(3, 3),
    	               pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=equal_channels, no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)

    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels,
    	               kernel=(1, 1), stride=(1, 1), no_bias=1)
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = mx.sym.Activation(data=branch2, act_type='relu')

    data = mx.sym.concat(branch1, branch2, dim=1)
    #data = mx.contrib.sym.ShuffleChannel(data=data, group=2)
    data = channel_shuffle(data=data, groups=2)

    return data

def shuffleUnitSE(residual, in_channels, out_channels, split):
    # for guideline 1
    equal_channels = int(out_channels / 2)
    print('channels:', in_channels, equal_channels, out_channels)

    if split==True:
        DWConv_stride = 1
        # split feature map
        branch1 = mx.sym.slice_axis(residual, axis=1, begin=0, end=in_channels / 2)
        branch2 = mx.sym.slice_axis(residual, axis=1, begin=in_channels / 2, end=in_channels)
    else:
        DWConv_stride = 2
        branch1 = residual
        branch2 = residual

        branch1 = mx.sym.Convolution(data=branch1, num_filter=in_channels, kernel=(3, 3),
                                     pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=in_channels, no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)

        branch1 = mx.sym.Convolution(data=branch1, num_filter=equal_channels,
                                     kernel=(1, 1), stride=(1, 1), no_bias=1)
        branch1 = mx.sym.BatchNorm(data=branch1)
        branch1 = mx.sym.Activation(data=branch1, act_type='relu')


    if split==False and in_channels != equal_channels:
        print(in_channels, '!=', equal_channels)
        branch2 = mx.sym.Convolution(data=branch2, num_filter=in_channels, kernel=(3, 3),
                                     pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=in_channels, no_bias=1)
        branch2 = mx.sym.BatchNorm(data=branch2)

        branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(1, 1), stride=(1, 1), no_bias=1)
        branch2 = mx.sym.BatchNorm(data=branch2)
        branch2 = mx.sym.Activation(data=branch2, act_type='relu')
    else:
        #se
        body = mx.sym.Pooling(data=branch2, global_pool=True, pool_type='avg')
        body = mx.sym.Convolution(data=body, num_filter=equal_channels//16, kernel=(1,1), stride=(1,1), pad=(0,0))
        body = mx.sym.Activation(data=body, act_type='relu')
        body = mx.sym.Convolution(data=body, num_filter=equal_channels, kernel=(1,1), stride=(1,1), pad=(0,0))
        body = mx.sym.Activation(data=body, act_type='sigmoid')
        branch2 = mx.symbol.broadcast_mul(branch2, body)

        branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(3, 3),
    	               pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=equal_channels, no_bias=1)
        branch2 = mx.sym.BatchNorm(data=branch2)

        branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels,
    	               kernel=(1, 1), stride=(1, 1), no_bias=1)
        branch2 = mx.sym.BatchNorm(data=branch2)
        branch2 = mx.sym.Activation(data=branch2, act_type='relu')

    data = mx.sym.concat(branch1, branch2, dim=1)
    data = channel_shuffle(data=data, groups=2)

    return data

def make_stage(data, stage, multiplier=1):
    stage_repeats = [3, 7, 3]

    if multiplier == 0.5:
        out_channels = [-1, 24, 48, 96, 192]
    elif multiplier == 1:
        out_channels = [-1, 24, 116, 232, 464]
    elif multiplier == 1.5:
        out_channels = [-1, 24, 176, 352, 704]
    elif multiplier == 2:
        #out_channels = [-1, 24, 244, 488, 976]
        out_channels = [-1, 24, 224, 448, 896]
    elif multiplier == 3:
        out_channels = [-1, 24, 348, 696, 1392]

    # DWConv_stride = 2
    data = shuffleUnitSE(data, out_channels[stage - 1], out_channels[stage],
    	               split=False)
    # DWConv_stride = 1
    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnitSE(data, out_channels[stage], out_channels[stage],
        	               split=True)

    return data

def get_symbol():
    num_classes = config.emb_size
    print('in_network', config)
    import ipdb
    #ipdb.set_trace()
    fc_type = config.net_output
    multiplier = config.net_multiplier
    data = mx.symbol.Variable(name="data")
    data = data-127.5
    data = data*0.0078125

    data = mx.sym.Convolution(data=data, num_filter=24,
        	                  kernel=(3, 3), stride=(2, 2), pad=(1, 1), no_bias=1)
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    #data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max',
    #	                  stride=(2, 2), pad=(1, 1))

    data = make_stage(data, 2, multiplier=multiplier)

    data = make_stage(data, 3, multiplier=multiplier)

    data = make_stage(data, 4, multiplier=multiplier)

    # extra_conv
    extra_conv = mx.sym.Convolution(data=data, num_filter=1024,
                                    kernel=(1, 1), stride=(1, 1), no_bias=1)
    extra_conv = mx.sym.BatchNorm(data=extra_conv)
    data = mx.sym.Activation(data=extra_conv, act_type='relu')

    fc1 = symbol_utils.get_fc1(data, num_classes, fc_type, input_channel=1024)
    return fc1
