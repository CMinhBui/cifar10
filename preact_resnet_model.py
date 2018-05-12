from keras.models import Model
from keras.layers import Input, Add
from keras.layers import Conv2D, AveragePooling2D, Activation, BatchNormalization, Flatten, Dense
from keras.regularizers import l2

def bottleneck_block(input_tensor, num_input_filters, num_res_filters, filter_size, block_index, init='glorot_normal', reg=0.0):
    """
    Parameters:
        input_tensor: a tensor shape: (width, height, num_input_filters)
        num_input_filters: number of previous layer's filters
        num_res_filters: number of filters using in resnet layer
        filter_size: size of the filter using in the middle layer of bottle neck block
        block_index: index of the current block in the resnet layer
        init: kernel initializer for Conv2D layer
        reg: regularization factor
    Output:
        A tensor with shape: (width, height, num_input_filters)
    -------
    Bottle neck block of a resnet layer, using skip connection, all convolution
    without max pooling. It have bottle neck structure:
        A: 1 x 1 x num_res_filters
        B: filter_size x filter_size x num_res_filters
        C: 1 x 1 x num_in_filter
    """
    #name
    bn_name = 'bn' + str(block_index)
    relu_name = 'relu' + str(block_index)
    conv_name = 'conv' + str(block_index)
    add_name = 'add' + str(block_index)

    #block A
    if(block_index > 0):
        x = BatchNormalization(name=bn_name+'A')(input_tensor)
        x = Activation('relu', name=relu_name+'A')(x)
    else:
        x = input_tensor

    x = Conv2D(
        filters=num_res_filters,
        kernel_size=(1,1),
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name+'A'
    )(x)

    #block B
    x = BatchNormalization(name=bn_name+'B')(x)
    x = Activation('relu', name=relu_name+'B')(x)
    x = Conv2D(
        filters=num_res_filters,
        kernel_size=(filter_size, filter_size),
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name+'B'
    )(x)

    #block C
    x = BatchNormalization(name=bn_name+'C')(x)
    x = Activation('relu', name=relu_name+'C')(x)
    x = Conv2D(
        filters=num_input_filters,
        kernel_size=(1, 1),
        kernel_regularizer=l2(reg),
        use_bias=False,
        name=conv_name+'C'
    )(x)

    # merger input and output (shortcut)
    x = Add(name=add_name)([x, input_tensor])
    
    return x

def ResNet_model(input_shape=(32, 32, 3), num_classes=10, layer1_params=(3, 128, 2), res_layer_params=(3, 32, 25), init='glorot_normal', reg=0.0001):
    """
    Parameters:
        input_shape: (width, height, channels)
        num_classes: number of classes or number of scores to produce in the final layer (softmax)
        layer1_params: (filter_size, num_filters, stride), parameters for the first conv layer
        before res layer
        res_layer_params: (filter_size, num_filters, num_blocks), parameters for the resnet layer
        init: conv kernel initializer
        reg: regularization factor
    Outputs:
        resnet model
    """

    inputs = Input(input_shape)
    
    #Convolution layer 1
    filter_size_layer1, num_filters_layer1, stride_layer1 = layer1_params
    x = Conv2D(
        filters=num_filters_layer1,
        kernel_size=(filter_size_layer1, filter_size_layer1),
        padding='same',
        strides=(stride_layer1, stride_layer1),
        kernel_initializer=init,
        kernel_regularizer=l2(reg),
        use_bias=False,
        name='conv_layer1'
    )(inputs)
    x = BatchNormalization(name='bn_layer1')(x)
    x = Activation('relu', name='relu_layer1')(x)

    #ResNet layer
    filter_size_res, num_filters_res, num_blocks = res_layer_params
    for block_index in range(num_blocks):
        x = bottleneck_block(x, num_filters_layer1, num_filters_res, filter_size_res, block_index, init=init, reg=reg)
    
    #Fully connected and softmax layers
    x = BatchNormalization(name='bn_f')(x)
    x = Activation('relu', name='relu_f')(x)

    pool_size = int(input_shape[0]/stride_layer1)
    x = AveragePooling2D(pool_size=(pool_size, pool_size))(x)

    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model