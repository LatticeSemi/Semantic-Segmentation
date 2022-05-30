from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages
#import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=256, weight_decay=0.0001, transfer_learning=False, tl_fine_tune=False):
    #depth = [16, 16, 32, 32, 32, 44, 48]  # 32KB for activation,  96KB for program (train_32_96_n3)
    #depth = [96, 192, 192, 192, 288, 288, 288, 384]  # 32KB for activation,  96KB for program (train_32_96_n3)
    depth = [10, 10, 40, 40, 80, 80, 160, 160]  # 32KB for activation,  96KB for program (train_32_96_n3)

    ####################################################################
    # Quantization layers
    ####################################################################
    if True:
        fl_w_bin = 8
        fl_a_bin = 8 

        ml_w_bin = 8
        ml_a_bin = 8

        ll_w_bin = 8
        ll_a_bin = 16 # 16b results

        min_rng =  0.0 # range of quanized activation
        max_rng =  2.0

        bias_on = False # no bias for T+

    ####################################################################
    if True:
      #channels = tf.unstack(images, axis=-1) # images=BGR, images_viz=RGB
      #images_viz = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
      images_viz = images
      tf.summary.image('images', images_viz)

    freeze_layers     = transfer_learning # if transfer_learning, freeze variables to stop further training
    freeze_last_layer = transfer_learning and not tl_fine_tune # last layer is frozen if no fine tune under transfer_learning

    fire1 = _conv3x3_layer('fire1', images, oc=depth[0], freeze=freeze_layers, w_bin=fl_w_bin, a_bin=fl_a_bin, pool_en=True, 
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire2 = _conv3x3_layer('fire2', fire1,  oc=depth[1], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, 
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire3 = _conv3x3_layer('fire3', fire2,  oc=depth[2], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True,
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire4 = _conv3x3_layer('fire4', fire3,  oc=depth[3], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, 
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire5 = _conv3x3_layer('fire5', fire4,  oc=depth[4], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True,
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire6 = _conv3x3_layer('fire6', fire5,  oc=depth[5], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=False, 
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire7 = _conv3x3_layer('fire7', fire6,  oc=depth[6], freeze=freeze_layers, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True,
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire8 = _conv3x3_layer('fire8', fire7,  oc=depth[7], freeze=freeze_last_layer, w_bin=ml_w_bin, a_bin=ml_a_bin, pool_en=True,
                                                     min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, phase_train=phase_train)
    fire_o = fire8
    ####################################################################
    net = _fc_layer('fire9', fire_o, bottleneck_layer_size, flatten=True, relu=False, xavier=True, 
                    w_bin=ll_w_bin, a_bin=ll_a_bin, min_rng=min_rng, max_rng=max_rng)
    print('fire9:', net)


    #return net, None, vars2save
    return net, None
    #return net, vars2save

def _conv3x3_layer(layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, 
               dilations=1, relu=True, min_rng=-0.5, max_rng=0.5, bias_on=True, phase_train=True):
    with tf.variable_scope(layer_name):
        net = _conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1, dilations=dilations, xavier=False,
                          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on)
        #tf.summary.histogram('before_bn', net)

        net = _batch_norm_tensor2('bn', net, freeze=freeze, phase_train=phase_train) # BatchNorm
        #tf.summary.histogram('before_relu', net)

        if relu:
            net = binary_wrapper(net, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # ReLU

        #tf.summary.histogram('after_relu', net)
        if pool_en:
            pool = _pooling_layer('pool', net, size=2, stride=2, padding='SAME')
        else:
            pool = net
        #tf.summary.histogram('pool', pool)

        return pool

def _conv1x1_layer(layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True, 
               relu=True, min_rng=-0.5, max_rng=0.5, bias_on=True, phase_train=True):
    with tf.variable_scope(layer_name):
        net = _conv_layer('conv1x1', inputs, filters=oc, size=1, stride=1, xavier=False,
                          padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin, bias_on=bias_on)
        #tf.summary.histogram('before_bn', net)

        net = _batch_norm_tensor2('bn', net, freeze=freeze, phase_train=phase_train) # BatchNorm
        #tf.summary.histogram('before_relu', net)

        if relu:
            net = binary_wrapper(net, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng) # ReLU

        #tf.summary.histogram('after_relu', net)
        if pool_en:
            pool = _pooling_layer('pool', net, size=2, stride=2, padding='SAME')
        else:
            pool = net
        #tf.summary.histogram('pool', pool)

        return pool

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):
  min_clip = tf.rint(min_rng*256/(max_rng-min_rng))
  max_clip = tf.rint(max_rng*256/(max_rng-min_rng)) - 1 # 127, 255

  wq = 256.0 * w / (max_rng - min_rng)              # to expand [min, max] to [-128, 128]
  wq = tf.rint(wq)                                  # integer (quantization)
  wq = tf.clip_by_value(wq, min_clip, max_clip)     # fit into 256 linear quantization
  wq = wq / 256.0 * (max_rng - min_rng)             # back to quantized real number, not integer
  wclip = tf.clip_by_value(w, min_rng, max_rng)     # linear value w/ clipping
  return wclip + tf.stop_gradient(wq - wclip)

def binary_wrapper(x, a_bin=16, min_rng=-0.5, max_rng=0.5): # activation binarization
  #if a_bin == 1:
  #  return binary_tanh(x)
  if a_bin == 8:
    x_quant = lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)
    return tf.nn.relu(x_quant)
  else:
    return tf.nn.relu(x)

def _batch_norm(name, x, phase_train=True): # works well w/ phase_train python variable
  with tf.variable_scope(name):
    params_shape = [x.get_shape()[-1]]

    beta  = tf.get_variable('beta',  params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    #tf.summary.histogram('bn_gamma', gamma)
    #tf.summary.histogram('bn_beta',  beta )

    control_inputs = []

    if phase_train:
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

      moving_mean = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
      moving_variance = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

      update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9)
      update_moving_var  = moving_averages.assign_moving_average(moving_variance, variance, 0.9)
      control_inputs = [update_moving_mean, update_moving_var]
    else:
      mean = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
      variance = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
   
    #self.model_params += [gamma, beta, mean, variance] # <- to save in snapshot
    with tf.control_dependencies(control_inputs):
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
    y.set_shape(x.get_shape())
    #return y, [gamma, beta, mean, variance]
    return y

# reference code from https://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def _batch_norm_tensor(name, x, phase_train=True): 
  with tf.variable_scope(name):
    params_shape = [x.get_shape()[-1]]

    beta  = tf.get_variable('beta',  params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    #tf.summary.histogram('bn_gamma', gamma)
    #tf.summary.histogram('bn_beta',  beta )
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, 
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    normed.set_shape(x.get_shape())
    return normed

def _batch_norm_tensor2(name, x, freeze=False, phase_train=True): # works well for phase_train tensor
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    params_shape = [x.get_shape()[-1]]

    beta  = tf.get_variable('beta',  params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=(not freeze))
    gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=(not freeze))
    #tf.summary.histogram('bn_gamma', gamma)
    #tf.summary.histogram('bn_beta',  beta )


    def mean_var_4_train():
      mean_train, variance_train = tf.nn.moments(x, [0, 1, 2], name='moments')

      moving_mean = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
      moving_variance = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

      update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean_train, 0.9)
      update_moving_var  = moving_averages.assign_moving_average(moving_variance, variance_train, 0.9)
      control_inputs_train = [update_moving_mean, update_moving_var]
      with tf.control_dependencies(control_inputs_train):
        return tf.identity(mean_train), tf.identity(variance_train) #, [control_inputs_train]

    def mean_var_4_eval():
      mean_eval = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
      variance_eval = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
      return mean_eval, variance_eval
   
    #mean, variance= tf.cond(phase_train, mean_var_4_train, mean_var_4_eval)
    if phase_train == True:
        mean, variance= mean_var_4_train()
    else:
        mean, variance= mean_var_4_eval()
    y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
    y.set_shape(x.get_shape())
    return y


def _conv_layer(layer_name, inputs, filters, size, stride, dilations=1, padding='SAME',
    freeze=False, xavier=False, relu=True, w_bin=16, bias_on=True, stddev=0.001):
  """Convolutional layer operation constructor.

  Args:
    layer_name: layer name.
    inputs: input tensor
    filters: number of output filters.
    size: kernel size.
    stride: stride
    padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    freeze: if true, then do not train the parameters in this layer.
    xavier: whether to use xavier weight initializer or not.
    relu: whether to use relu or not.
    stddev: standard deviation used for random weight initializer.
  Returns:
    A convolutional layer operation.
  """

  with tf.variable_scope(layer_name) as scope:
    channels = inputs.get_shape()[3]

    # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
    # shape [h, w, in, out]
    if xavier:
      kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      bias_init = tf.constant_initializer(0.0)
    else:
      kernel_init = tf.truncated_normal_initializer(
          stddev=stddev, dtype=tf.float32)
      bias_init = tf.constant_initializer(0.0)

    kernel = _variable_with_weight_decay(
      'kernels', shape=[size, size, int(channels), filters], wd=0.0001, initializer=kernel_init, trainable=(not freeze))

    #if w_bin == 1: # binarized conv
    #  kernel_bin = binarize(kernel)
    #  tf.summary.histogram('kernel_bin', kernel_bin)
    #  conv = tf.nn.conv2d(inputs, kernel_bin, [1, stride, stride, 1], padding=padding, name='convolution')
    #  conv_bias = conv
    if w_bin == 8: # 8b quantization
      kernel_quant = lin_8b_quant(kernel)
      #tf.summary.histogram('kernel_quant', kernel_quant)
      if dilations == 1:
        conv = tf.nn.conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution')
      else:
        conv = tf.nn.conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding='SAME', dilations=dilations, name='convolution')
        #paddings = tf.convert_to_tensor([[0,0], [dilations, dilations], [dilations, dilations], [0, 0]])
        #inputs_pad = tf.pad(inputs, paddings=paddings, name='input_pad')
        #conv = tf.nn.conv2d(inputs_pad, kernel_quant, [1, stride, stride, 1], padding='VALID', dilations=dilations, name='convolution')

      if bias_on:
        biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
        biases_quant = lin_8b_quant(biases)
        #tf.summary.histogram('biases_quant', biases_quant)
        conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')
      else:
        conv_bias = conv
    else: # 16b quantization
      if dilations == 1:
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
      else:
        conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=[[0, 0], [0, 0], [0, 0], [0, 0]], dilations=dilations, name='convolution')
      if bias_on:
        biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
        conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      else:
        conv_bias = conv
  
    if relu:
      out = tf.nn.relu(conv_bias, 'relu')
    else:
      out = conv_bias

    return out

def _pooling_layer(layer_name, inputs, size, stride, padding='SAME'):
  """Pooling layer operation constructor.

  Args:
    layer_name: layer name.
    inputs: input tensor
    size: kernel size.
    stride: stride
    padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
  Returns:
    A pooling layer operation.
  """

  with tf.variable_scope(layer_name) as scope:
    out =  tf.nn.max_pool(inputs, 
                          ksize=[1, size, size, 1], 
                          strides=[1, stride, stride, 1],
                          padding=padding)
    return out

  
def _fc_layer(layer_name, inputs, hiddens, flatten=False, relu=True, xavier=False, stddev=0.001, w_bin=16, a_bin=16, 
              min_rng=0.0, max_rng=2.0):
  """Fully connected layer operation constructor.

  Args:
    layer_name: layer name.
    inputs: input tensor
    hiddens: number of (hidden) neurons in this layer.
    flatten: if true, reshape the input 4D tensor of shape 
        (batch, height, weight, channel) into a 2D tensor with shape 
        (batch, -1). This is used when the input to the fully connected layer
        is output of a convolutional layer.
    relu: whether to use relu or not.
    xavier: whether to use xavier weight initializer or not.
    stddev: standard deviation used for random weight initializer.
  Returns:
    A fully connected layer operation.
  """

  with tf.variable_scope(layer_name) as scope:
    input_shape = inputs.get_shape().as_list()
    if flatten:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs = tf.reshape(inputs, [-1, dim])
    else:
      dim = input_shape[1]

    if xavier:
      kernel_init = tf.contrib.layers.xavier_initializer()
      bias_init = tf.constant_initializer(0.0)
    else:
      kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
      bias_init = tf.constant_initializer(0.0)

    weights = _variable_with_weight_decay('weights', shape=[dim, hiddens], wd=0.0001, initializer=kernel_init)
    biases = _variable_on_device('biases', [hiddens], bias_init)

    #====================
    if w_bin == 8: # 8b quantization
      weights_quant = lin_8b_quant(weights)
    else: # 16b quantization
      weights_quant = weights
    #tf.summary.histogram('weights_quant', weights_quant)
    #====================
    # no quantization on bias since it will be added to the 16b MUL output
    #====================

    outputs = tf.nn.bias_add(tf.matmul(inputs, weights_quant), biases)
    #tf.summary.histogram('outputs', outputs)

    if a_bin == 8:
      outputs_quant = lin_8b_quant(outputs, min_rng=min_rng, max_rng=max_rng)
    else:
      outputs_quant = outputs
    #tf.summary.histogram('outputs_quant', outputs_quant)

    if relu:
      outputs = tf.nn.relu(outputs_quant, 'relu')

    # count layer stats

    return outputs
