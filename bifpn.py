import tensorflow as tf
import itertools
import numpy as np
class SeparableConvBlock(tf.keras.layers.Layer):

  def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
    super(SeparableConvBlock, self).__init__()
    if out_channels is None:
      out_channels = in_channels
    self.out_channels=out_channels
    self.in_channels=in_channels
    self.norm=norm
    self.activation = activation
    self._build()
    # Q: whether separate conv
    #  share bias between depthwise_conv and pointwise_conv
    #  or just pointwise_conv apply bias.
    # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
  def _build(self):
    self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
             kernel_size=[3,3],
             strides=1,
             depthwise_initializer='normal',
             padding='same',
             data_format='channels_last',
             use_bias=False)
    self.pointwise_conv =tf.keras.layers.Conv2D(
        	filters=self.out_channels,
        	kernel_size=[1, 1],
        	strides=[1, 1],
        	kernel_initializer='normal',
        	padding='same',
        	data_format='channels_last',
        	use_bias=True)
    if self.norm:
      self.bn=tf.keras.layers.BatchNormalization(axis=-1,
               momentum=0.99,
               epsilon=1e-3)
    if self.activation:
       self._relu_fn = tf.keras.layers.ELU()
  def call(self, inputs):
    x = inputs
    x = self.depthwise_conv(x)
    x = self.pointwise_conv(x)

    if self.norm:
      x = self.bn(x)

    if self.activation:
      x = self._relu_fn(x)

    return x

class BiFPN(tf.keras.layers.Layer):
  def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, attention=False):
    super(BiFPN, self).__init__()
    self.epsilon = epsilon
    self._relu_fn = tf.keras.layers.ELU()
    self.first_time = first_time
    self.num_channels=num_channels
    self.conv_channels=conv_channels
    self._build()
  def _build(self):
    # Conv layers
    self.conv6_up = SeparableConvBlock(self.num_channels)
    self.conv5_up = SeparableConvBlock(self.num_channels)
    self.conv4_up = SeparableConvBlock(self.num_channels)
    self.conv3_up = SeparableConvBlock(self.num_channels)
    self.conv4_down = SeparableConvBlock(self.num_channels)
    self.conv5_down = SeparableConvBlock(self.num_channels)
    self.conv6_down = SeparableConvBlock(self.num_channels)
    self.conv7_down = SeparableConvBlock(self.num_channels)
    # Feature scaling layers
    self.p6_upsample =tf.image.resize
    self.p5_upsample =tf.image.resize
    self.p4_upsample =tf.image.resize
    self.p3_upsample =tf.image.resize
     
    self.p4_downsample = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
    self.p5_downsample = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
    self.p6_downsample = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
    self.p7_downsample = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
    
    if self.first_time:
      self.p5_down_channel=[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
      self.p4_down_channel=[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
      self.p3_down_channel=[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
      self.p5_to_p6 =[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3),tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
]
      self.p6_to_p7=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same',data_format='channels_last')
      self.p4_down_channel_2=[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
      self.p5_down_channel_2=[tf.keras.layers.Conv2D(filters=self.num_channels,kernel_size=[1, 1], strides=[1, 1],
        kernel_initializer='normal',
        padding='same',
        data_format='channels_last',
        use_bias=True),tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=1e-3)]
  def call(self,inputs):
    
    if self.first_time:
      p3, p4, p5 = inputs
      p6_in=p5
      for _ in self.p5_to_p6:
        p6_in=_(p6_in)
      p7_in=self.p6_to_p7(p6_in)
      p3_in=p3
      p4_in=p4
      p5_in=p5
      for _ in self.p3_down_channel:
        p3_in=_(p3_in)
      for _ in self.p4_down_channel:
        p4_in=_(p4_in)
      for _ in self.p5_down_channel:
        p5_in=_(p5_in)
      
    else:
      # P3_0, P4_0, P5_0, P6_0 and P7_0
      p3_in, p4_in, p5_in, p6_in, p7_in = inputs
    p6_up = self.conv6_up(self._relu_fn(p6_in + self.p6_upsample(p7_in,[p7_in.shape[1]*2,p7_in.shape[2]*2],method='nearest')))
    
    p5_up = self.conv5_up(self._relu_fn(p5_in + self.p5_upsample(p6_up,[p6_up.shape[1]*2,p6_up.shape[2]*2],method='nearest')))
    p4_up = self.conv4_up(self._relu_fn(p4_in + self.p4_upsample(p5_up,[p5_up.shape[1]*2,p5_up.shape[2]*2],method='nearest')))
    p3_out = self.conv3_up(self._relu_fn(p3_in + self.p3_upsample(p4_up,[p4_up.shape[1]*2,p4_up.shape[2]*2],method='nearest')))
    if self.first_time:
      
      p4_in=p4
      p5_in=p5
      for _ in self.p4_down_channel_2:
        p4_in=_(p4_in)
      for _ in self.p5_down_channel_2:
        p5_in=_(p5_in)
    p4_out = self.conv4_down(
            self._relu_fn(p4_in + p4_up + self.p4_downsample(p3_out)))
    p5_out = self.conv5_down(
            self._relu_fn(p5_in + p5_up + self.p5_downsample(p4_out)))
    p6_out = self.conv6_down(
            self._relu_fn(p6_in + p6_up + self.p6_downsample(p5_out)))
    p7_out = self.conv7_down(self._relu_fn(p7_in + self.p7_downsample(p6_out)))
    return p3_out, p4_out, p5_out, p6_out, p7_out
