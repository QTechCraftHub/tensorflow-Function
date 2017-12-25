### tf.nn.conv2d ###
函数原型：
<pre><code>
conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    name=None
)
</code></pre>

参数说明：
+ input: A Tensor. Must be one of the following types: half, float32. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
+ filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
+ strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. The dimension order is determined by the value of data_format, see below for details.
+ padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
+ use_cudnn_on_gpu: An optional bool. Defaults to True.
+ data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
+ name: A name for the operation (optional).

返回值：
+ A Tensor. Has the same type as input. A 4-D tensor. The dimension order is determined by the value of data_format, see below for details.
