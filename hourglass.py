import tensorflow as tf


class Hourglass():
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images
    Please check README.txt for further information on model management.
    """

    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=4, outputDim=16, batch_size=16, drop_rate=0.2, learning_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True, logdir_train=None, logdir_test=None, w_loss=False, name='tiny_hourglass',  joints=['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']):
        """ Initializer
        Args:
                nStack				: number of stacks (stage/Hourglass modules)
                nFeat				: number of feature channels on conv layers
                nLow				: number of downsampling (pooling) per module
                outputDim			: number of output Dimension (16 for MPII)
                batch_size			: size of training/testing Batch
                dro_rate			: Rate of neurons disabling for Dropout Layers
                lear_rate			: Learning Rate starting value
                decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
                decay_step			: Step to apply decay
                dataset			: Dataset (class DataGenerator)
                training			: (bool) True for training / False for prediction
                w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
                name				: name of the model
        """
        self.nStack = nStack
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.dropout_rate = drop_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = joints
        self.w_loss = w_loss

    def _conv(self, inputs, num_in, num_out, kernel_size, name_scope):
        with tf.name_scope(name_scope):
            filter = tf.Variable(
                initial_value=tf.glorot_uniform_initializer()(
                    shape=[kernel_size, kernel_size, num_in, num_out]
                ),
                name='filter'
            )
            norm = tf.layers.batch_normalization(
                inputs=inputs,
                momentum=0.9,
                epsilon=1e-5,
                is_training=self.training
                name='batch_norm'
            )
            relu = tf.nn.relu(
                input=norm,
                name='relu'
            )
            conv = tf.layers.conv2d(
                input=relu,
                filter=filter,
                strides=[1, 1, 1, 1],
                padding='VALID',
                data_format='NHWC',
                name='conv'
            )
        return conv

    def _conv_block(self, inputs, num_in, num_out, name='conv_block'):
		""" Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
        with tf.name_scope(name):
            inputs = self._conv(inputs, num_in, num_out // 2, '1')
            inputs = self._conv(inputs, num_out // 2, num_out // 2, 3, '2')
            block = self._conv(inputs, num_out // 2, num_out, 1, '3')
        return block
	
    def _skip_layer(self, inputs, num_in, num_out, name='skip_layer'):
        with tf.name_scope(name):
            if num_in == num_out:
                return tf.identity(inputs, name='identity')
            else:
                filter = tf.Variable(
                    initial_value=tf.glorot_uniform_initializer()(
                        shape=[kernel_size, kernel_size, num_in, num_out]
                    ),
                    name='filter'
                )
                return tf.layers.conv2d(
                    input=inputs,
                    filter=filter,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    data_format='NHWC',
                    name='conv'
                )

    def _residual(self, inputs, num_in, num_out, name='residual'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
            convb = self._conv_block(inputs, num_in, num_out)
            skipl = self._skip_layer(inputs, num_in, num_out)
            return tf.add_n([convb, skipl], name = 'add')
    
    def _stack_residual(self, inputs, nModule, num_in, num_out):
        for idx in xrange(nModules):
            inputs = self._residual(inputs, num_in, num_out, idx, 'residual' + str(idx))
        return inputs

    def _hourglass(self, inputs, n, f, name = 'hourglass'):
		""" Hourglass Module
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
            with tf.name_scope(str(n)):
                # Upper Branch
                up1 = inputs
                up1 = self._stack_residual(up1, self.nModule, f, f)
                        
                low1 = tf.nn.max_pool(
                    value=inputs, 
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='VALID'
                )

                low1 = self._stack_residual(low1, self.nModule, f, f)

                if n > 1:
                    low2 = self._hourglass(low1, n - 1, f)
                else:
                    low2 = self._stack_residual(low1, self.nModule, f, f)

                low3 = self._residual(low2, self.nFeat, self.nFeat, idx)
                up2 = tf.image.resize_nearest_neighbor(
                    low_3, 
                    tf.shape(low_3)[1:3] * 2,
                    name = 'upsampling'
                )
                return tf.add_n([up1, up2], name = 'out_hg')
