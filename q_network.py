import tensorflow as tf



class Network(object):

    def __init__(self, sess, state_size, action_dim, learning_rate, device, layer_norm=True ):
        self.sess = sess
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.currentState = -1.
        self.device = device
        self.state_size = state_size
        self.layer_norm = layer_norm
        # Q network
        self.inputs, self.out, self.saver = self.create_q_network('q')
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_saver = self.create_q_network('q_target')

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.update_target_network_params = [self.target_network_params[i].assign(self.network_params[i]) for i in range(len(self.target_network_params))]

        with tf.variable_scope('learning_rate'): 
            # global step
            self.global_step = tf.Variable(0, trainable=False)
            self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 100000, 0.96, staircase=True)
            tf.summary.scalar('decay_learning_rate', self.decay_learning_rate)

        with tf.variable_scope('loss'): 
            with tf.device(self.device):

                # placeholders for training
                self.target_q_t = tf.placeholder(tf.float32, [None, 1], name='target_q')
                self.action = tf.placeholder(tf.int32, [None,1])
                # obtain the q scores of the selected action
                self.action_one_hot = tf.one_hot(self.action, self.a_dim, name='action_one_hot')
                self.q_acted_0 = self.out * tf.squeeze(self.action_one_hot)
                self.q_acted = tf.reduce_sum(self.q_acted_0, reduction_indices=1, name='q_acted')
                
                # calculate the loss
                self.delta = tf.subtract(tf.squeeze(self.target_q_t), self.q_acted)
                self.loss = tf.reduce_mean(self.clipped_error(self.delta), name='loss')
                # optimize, with gradient clipping
                self.optimizer = tf.train.AdamOptimizer(self.decay_learning_rate)
                gradients = self.optimizer.compute_gradients(self.loss)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        gradients[i] = (tf.clip_by_norm(grad, 5.), var)
                self.optimize = self.optimizer.apply_gradients(gradients, global_step=self.global_step)
                # or just optimize
                #self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        tf.summary.scalar('huber_loss', self.loss)
        self.merged = tf.summary.merge_all()
        

    def create_q_network(self, scope):

        with tf.device(self.device):
            with tf.variable_scope(scope): 
                
                stateInput = tf.placeholder(tf.float32, shape=[None,self.state_size])
                # fully connected layer
                # stateInput = tf.to_float(stateInput)
                fc1 = tf.contrib.layers.fully_connected(stateInput, 100, activation_fn=None)
                if self.layer_norm:
                    fc1 = tf.contrib.layers.layer_norm(fc1, center=True, scale=True)
                fc1 = tf.nn.relu(fc1)
                fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=None)
                # normalization
                if self.layer_norm:
                    fc2 = tf.contrib.layers.layer_norm(fc2, center=True, scale=True)
                fc2 = tf.nn.relu(fc2)
                out = tf.contrib.layers.fully_connected(fc2, self.a_dim,activation_fn=None)
                
               
        saver = tf.train.Saver()
        return stateInput, out, saver

    def train(self, actions, target_q_t, inputs):
        with tf.device(self.device):
            return self.sess.run([self.merged, self.loss, self.optimize], feed_dict={
                self.action: actions,
                self.target_q_t: target_q_t,
                self.inputs: inputs
            })

    def train_v2(self, actions, target_q_t, inputs):
        with tf.device(self.device):
            return self.sess.run([self.action, self.action_one_hot, self.out, self.target_q_t, self.q_acted_0, self.q_acted, self.delta,  self.loss, self.optimize], feed_dict={
                self.action: actions,
                self.target_q_t: target_q_t,
                self.inputs: inputs
            })

    def predict(self, inputs):
        with tf.device(self.device):
            return self.sess.run(self.out, feed_dict={
                self.inputs: inputs
            })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    
    def update_target_network(self):
        with tf.device(self.device):
            self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save(self):
        self.saver.save(self.sess,'./model.ckpt')
        self.target_saver.save(self.sess,'./model_target.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file: model")

    
    def recover(self):
        self.saver.restore(self.sess,'./model.ckpt')
        self.target_saver.restore(self.sess,'./model_target.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    
    def conv2d(self,x, W, stride):
        with tf.device(self.device):
            return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def clipped_error(self,x):
      # Huber loss
      try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
      except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)