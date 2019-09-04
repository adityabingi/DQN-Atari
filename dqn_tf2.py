import tensorflow as tf
from tensorflow.keras import Input, layers, initializers
from config import Config

class DQN:

	def __init__(self, n_actions):

		self.n_actions = n_actions
		self.obs_shape= Config.atari_obs_shape
		self.gamma = Config.discounting_factor  # discounting factor used in computation of td taregt
		self.learning_rate = Config.learning_rate

		self.online_model = self._get_qfunc()
		self.target_model = self._get_qfunc()
		self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


	@tf.function
	def choose_action(self, obs, epsilon):

		q_values = self.online_model(obs)
		batch_size = (q_values.shape)[0]
		random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype= tf.float32)
		random_action = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.n_actions, dtype=tf.int64)
		best_action = tf.argmax(q_values, axis=1)
		action = tf.where(tf.less(random, epsilon), random_action, best_action)
		return action


	def _get_qfunc(self):

		inputs = Input(shape=self.obs_shape, dtype='uint8')
		cast_input = tf.dtypes.cast(inputs, tf.float32)
		input_scaled = cast_input/255.0

		conv1 = layers.Conv2D(filters=32,kernel_size=(8,8),strides=4,kernel_initializer=initializers.VarianceScaling(scale=2,distribution='untruncated_normal'),
		activation=tf.nn.relu)(input_scaled)

		conv2 = layers.Conv2D(filters=64,kernel_size=(4,4),strides=2,kernel_initializer=initializers.VarianceScaling(scale=2,distribution='untruncated_normal'),
		activation=tf.nn.relu)(conv1)

		conv3 = layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,kernel_initializer=initializers.VarianceScaling(scale=2,distribution='untruncated_normal'),
		activation=tf.nn.relu)(conv2)

		flatten = layers.Flatten()(conv3)
		dense = layers.Dense(units=512,kernel_initializer=initializers.VarianceScaling(scale=2,distribution='untruncated_normal'),
		activation=tf.nn.relu)(flatten)

		outputs = layers.Dense(units=self.n_actions)(dense)
		model = tf.keras.Model(inputs, outputs)

		return model

	@tf.function
	def compute_td_error(self, q_values, actions, rewards, target_q_values, dones):

		q_selected = tf.reduce_sum(tf.multiply(q_values,tf.one_hot(actions,self.n_actions)),1)
		target_q = tf.reduce_max(target_q_values,1)
		target_q = tf.multiply(tf.cast((1-dones),tf.float32), target_q)
		target_q_selected = tf.stop_gradient(tf.cast(rewards,tf.float32) + self.gamma * target_q)
		td_error = q_selected - target_q_selected

		return td_error

	@tf.function
	def train_step(self, obses, actions, rewards, obses_tp1, dones):

		with tf.GradientTape() as tape:

			q_values = self.online_model(obses)
			target_q_values = self.target_model(obses_tp1)
			td_error = self.compute_td_error(q_values, actions, rewards, target_q_values, dones)
			loss_value = self.huber_loss(td_error)

		params = self.online_model.trainable_variables
		grads = tape.gradient(loss_value, params)
		self.optimizer.apply_gradients(zip(grads,params))

	@tf.function
	def huber_loss(self, x, delta=1.0):
		loss = tf.where(
		tf.abs(x) < delta,
		tf.square(x) * 0.5,
		delta * (tf.abs(x)-(0.5*delta))
		)
		return loss

	@tf.function
	def update_target(self):

		for online_param, target_param in zip(self.online_model.trainable_variables, self.target_model.trainable_variables):
			target_param.assign(online_param)

	def layers_summary(self, log_step):

		for layer in self.online_model.layers:
			if(len(layer.weights)!=0):
				for weights in layer.weights:
					tf.summary.histogram(weights.name, weights, step=log_step)