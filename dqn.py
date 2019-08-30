import tensorflow as tf
from config import Config

class DQN(object):

	def __init__(self, n_actions):

		self.n_actions = n_actions
		self.obs_shape= Config.atari_obs_shape
		self.gamma = Config.discounting_factor  # discounting factor used in computation of td taregt
		self.learning_rate = Config.learning_rate

		self.choose_action = self._exploration()
		self.optimize_op, self.loss, self.gradients_summary  = self._optimize()
		self.update_target  = self._update_target()
		self.parameters_summary = self._parameters_summary()
		self.performance_summary, self.eval_summary = self._performance_summary()

		self.init = tf.global_variables_initializer()

	def _build_q_func(self, input_placeholder, scope, reuse=None):

		with tf.variable_scope(scope, reuse=reuse):

			inputs = input_placeholder/255.0

			conv1 = tf.layers.conv2d(inputs,filters = 32,kernel_size=(8,8),strides=(4, 4),bias_initializer=tf.constant_initializer(0.0),
								kernel_initializer=tf.initializers.he_normal(seed=None),activation=tf.nn.relu)
			conv2 = tf.layers.conv2d(conv1,filters = 64,kernel_size =(4,4),strides=(2, 2),bias_initializer=tf.constant_initializer(0.0),
								kernel_initializer=tf.initializers.he_normal(seed=None),activation=tf.nn.relu)
			conv3 = tf.layers.conv2d(conv2,filters = 64,kernel_size=(3,3),strides=(1, 1),bias_initializer=tf.constant_initializer(0.0),
								kernel_initializer=tf.initializers.he_normal(seed=None),activation=tf.nn.relu)

			flatten = tf.layers.flatten(conv3)
			dense1 = tf.layers.dense(flatten,units=512,bias_initializer=tf.constant_initializer(0.0),
								kernel_initializer=tf.initializers.he_normal(seed=None),activation=tf.nn.relu)
			#dense2 = tf.layers.dense(dense1,units=256,bias_initializer=tf.constant_initializer(0.0),
								#kernel_initializer=tf.initializers.he_normal(seed=None),activation=tf.nn.relu)

			outputs = tf.layers.dense(dense1, units=self.n_actions)

		return outputs

	def _exploration(self):

		with tf.name_scope("Exploration"):

			self.obs_input_ph = tf.placeholder(tf.float32, shape=(None,)+self.obs_shape)
			self.epsilon_ph = tf.placeholder(tf.float32, shape=None)
			q_values = self._build_q_func(self.obs_input_ph, 'online_q_func')
			batch_size = tf.shape(q_values)[0]
			random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype= tf.float32)
			random_action = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.n_actions, dtype=tf.int64)
			best_action = tf.argmax(q_values, axis=1)
			action = tf.where(tf.less(random, self.epsilon_ph), random_action, best_action)

		return action

	def _parameters(self):

		with tf.name_scope('Parameters'):

			online_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="online_q_func")
			target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_q_func")

		return online_q_vars, target_q_vars

	def _optimize(self):

		with tf.name_scope("Optimization"):

			self.obses_t_ph  = tf.placeholder(tf.float32, shape=(None,)+ self.obs_shape)
			self.actions_ph  = tf.placeholder(tf.int32, shape=[None])
			self.rewards_ph  = tf.placeholder(tf.float32, shape=[None])
			self.obses_tp1_ph= tf.placeholder(tf.float32, shape=(None,) + self.obs_shape)
			self.dones_ph = tf.placeholder(tf.float32, shape=[None])

			q_values = self._build_q_func(self.obses_t_ph, 'online_q_func', reuse=True)
			target_q_values = self._build_q_func(self.obses_tp1_ph, 'target_q_func')

			q_selected = tf.reduce_sum(tf.multiply(q_values, tf.one_hot(self.actions_ph, self.n_actions)), 1)
			target_q = tf.reduce_max(target_q_values, axis=-1)
			target_q = (1.0 - self.dones_ph) * target_q
			target_q_selected = self.rewards_ph + self.gamma * target_q

			td_error = q_selected - tf.stop_gradient(target_q_selected)
			loss = tf.reduce_sum(self._huber_loss(td_error))

			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

			online_q_vars, _ = self._parameters()

			grads = optimizer.compute_gradients(loss, var_list=online_q_vars)

			grad_summaries =[]
			for grad, var in grads:
				grad_summ = tf.summary.histogram('{}_summary'.format(var.name), grad)
				grad_summaries.extend([grad_summ])

			grads_summary = tf.summary.merge(grad_summaries)

			optimize_op = optimizer.apply_gradients(grads)

		return optimize_op, loss, grads_summary

	def _update_target(self):

		with tf.name_scope("Update_Target"):

			online_q_vars, target_q_vars = self._parameters()

			updates = []

			for var, var_target in zip(online_q_vars, target_q_vars):
				update_op = var_target.assign(var)
				updates.append(update_op)

			update_target_op = tf.group(*updates)

		return update_target_op

	def _huber_loss(self, x, delta=1.0):

		with tf.name_scope("Huber_Loss"):
			loss = tf.where(
				tf.abs(x) < delta,
				tf.square(x) * 0.5,
				delta * (tf.abs(x)-(0.5*delta))
				)
		return loss

	def _parameters_summary(self):

		with tf.name_scope('Parameters_Summary'):

			online_q_vars, _ = self._parameters()

			param_summaries=[]

			for var_online in online_q_vars[:10]:
				online_param = tf.summary.histogram('{}_summary'.format(var_online.name), var_online)
				param_summaries.extend([online_param])

			parameters_summary = tf.summary.merge(param_summaries)

		return parameters_summary

	def _performance_summary(self):

		with tf.name_scope('Performance_Summary'):
			self.loss_ph = tf.placeholder(tf.float32, shape=None)
			loss_summary = tf.summary.scalar('loss', self.loss_ph)
			self.episode_reward_ph = tf.placeholder(tf.float32, shape=None)
			reward_summary = tf.summary.scalar('mean_100episode_reward', self.episode_reward_ph)
			self.eval_reward_ph = tf.placeholder(tf.float32, shape=None)
			eval_reward_summary = tf.summary.scalar('eval_episode_reward', self.eval_reward_ph)

			performance_summary=tf.summary.merge([loss_summary, reward_summary])

		return performance_summary, eval_reward_summary