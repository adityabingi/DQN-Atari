import gym
import tensorflow as tf
import numpy as np
import argparse

from replay_buffer import ReplayBuffer
from atari_wrappers import make_atari, wrap_deepmind
from dqn import DQN
from config import Config

from gym.wrappers import monitor


modelDir            = Config.modelDir
summariesDir        = Config.summariesDir
saveVideoDir        = Config.saveVideoDir
initial_ep          = Config.initial_ep
first_final_ep      = Config.first_final_ep
second_final_ep     = Config.second_final_ep
first_schedule_max  = Config.first_schedule_max
second_schedule_max = Config.second_schedule_max
evaluation_ep       = Config.evaluation_ep
play_ep             = Config.play_ep
replay_size         = Config.replay_memory_size
initial_steps       = Config.replay_memory_start_size
total_time_steps    = Config.total_time_steps
batch_size          = Config.batch_size
train_freq          = Config.train_frequency
update_target_freq  = Config.update_target_frequency
eval_freq           = Config.evaluation_frequency
print_freq          = Config.print_frequency
eval_episodes       = Config.evaluation_episodes


def LinearSchedule(timestep):

	if(timestep<initial_steps):
		return initial_ep

	if(timestep>initial_steps and timestep<first_schedule_max):
		fraction = float((timestep-initial_steps)/(first_schedule_max-initial_steps))
		epsilon = initial_ep + fraction * (first_final_ep - initial_ep)
		return epsilon

	fraction = min(float((timestep-first_schedule_max)/(second_schedule_max-first_schedule_max)), 1.0)
	epsilon =  first_final_ep + fraction *(second_final_ep-first_final_ep)

	return epsilon

def train(agent, env, sess):

	saver = tf.compat.v1.train.Saver(max_to_keep=3)
	train_env = wrap_deepmind(env, frame_stack=True)
	sess.run(agent.init)
	print("Initialized with new values")

	sess.run(agent.update_target)

	train_writer = tf.compat.v1.summary.FileWriter(summariesDir+'train', sess.graph)
	test_writer = tf.compat.v1.summary.FileWriter(summariesDir+'eval')

	replaybuffer = ReplayBuffer(size=replay_size)
	obs = train_env.reset()

	episode_rewards = [0.0]
	max_mean_reward = None
	snapshotID = 0

	for t in range(total_time_steps):

		epsilon = LinearSchedule(t)
		action = sess.run(agent.choose_action, feed_dict = {agent.obs_input_ph:np.array(obs)[None],agent.epsilon_ph:epsilon})
		next_obs, reward, done, info = train_env.step(action)
		replaybuffer.add(obs, action, reward, next_obs, float(done))

		episode_rewards[-1] += reward
		obs = next_obs

		if done:
			obs = train_env.reset()
			episode_rewards.append(0.0)

		if (t > initial_steps and t % train_freq==0):
			obses_t, actions, rewards, obses_tp1, dones = replaybuffer.sample(batch_size)
			actions = np.squeeze(actions, axis=1)
			_, loss_val, grad_summary = sess.run([agent.optimize_op, agent.loss, agent.gradients_summary], feed_dict={agent.obses_t_ph:obses_t,agent.actions_ph:actions,
				agent.rewards_ph:rewards, agent.obses_tp1_ph:obses_tp1, agent.dones_ph:dones})

		if (t> initial_steps and t % update_target_freq==0 ):
			print("Updating target network at timestep {}".format(t))
			sess.run(agent.update_target)

		num_episodes = len(episode_rewards)
		if num_episodes>100: 
			mean_reward = np.mean(episode_rewards[-101:-1])

		if(num_episodes>100 and t>initial_steps and t%print_freq==0):
			if max_mean_reward is None or mean_reward > max_mean_reward:
				max_mean_reward = mean_reward
				print("Improvement in mean_100ep_reward: {}".format(max_mean_reward))
			else:
				print("No improvement observed in mean_100ep_reward.Best mean_100ep_reward:{}".format(max_mean_reward))

		if (t > initial_steps and t % eval_freq==0):
			eval_mean_reward = evaluate(agent, env, sess, saver)
			snapshotID += 1
			saver.save(sess, modelDir+'snapshot', global_step = snapshotID)
			print("Evaluation reward at {} step is {}".format(t, eval_mean_reward))
			evaluation_summary = sess.run(agent.eval_summary, feed_dict={agent.eval_reward_ph:eval_mean_reward})
			test_writer.add_summary(evaluation_summary, t)		

		if(t> initial_steps and t % (train_freq*100) ==0):

			param_summary = sess.run(agent.parameters_summary)
			perf_summary = sess.run(agent.performance_summary, feed_dict={agent.loss_ph:loss_val, agent.episode_reward_ph:mean_reward})

			train_writer.add_summary(param_summary, t)
			train_writer.add_summary(grad_summary, t)
			train_writer.add_summary(perf_summary,  t)


def evaluate(agent, env, sess, restore=False, eval_episodes=eval_episodes, play=False):

	if restore:
		saver = tf.compat.v1.train.Saver()
		latestSnapshot = tf.train.latest_checkpoint(modelDir)
		if not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		saver.restore(sess, latestSnapshot)
		print("Restored saved model from latest snapshot")

	eval_env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, evaluate=True)

	obs = eval_env.reset()	
	eval_episode_rewards=[0.0]

	while (True):

		action = sess.run(agent.choose_action, feed_dict={agent.obs_input_ph:np.array(obs)[None,:], agent.epsilon_ph:evaluation_ep})
		next_obs, reward, done, info = eval_env.step(action)
		eval_episode_rewards[-1] += reward
		obs = next_obs

		eval_mean_reward = np.mean(eval_episode_rewards)	

		if done:
			obs = eval_env.reset()
			no_of_episodes = len(eval_episode_rewards)
			if(restore):
				print("Mean reward after {} episodes is {}".format(no_of_episodes, round(eval_mean_reward, 2)))
			if(play):
				break
			if(no_of_episodes>=eval_episodes):
				break
			eval_episode_rewards.append(0.0)

	return round(eval_mean_reward, 2)

def make_session():

	config = tf.compat.v1.ConfigProto(allow_soft_placement=True) #log_device_placement=True)
	config.gpu_options.allow_growth = True
	sess  = tf.compat.v1.Session(config=config)
	return sess

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train an agent to find optimal policy', action='store_true')
	parser.add_argument('--evaluate', help='evaluate trained policy of an agent', action='store_true')
	parser.add_argument('--play', help='let trained agent play', action='store_true')
	parser.add_argument('--env', nargs=1, help='env used for DQN', type=str)

	args = parser.parse_args()

	env_id = args.env[0] + 'NoFrameskip-v0'
	env = make_atari(env_id)

	n_actions = env.action_space.n
	agent = DQN(n_actions)
	
	sess = make_session()

	if(args.train):
		train(agent, env, sess)

	if(args.evaluate):

		test_env = gym.wrappers.Monitor(env, saveVideoDir+'testing', force=True)
		evaluate(agent, test_env, sess, restore=True)
		test_env.close()

	if(args.play):
		play_env = gym.wrappers.Monitor(env, saveVideoDir+'play', force=True)
		evaluate(agent, play_env, sess, restore=True, play=True)
		play_env.close()	

	env.close()


if __name__ == '__main__':
	main()