import gym
import numpy as np
import argparse

from replay_buffer import ReplayBuffer
from atari_wrappers import make_atari, wrap_deepmind
from dqn_tf2 import DQN
from config import Config

from gym.wrappers import monitor

import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())


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

def train(agent, env):

	train_writer = tf.summary.create_file_writer(summariesDir+'train')
	test_writer = tf.summary.create_file_writer(summariesDir+'test')
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=agent.optimizer, model=agent.online_model)
	manager = tf.train.CheckpointManager(ckpt, modelDir, max_to_keep=3)

	agent.update_target()
	replaybuffer = ReplayBuffer(size=replay_size)

	train_env = wrap_deepmind(env, frame_stack=True)
	obs = train_env.reset()

	episode_rewards = [0.0]
	max_mean_reward = None
	
	for t in range(total_time_steps):

		epsilon = LinearSchedule(t)
		action = agent.choose_action(obs = np.array(obs)[None], epsilon = epsilon)
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
			agent.train_step(obses_t, actions, rewards, obses_tp1, dones)

		if (t> initial_steps and t % update_freq==0 ):
			print("Updating target network at timestep {}".format(t))
			agent.update_target()

		num_episodes = len(episode_rewards)
		if num_episodes>100: 
			mean_reward = np.mean(episode_rewards[-101:-1])

		if(num_episodes>100 and t>initial_steps and t%print_freq==0):
			if max_mean_reward is None or mean_reward > max_mean_reward:
				max_mean_reward = mean_reward
				print("Improvement in mean_100ep_reward: {}".format(max_mean_reward))
			else:
				print("No improvement observed in mean_100ep_reward.Best mean_100ep_reward:{}".format(max_mean_reward))

		if (num_episodes>100 and t > initial_steps and t % eval_freq==0):
			eval_mean_reward = evaluate(agent, env)
			ckpt.step.assign_add(1)	
			save_path = manager.save()
			print("Evaluation reward at {} step is {}".format(t, eval_mean_reward))
			with test_writer.as_default():
				tf.summary.scalar('eval_reward', mean_episdoe_reward, step=t)

		if(num_episodes>100 and t> initial_steps and t % (train_freq*100) ==0):

			with train_writer.as_default():
				tf.summary.scalar('mean_100ep_reward', mean_reward, step=t)

			with train_writer.as_default():
				agent.layers_summary(t)

def evaluate(agent, env, eval_episodes=eval_episodes, restore=False, play=False):

	if(restore):
		ckpt = tf.train.checkpoint(model=agent.online_model())
		latestSnapshot= tf.train.latest_checkpoint(modelDir)
		if not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		ckpt.restore(latestSnapshot)
		print("Restored saved model from latest snapshot")


	eval_env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, evaluate=True)
	obs = eval_env.reset()
	
	eval_episode_rewards=[0.0]

	while(True):

		action = agent.choose_action(obs =np.array(obs)[None,:], epsilon=evaluation_ep)
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

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train an agent to find optimal policy', action='store_true')
	parser.add_argument('--evaluate', help='evaluate trained policy of an agent', action='store_true')
	parser.add_argument('--play', help='let trained agent play', action='store_true')
	parser.add_argument('--env', nargs=1, help='env used for DQN', type=str)

	args = parser.parse_args()

	env_id = args.env[0] + 'NoFrameskip-v4'
	env = make_atari(env_id)
	n_actions = env.action_space.n
	agent = DQN(n_actions)


	if(args.train):
		train(agent, env)

	if(args.evaluate):
		test_env = gym.wrappers.Monitor(env, saveVideoDir+'testing', force=True)
		evaluate(test_env, eval_steps=evaluation_steps)
		test_env.close()

	if(args.play):
		play_env = gym.wrappers.Monitor(env, saveVideoDir+'play', force=True)
		evaluate(play_env)
		play_env.close()	

	env.close()


if __name__ == '__main__':
	main()