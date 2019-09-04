
class Config:

	"Config class for Hyperparameters used for Atari games"

	replay_memory_start_size = 50000
	replay_memory_size = 1000000

	initial_ep = 1.0
	evaluation_ep = 0.05
	play_ep = 0.0

	first_final_ep = 0.1
	second_final_ep = 0.01
	first_schedule_max = 1050000
	second_schedule_max = 25000000

	total_time_steps = 30000000

	modelDir = "model/"
	summariesDir = "summaries/run1/"
	saveVideoDir = "savedvideos/"

	batch_size = 32
	discounting_factor = 0.99
	learning_rate = 0.00025 

	evaluation_frequency = 200000
	update_target_frequency = 10000
	train_frequency = 4
	print_frequency = 1000
	
	evaluation_episodes = 100
	atari_obs_shape = (84,84,4) # Last four frames stacked 84 x 84 grayscale images
