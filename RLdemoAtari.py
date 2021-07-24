import gym
import torch
import datetime
from pathlib import Path
from gym.wrappers import FrameStack
from RLdemoAtariWrappers import SkipFrame, GreyScaleObservation, ResizeObservation
from RLdemoAtariSpaceshipAgent import spaceInvadersLearner

env = gym.make('SpaceInvaders-v0')
env.reset()
next_state, reward, done, info = env.step(action=0)

# calculating shape of samples
sample_shape = 100
num_stack = 4
compressed_shape = (num_stack, sample_shape, sample_shape)

# changing scale and size of screen samples
env = SkipFrame(env, skip=num_stack)
env = GreyScaleObservation(env)
env = ResizeObservation(env, shape=sample_shape)
env = FrameStack(env, num_stack=num_stack)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

space_ship = spaceInvadersLearner(env=env, state_dim=compressed_shape, action_dim=env.action_space.n,
                                  save_dir=save_dir, expected_input_dim=compressed_shape)
space_ship.train()
