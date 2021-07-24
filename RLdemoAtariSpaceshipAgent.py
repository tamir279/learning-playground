import torch
from torch import nn
import numpy as np
from time import sleep
from collections import deque
import random
import copy

class AgentNet(nn.Module):
    def __init__(self, input_dim, output_dim, expected_input_dim):
        super().__init__()
        c, h, w = input_dim
        if h != expected_input_dim[1]:
            raise ValueError(f"Expecting input height: {expected_input_dim[0]}, got: {h}")
        if w != expected_input_dim[2]:
            raise ValueError(f"Expecting input width: {expected_input_dim[1]}, got: {w}")
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*9*9, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, inputs, model):
        if model == "online":
            return self.online(inputs)
        elif model == "target":
            return self.target(inputs)

class spaceShipAct:
    def __init__(self, state_dim, action_dim, save_dir, expected_input_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expected_input_dim = expected_input_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = AgentNet(self.state_dim, self.action_dim, self.expected_input_dim).float()

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5

    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

class spaceShipMemory(spaceShipAct):
    def __init__(self, state_dim, action_dim, save_dir, expected_input_dim):
        super().__init__(state_dim, action_dim, save_dir, expected_input_dim)
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

class spaceShipTD(spaceShipMemory):
    def __init__(self, state_dim, action_dim, save_dir, expected_input_dim):
        super().__init__(state_dim, action_dim, save_dir, expected_input_dim)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

class spaceShipQUpdate(spaceShipTD):
    def __init__(self, state_dim, action_dim, save_dir, expected_input_dim):
        super().__init__(state_dim, action_dim, save_dir, expected_input_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        # self.loss_fn = torch.nn.MSELoss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

class spaceShipSave(spaceShipQUpdate):
    def save(self):
        save_path = (
                self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

class spaceShipAgent(spaceShipSave):
    def __init__(self, state_dim, action_dim, save_dir, expected_input_dim):
        super().__init__(state_dim, action_dim, save_dir, expected_input_dim)
        self.burnin = 5*1e2  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e3  # no. of experiences between Q_target & Q_online sync

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

class spaceInvadersLearner(spaceShipAgent):
    def __init__(self, env, state_dim, action_dim, save_dir, expected_input_dim):
        super().__init__(state_dim, action_dim, save_dir, expected_input_dim)
        self.episodes = 15
        self.env = env

    def train(self):
        for e in range(self.episodes):
            state = self.env.reset()
            # plot average loss
            avgLoss, numSteps = 0, 0
            # Play the game!
            while True:
                # render the game
                self.env.render()
                sleep(0.0416)
                # Run agent on the state
                action = self.act(state)
                # Agent performs action
                next_state, reward, done, info = self.env.step(action)
                # Remember
                self.cache(state, next_state, action, reward, done)
                # Learn
                q, loss = self.learn()
                # calculate accumulated loss
                if loss is not None:
                    avgLoss += loss
                    print(f"EXPLOITING-learning from sampled examples -- LOSS: {loss}")
                else:
                    avgLoss += 0
                    print(f"EXPLORING-saving step to cache")
                # Update state
                state = next_state
                # update step number
                numSteps += 1
                # Check if end of game
                if done:
                    break
            print(f"-------average LOSS: {avgLoss / numSteps}-------")
        self.env.close()
