import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features


        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))


        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))


        bound = 1/np.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 / np.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma0 / np.sqrt(out_features))
        self.reset_noise()


    @staticmethod
    def _noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        x = self._noise(self.in_features, self.weight_mu.device)
        y = self._noise(self.out_features, self.in_features)
        self.weight_epsilon.copy(y.outer(x))
        self.bias_epsilon.copy_(y)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b  = self.weight_mu, self.bias_mu
        return F.linear(x,w,b)


class RainbowDQN(nn.Module):

    def __init__(self, state_size, action_size, num_atoms=51, hidden=256):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU()
        )

        #Value Stream
        self.value_hidden = NoisyLinear(hidden, 128)
        self.value_out = NoisyLinear(128, num_atoms)

        #Advantage Stream
        self.adv_hidden = NoisyLinear(hidden, 128)
        self.adv_out = NoisyLinear(128, action_size* num_atoms)

    def forward(self, x):
        x = self.features(x)
        value = self.value_out(F.relu(self.value_hidden(x))).view(
            -1, 1, self.num_atoms
        )
        advantage = self.adv_out(F.relu(self.adv_hidden(x))).view(
            -1, self.action_size, self.num_atoms
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for layer in (self.value_hidden, self.value_out,
                      self.adv_hidden, self.adv_out):
            layer.reset_noise()


class PrioritizedNStepReplay:

    """
    Combines two techniques:

    1. Prioritized Experience Replay (PER):
       Experiences with higher priorities are sampled more frequently.

    2. N-step Learning:
       Combines rewards from multiple consecutive steps
       into a single experience.
    """

    def __init__(self,capacity, batch_size, n_step, gamma, alpha=0.6, beta_start=0.4,beta_frames = 1_000_000):

        # Maximum number of experiences the replay buffer can store.
        self.capacity,self.batch_size =capacity, self.batch_size      # Number of experiences to sample for each training update.

        # Number of consecutive rewards to combine (e.g. 3-step learning).
        self.n_step = n_step

        # Discount factor: determines how much future rewards are valued.
        self.gamma = gamma

        # Controls how strongly priorities affect sampling probabilities.
        # alpha = 0 means uniform sampling.
        # Larger alpha means stronger prioritization.
        self.alpha = alpha

        # Initial strength of importance-sampling bias correction.
        self.beta_start = beta_start

        # Number of sampling steps over which beta increases towards 1.
        self.beta_frames = beta_frames

        # Main replay memory: stores completed N-step experiences.
        self.memory = []

        # Stores the priority associated with each experience.
        # Initially, every priority is zero.
        self.priorities = np.zeros(capacity, dtype=np.float32)

        # Position where the next experience will be inserted.
        # Used to implement a circular replay buffer.
        self.position = 0

        # Temporary queue that collects recent experiences
        # until enough steps are available for N-step aggregation.
        self.queue = deque()

        # Counts how many batches have been sampled.
        # Used to gradually increase beta.
        self.samples = 0

        # Small positive value to prevent zero priorities.
        self.eps = 1e-6

    def __len__(self):
        """
        Returns the number of experiences currently stored
        in the main replay memory.
        """
        return len(self.memory)

    def _aggregate(self,seq):
        """
        Combines consecutive experiences into one N-step experience.

        Input:
            seq = [experience_1, experience_2, experience_3]

        Output:
            (initial_state,
             initial_action,
             accumulated_reward,
             final_next_state,
             final_done,
             actual_number_of_steps)
        """

        # Accumulator for discounted rewards.
        reward_sum = 0.0

        # Initially assume the final next state and done flag
        # come from the first experience.
        #experience_1 =
            #Index	Contents
            # [0]	State
            # [1]	Action
            # [2]	Reward
            # [3]	Next state
            # [4]	Done
        #[0][3] --> experience_1 , next_state
        last_next, last_done = seq[0][3], seq[0][4]

        # Tracks how many steps were actually aggregated.
        actual_n = 0

        # Go through each experience in the sequence.
        for i, item in enumerate(seq):
            """
             Add the current reward after discounting it.
            
             First reward:  gamma^0 * reward
             Second reward: gamma^1 * reward
             Third reward:  gamma^2 * reward
            
             Formula:
             R = r0 + gamma*r1 + gamma^2*r2 + ...

             itemp[2] = reward or the following experience
            """
            reward_sum += (self.gamma ** i) *item[2]

            # Update the final next state and done flag.
            # After the loop, these will represent the last
            # transition included in this N-step experience.
            last_next, last_done = item[3], item[4]

            # Count the number of transitions processed.
            actual_n = i + 1

             # If the episode ended, do not include any more steps.
            if last_done:
                break

        # The combined experience starts from the FIRST
        # experience's state and action.
        s, a = seq[0][0], seq[0][1]

        # Return the newly constructed N-step experience.
        return (s.copy(), 
                a, 
                reward_sum, 
                last_next.cpu(), 
                last_done, 
                actual_n 
               )

    def store(self, item):
        """
        Stores a completed N-step experience in the main memory.

        Assigns the new experience the maximum existing priority
        so that it has a reasonable chance of being sampled.
        """

        # If memory is not empty, find the highest existing priority.
        # If memory is empty, give the first experience priority 1.0.
        
        p = (
            float(self.priorities[:len(self.memory)].max())
            if self.memory else 1.0
        )

        # Safety check: priority must be positive and finite.
        if not np.isfinite(p) or p <= 0:
            p = 1.0


        # If memory has available space, append the new experience.
        if len(self.memory) < self.capacity:
            self.memory.append(item)

        # If memory is full, replace the oldest experience
        # at the current circular-buffer position.
        else:
            self.memory[self.position] = item

        self.priorities[self.position] = p
        # Move the position pointer forward in a circular manner.
        self.position = (self.position + 1) % self.capacity

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the temporary queue.

        Once enough experiences are collected, they are aggregated
        into a single N-step experience and stored in the main memory.
        """

        # Append the new experience to the temporary queue.
        self.queue.append((
            np.asarray(state, dtype=np.float32).copy(), int(action),
            float(reward), np.asarray(next_state, dtype=np.float32).copy(),
            bool(done)
        ))

        # If enough experiences are collected for N-step aggregation:
        if len(self.queue) >= self.n_step:

            # Aggregate the first N experiences into one.
            