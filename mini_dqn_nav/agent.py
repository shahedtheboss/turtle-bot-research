import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_size)
        )

    def forward(self, x):
        return self.network(x)
    

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        # Shared feature extractor:
        # learns useful representations of the robot's state.
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value stream: estimates V(s), one value per state.
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream: estimates A(s, a), one value per action.
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        # state: [batch_size, state_size]
        features = self.feature_layer(state)

        # Value: [batch_size, 1]
        value = self.value_stream(features)

        # Advantage: [batch_size, action_size]
        advantage = self.advantage_stream(features)

        # Center advantages so their mean is zero per state.
        # This makes the value/advantage decomposition identifiable.
        advantage_mean = advantage.mean(dim=1, keepdim=True)

        # Broadcast value across actions and combine both streams.
        q_values = value + (advantage - advantage_mean)

        # Output: [batch_size, action_size]
        return q_values


class DQNAgent():
    def __init__(self, state_size=52, action_size=7):
        self.learn_steps = 0
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )

        #For DQN and DDQN
        # self.policy_net = QNetwork(state_size, action_size).to(self.device)
        # self.target_net = QNetwork(state_size, action_size).to(self.device)

        #Dueling DQN
        self.policy_net = DuelingQNetwork(
            state_size, action_size
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            state_size, action_size
        ).to(self.device)

       
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=50000)

        self.gamma = 0.99
        self.batch_size = 128

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def select_greedy_action(self, state):
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    """
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.asarray(states), dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.asarray(next_states), dtype=torch.float32,
            device=self.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # STEP 1: The POLICY network chooses the best next action.
            # Shape: [batch_size, action_size] -> [batch_size]
            next_actions = self.policy_net(next_states).argmax(
                dim=1,
                keepdim=True
            )

            # STEP 2: The TARGET network evaluates those chosen actions.
            # Shape: [batch_size, action_size] -> [batch_size, 1]
            next_q = self.target_net(next_states).gather(
                dim=1,
                index=next_actions
            ).squeeze(1)

            # STEP 3: Calculate the Double DQN Bellman target.
            # Terminal transitions receive no future Q-value.
            target_q = rewards + self.gamma * next_q * (1 - dones)


        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())
    """


    def learn(self):
          """
          PURPOSE:
              Train the policy network using a random batch of past
              experiences stored in the replay memory.
      
          DOUBLE DQN IDEA:
              The policy network selects the best next action.
              The target network evaluates that action.
              The policy network is updated to reduce the difference
              between its predicted Q-values and the calculated targets.
          """
      
          """
          SECTION 1: CHECK REPLAY MEMORY
      
          self.memory stores past experiences in the form:
              (state, action, reward, next_state, done)
      
          self.batch_size is the number of experiences needed for
          one training update.
      
          We return early if there aren't enough experiences.
          This avoids trying to sample a batch larger than the
          number of experiences currently available.
          """
        #   if len(self.memory) < self.batch_size:
        #       return
          if len(self.memory) < max(self.batch_size, 2000):   # warmup: no learning until 2000 transitions
              return
      
          """
          SECTION 2: SAMPLE A MINI-BATCH
      
          random.sample() selects batch_size experiences randomly
          without replacement.
      
          The experiences remain in memory after sampling and may
          be selected again during future learning updates.
      
          zip(*batch) reorganizes the experiences by their fields.
      
          For example:
              states      -> all sampled states
              actions     -> all sampled actions
              rewards     -> all sampled rewards
              next_states -> all sampled next states
              dones       -> all sampled terminal flags
      
          Random sampling helps reduce the correlation between
          consecutive experiences used in a single update.
          """
          batch = random.sample(self.memory, self.batch_size)
          states, actions, rewards, next_states, dones = zip(*batch)
      
          """
          SECTION 3: CONVERT DATA INTO PYTORCH TENSORS
      
          Neural networks operate on tensors rather than ordinary
          Python lists of experiences.
      
          states:
              Converts the states into float32 values.
              For batch_size=128 and state_size=52:
                  shape = [128, 52]
      
          actions:
              Contains the discrete action indices, such as 0 to 6.
              torch.long is required for indexing with gather().
              unsqueeze(1) changes its shape from [128] to [128, 1].
      
          rewards:
              Contains one reward per experience.
              Shape = [batch_size].
      
          next_states:
              Contains the observations after each action.
              With 22 state features:
                  shape = [batch_size, 22]
      
          dones:
              Contains episode-ending flags:
                  False -> 0.0
                  True  -> 1.0
      
          device=self.device places each tensor on the selected
          device, such as the CPU or CUDA-enabled GPU.
          """
          states = torch.tensor(
              np.asarray(states), dtype=torch.float32,
              device=self.device
          )
      
          actions = torch.tensor(
              actions, dtype=torch.long,
              device=self.device
          ).unsqueeze(1)
      
          rewards = torch.tensor(
              rewards, dtype=torch.float32,
              device=self.device
          )
      
          next_states = torch.tensor(
              np.asarray(next_states), dtype=torch.float32,
              device=self.device
          )
      
          dones = torch.tensor(
              dones, dtype=torch.float32,
              device=self.device
          )
      
          """
          SECTION 4: GET THE CURRENT Q-VALUES
      
          self.policy_net(states) predicts a Q-value for every
          available action in each current state.
      
          With 128 experiences and 7 actions:
              output shape = [128, 7]
      
          gather(1, actions) selects the Q-value for the action
          actually taken in each stored experience.
      
          squeeze(1) removes the extra dimension:
              [128, 1] -> [128]
      
          current_q contains the predictions we want to improve.
          """
          current_q = self.policy_net(states).gather(
              1, actions
          ).squeeze(1)
      
          """
          SECTION 5: CALCULATE THE LEARNING TARGETS
      
          We use torch.no_grad() because the target values are
          treated as fixed reference values during this update.
      
          This avoids calculating gradients through the target
          calculation and reduces unnecessary memory usage.
          """
          with torch.no_grad():
      
              """
              STEP 1: SELECT THE NEXT ACTION USING THE POLICY NETWORK
      
              The policy network predicts Q-values for every action
              in each next state.
      
              argmax(dim=1) selects the action index with the
              largest predicted Q-value for each state.
      
              keepdim=True preserves the action tensor's second
              dimension, giving it shape [batch_size, 1].
      
              This is the action-selection part of Double DQN.
              """
              next_actions = self.policy_net(next_states).argmax(
                  dim=1,
                  keepdim=True
              )
      
              """
              STEP 2: EVALUATE THE SELECTED ACTION USING THE TARGET NETWORK
      
              The target network predicts Q-values for every action
              in each next state.
      
              gather() selects the value corresponding to the action
              selected by the policy network in Step 1.
      
              Shape:
                  Target network output: [batch_size, action_size]
                  After gather:           [batch_size, 1]
                  After squeeze(1):       [batch_size]
      
              Double DQN separates action selection from action
              evaluation, helping reduce Q-value overestimation.
              """
              next_q = self.target_net(next_states).gather(
                  dim=1,
                  index=next_actions
              ).squeeze(1)
      
              """
              STEP 3: CALCULATE THE DOUBLE DQN BELLMAN TARGET
      
              Formula:
      
                  target_q = reward + gamma * next_q * (1 - done)
      
              gamma (self.gamma) is the discount factor controlling
              how much future rewards contribute to the target.
      
              If done = 0:
                  target_q = reward + gamma * next_q
      
              If done = 1:
                  target_q = reward
      
              A terminal transition receives no future Q-value
              contribution because the episode has ended.
      
              All tensors in this expression have shape [batch_size].
              """
              target_q = rewards + self.gamma * next_q * (1 - dones)
        
          """
          SECTION 6: CALCULATE THE LOSS
      
          self.loss_fn compares the current predictions with the
          Bellman targets.
      
          The exact loss calculation depends on how loss_fn was
          initialized, for example MSELoss or SmoothL1Loss.
      
          A large loss indicates a large prediction-target
          discrepancy under that loss function. It does not,
          by itself, prove that navigation performance is poor.
          """
          loss = self.loss_fn(current_q, target_q)
      
          """
          SECTION 7: UPDATE THE POLICY NETWORK
      
          zero_grad():
              Clears gradients left over from the previous update.
              PyTorch accumulates gradients unless they are cleared.
      
          backward():
              Calculates gradients of the loss with respect to
              the trainable policy-network parameters.
      
          clip_grad_norm_():
              Limits the global gradient norm to 10.
              This can help prevent excessively large gradient
              updates, although it does not guarantee stability.
      
          optimizer.step():
              Updates the policy-network parameters using the
              calculated gradients.
      
          IMPORTANT:
              The target network is not updated here.
              It is synchronized separately by update_target().
          """
          self.optimizer.zero_grad()
      
          loss.backward()
      
          torch.nn.utils.clip_grad_norm_(
              self.policy_net.parameters(),
              max_norm=10.0
          )
      
          self.optimizer.step()
      
          """
          SECTION 8: RETURN THE LOSS
      
          loss.item() extracts the scalar value from the loss tensor.
          float() returns it as a standard Python float.
      
          The training loop can record this value and calculate
          the mean loss across the optimization steps in an episode.
      
          If there weren't enough replay experiences, the function
          returned None at the beginning and performed no update.
          """
          self.learn_steps += 1                 # count optimization steps
          if self.learn_steps % 500 == 0:       # hard target update every 500 steps
              self.update_target()

          return float(loss.item())


    ''''''
    def update_target(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
    ''''''

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * self.epsilon_decay
        )


    
    def save(self, path):
        """Save enough information to resume training later."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)


    def load(self, path):
        """Restore a previously saved checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]



#Test Dueling net

# state_size = 52  # Replace with your actual observation size
# action_size = 7

# model = DuelingQNetwork(state_size, action_size)

# dummy_states = torch.randn(4, state_size)
# q_values = model(dummy_states)

# print("Input shape:", dummy_states.shape)
# print("Output shape:", q_values.shape)
# print("Q-values:", q_values)

# assert q_values.shape == (4, action_size)




        