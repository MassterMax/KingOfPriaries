import os

import numpy
import torch
import torch.nn as nn
import numpy as np

# from callCS.mini_game import SIZE


class Agent:

    def __init__(self, n_frames: int, n_features: int, n_actions: int, epsilon: float, delta_epsilon: float,
                 min_epsilon: float, gamma: float, lr: float, model_path: str, layers):
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.n_features = n_features
        self.model_path = model_path
        self.layers = layers
        self.__make_nn()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.delta_epsilon = delta_epsilon
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def __make_nn(self):
        if os.path.exists(self.model_path):
            network = torch.load(self.model_path)
            # network.load_state_dict(torch.load(self.model_path))
            # network = torch.load(os.path.abspath(self.model_path))
        else:
            # img_height = SIZE
            network = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),

                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, self.n_actions),
            )

            # network = nn.Sequential()
            # for i in range(len(self.layers) - 1):
            #    network.add_module('l' + str(i), nn.Linear(self.layers[i], self.layers[i + 1]))
            #    network.add_module('relu' + str(i), nn.ReLU())
            #
            # network.add_module('last_l', nn.Linear(self.layers[-1], self.n_actions))
        self.network = network

    def __compute_td_loss(self, states, actions, rewards, next_states, is_done):
        states = torch.tensor(numpy.array(states), dtype=torch.float32)  # shape: [batch_size, state_size]
        actions = torch.tensor(actions, dtype=torch.long)  # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
        next_states = torch.tensor(numpy.array(next_states), dtype=torch.float32)  # shape: [batch_size, state_size]
        is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

        predicted_qvalues = self.network(states)
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]
        predicted_next_qvalues = self.network(next_states)

        next_state_values = torch.max(predicted_next_qvalues, dim=-1)[0]

        target_qvalues_for_actions = rewards + self.gamma * next_state_values
        target_qvalues_for_actions = torch.where(is_done, rewards, target_qvalues_for_actions)

        loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

        print(f'loss was: {loss.item()}')
        return loss

    def learn(self, states, actions, rewards, next_states, is_done):
        self.network.train()

        self.opt.zero_grad()
        self.__compute_td_loss(states, actions, rewards, next_states, is_done).backward()
        self.opt.step()
        self.epsilon = max(self.epsilon * self.delta_epsilon, self.min_epsilon)

    def get_action(self, state):
        self.network.eval()

        state = torch.tensor(numpy.array([state]), dtype=torch.float32)
        q_values = self.network(state).detach().numpy()
        # print(q_values.shape)

        greedy_action = np.argmax(q_values)
        should_explore = np.random.binomial(n=1, p=self.epsilon)

        chosen_action = np.random.choice(range(q_values.shape[-1])) if should_explore else greedy_action

        # print(q_values)
        # print(chosen_action)

        return int(chosen_action)

    def save(self):
        torch.save(self.network, self.model_path)
