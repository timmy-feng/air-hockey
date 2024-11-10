import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, num_units, **kwargs):
        super().__init__()

        assert(num_layers >= 2)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.num_units = num_units

        self.layers = nn.ModuleList(
            [nn.Linear(input_shape[0], num_units).double()] + \
            [nn.Linear(num_units, num_units).double() for _ in range(num_layers - 2)] + \
            [nn.Linear(num_units, output_shape[0]).double()]
        )

    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers):
            x = layer(F.relu(x) if i > 0 else x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, num_units, **kwargs):
        super().__init__()

        assert(num_layers >= 2)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
        self.num_units = num_units

        self.layers = nn.ModuleList(
            [nn.Linear(input_shape[0], num_units).double()] + \
            [nn.Linear(num_units, num_units).double() for _ in range(num_layers - 2)] + \
            [nn.Linear(num_units, output_shape[0]).double()]
        )
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(F.relu(x) if i > 0 else x)
        return torch.squeeze(x)

def get_sac_agent(mdp_info):
    initial_replay_size = 5000
    max_replay_size = 200000
    batch_size = 64
    warmup_transitions = 10000
    tau = 0.001
    lr_alpha = 3e-4

    actor_mu_params = dict(
        network=ActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        num_layers=3,
        num_units=128,
    )

    actor_sigma_params = dict(
        network=ActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        num_layers=3,
        num_units=128,
    )

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 5e-4}}

    critic_params = dict(
        network=CriticNetwork,
        optimizer={'class': optim.Adam,
                   'params': {'lr': 5e-4}},
        loss=F.mse_loss,
        input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
        output_shape=(1,),
        num_layers=3,
        num_units=128,
    )

    return SAC(mdp_info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)
