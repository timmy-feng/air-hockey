import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC


class ActorNetwork(nn.Module):
    def __init__(
        self, input_shape, output_shape, layer_sizes, set_last_layer=None, **kwargs
    ):
        super().__init__()

        assert len(layer_sizes) > 0

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layers = nn.ModuleList(
            [nn.Linear(input_shape[0], layer_sizes[0]).double()]
            + [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]).double()
                for i in range(len(layer_sizes) - 1)
            ]
            + [nn.Linear(layer_sizes[-1], output_shape[0]).double()]
        )

        if set_last_layer is not None:
            self.layers[-1].weight.data.fill_(0)
            self.layers[-1].bias.data.fill_(set_last_layer)

    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers):
            x = layer(F.relu(x) if i > 0 else x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, layer_sizes, **kwargs):
        super().__init__()

        assert len(layer_sizes) > 0

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layers = nn.ModuleList(
            [nn.Linear(input_shape[0], layer_sizes[0]).double()]
            + [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]).double()
                for i in range(len(layer_sizes) - 1)
            ]
            + [nn.Linear(layer_sizes[-1], output_shape[0]).double()]
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(F.relu(x) if i > 0 else x)
        return torch.squeeze(x)


def get_sac_agent(
    mdp_info,
    layer_sizes=(256, 256, 256),
    initial_replay_size=5000,
    max_replay_size=200000,
    batch_size=64,
    warmup_transitions=10000,
    tau=0.001,
    lr_alpha=3e-5,
    use_cuda=False,
    **kwargs
):
    actor_mu_params = dict(
        network=ActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        layer_sizes=layer_sizes,
        use_cuda=use_cuda,
    )

    actor_sigma_params = dict(
        network=ActorNetwork,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        layer_sizes=layer_sizes,
        use_cuda=use_cuda,
    )

    actor_optimizer = {"class": optim.AdamW, "params": {"lr": 3e-5}}

    critic_params = dict(
        network=CriticNetwork,
        optimizer={"class": optim.AdamW, "params": {"lr": 3e-5}},
        loss=F.mse_loss,
        input_shape=(
            mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],
        ),
        output_shape=(1,),
        layer_sizes=layer_sizes,
        use_cuda=use_cuda,
    )

    return SAC(
        mdp_info,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        critic_fit_params=None,
    )
