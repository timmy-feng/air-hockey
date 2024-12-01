import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy


class Network(nn.Module):
    def __init__(
        self, input_shape, output_shape, layer_sizes, set_last_layer=None, **kwargs
    ):
        super().__init__()

        assert len(layer_sizes) > 0

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layers = nn.ModuleList(
            [nn.Linear(input_shape[0], layer_sizes[0])]
            + [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
            + [nn.Linear(layer_sizes[-1], output_shape[0])]
        )

        if set_last_layer is not None:
            self.layers[-1].weight.data.fill_(0)
            self.layers[-1].bias.data.fill_(set_last_layer)

    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers):
            x = layer(F.relu(x) if i > 0 else x)
        return x

def get_ppo_agent(
    mdp_info,
    layer_sizes=(256, 256),
    n_epochs_policy=4,
    batch_size=64,
    eps_ppo=.2,
    lam=.95,
    use_cuda=False,
    **kwargs
):
    policy = GaussianTorchPolicy(
        network=Network,
        input_shape=mdp_info.observation_space.shape,
        output_shape=mdp_info.action_space.shape,
        layer_sizes=layer_sizes,
        use_cuda=use_cuda,
    )

    actor_optimizer = {"class": optim.AdamW, "params": {"lr": 3e-4}}

    critic_params = dict(
        network=Network,
        optimizer={"class": optim.AdamW, "params": {"lr": 3e-4}},
        loss=F.mse_loss,
        input_shape=mdp_info.observation_space.shape,
        output_shape=(1,),
        layer_sizes=layer_sizes,
        use_cuda=use_cuda,
    )

    return PPO(
        mdp_info,
        policy,
        actor_optimizer,
        critic_params,
        n_epochs_policy=n_epochs_policy,
        batch_size=batch_size,
        eps_ppo=eps_ppo,
        lam=lam,
        ent_coeff=1e-3,
        critic_fit_params=None,
    )
