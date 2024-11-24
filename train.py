import numpy as np
from argparse import ArgumentParser

from air_hockey_challenge.framework import AirHockeyChallengeWrapper, ChallengeCore
from air_hockey_challenge.utils.tournament_agent_wrapper import (
    TrainingTournamentAgentWrapper,
)
from baseline.baseline_agent.baseline_agent import BaselineAgent

from air_hockey_agent.environments.env_joint_control import IiwaPositionJointControl
from air_hockey_agent.agent_builder import build_agent
from air_hockey_agent.reward import *

from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.core import Logger, Core
from mushroom_rl.utils.spaces import Box


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        choices=["joint", "ee"],
        default="joint",
        help="action space",
    )
    parser.add_argument(
        "--alg",
        type=str,
        choices=["sac", "ppo"],
        default="ppo",
        help="algorithm to use",
    )
    parser.add_argument(
        "--use_cuda", action="store_true", help="flag to use CUDA for training"
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="epoch to start training from"
    )
    parser.add_argument(
        "--end_epoch", type=int, default=1000, help="epoch to end training at"
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="checkpoint to continue training from",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="frequency of checkpointing in epochs",
    )
    parser.add_argument(
        "--layer_sizes",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256, 256],
        help="sizes of the layers in the neural network",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument(
        "--initial_replay_size",
        type=int,
        default=4096,
        help="initial size of the replay buffer (only for SAC)",
    )
    parser.add_argument(
        "--n_steps", type=int, default=4096, help="number of steps per epoch"
    )
    return parser.parse_args()

def get_constraint_rewards(env_info):
    tolerance = 0.02

    x_lb = -env_info['robot']['base_frame'][0][0, 3] - (
        env_info['table']['length'] / 2 - env_info['mallet']['radius'])
    y_lb = -(env_info['table']['width'] / 2 - env_info['mallet']['radius'])
    y_ub = (env_info['table']['width'] / 2 - env_info['mallet']['radius'])
    z_lb = env_info['robot']['ee_desired_height'] - tolerance
    z_ub = env_info['robot']['ee_desired_height'] + tolerance

    link_lb = 0.25

    return RewardList(
        rewards=[
            PlaneAvoidanceReward(
                plane=np.array([0, 0, 1]),
                offset=0.25,
                link='4',
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, 1]),
                offset=0.25,
                link='7',
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, 1]),
                offset=z_lb,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, -1]),
                offset=-z_ub,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 1, 0]),
                offset=y_lb,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, -1, 0]),
                offset=-y_ub,
            ),
            PlaneAvoidanceReward(
                plane=np.array([1, 0, 0]),
                offset=x_lb,
            ),
        ],
        weights=[1, 1, 1, 1, 1, 1, 1],
    )


if __name__ == "__main__":
    args = get_args()

    mdp = IiwaPositionJointControl()

    mdp.env_info['rl_info'].action_space = Box(*mdp.env_info["robot"]["joint_vel_limit"])

    custom_reward = RewardList(
        rewards=[
            PuckDistanceReward(lam=20),
            EffortReward(),
            get_constraint_rewards(mdp.env_info),
        ],
        weights=[1, 0.005, 0.1],
    )

    mdp.reward = lambda obs, action, next_obs, absorbing: \
        custom_reward(mdp, obs, action, next_obs, absorbing)

    if args.continue_from is not None:
        agent = build_agent(
            mdp.env_info,
            agent=args.action + "_policy",
            policy=args.alg,
            load_agent=f"checkpoints/{args.continue_from}/epoch_{args.start_epoch}.msh",
        )
    else:
        agent = build_agent(
            mdp.env_info,
            agent=args.action + "_policy",
            policy=args.alg,
            layer_sizes=args.layer_sizes,
            batch_size=args.batch_size,
            initial_replay_size=args.initial_replay_size,
            is_off_policy=args.alg == "sac",
            use_cuda=args.use_cuda,
            is_joint_policy=True,
        )

    core = Core(agent, mdp)

    if args.continue_from is None:
        logger = Logger(
            "train", results_dir="./logs", use_timestamp=True, log_console=True
        )
    else:
        logger = Logger(
            args.continue_from,
            results_dir="./logs",
            use_timestamp=True,
            log_console=True,
            append=True,
        )

    logger.strong_line()
    logger.info(f"Training started at {logger._log_id}")

    if args.continue_from is None:
        if args.alg == "sac":
            core.learn(
                n_steps=args.initial_replay_size,
                n_steps_per_fit=args.initial_replay_size,
            )

    def checkpoint(name):
        agent.save(f"checkpoints/{logger._log_id}/{name}.msh", full_save=True)
        logger.info(f"Saved model to checkpoints/{logger._log_id}/{name}.msh")

    smoothed_J = 0
    alpha_J = 0.25

    for epoch in range(args.start_epoch, args.end_epoch):
        if epoch % args.checkpoint_every == 0 and epoch != args.start_epoch:
            checkpoint(f"epoch_{epoch}")

        core.learn(
            n_steps=args.n_steps, n_steps_per_fit=1 if args.alg == "sac" else args.n_steps // 4
        )

        dataset = core.evaluate(n_episodes=1)

        states, actions, rewards, next_states, abosorbing, _ = map(
            np.array, zip(*dataset)
        )

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))

        smoothed_J = alpha_J * J + (1 - alpha_J) * smoothed_J if epoch > args.start_epoch else J
        if smoothed_J >= 100:
            mdp.grow_start_range(0.25)
            logger.info(f"Increased size of start region")

        if args.alg == "sac":
            E = agent.policy.policy.entropy(states)
            logger.epoch_info(epoch, J=J, R=R, E=E)
        else:
            logger.epoch_info(epoch, J=J, smoothed_J=smoothed_J, R=R)

    checkpoint("final")

    print('Press [Enter] to test the agent')
    input()

    core.evaluate(n_episodes=10, render=True)