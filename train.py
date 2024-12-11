import numpy as np
from argparse import ArgumentParser

from air_hockey_agent.environments.env_joint_control import IiwaPositionJointControl
from air_hockey_agent.agents.training_tournament_agent_wrapper import TrainingTournamentAgentWrapper
from air_hockey_agent.agent_builder import build_agent
from air_hockey_agent.reward import *

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from baseline.baseline_agent.baseline_agent import BaselineAgent

from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import to_parameter
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
        "--environment",
        type=str,
        choices=["control", "tournament"],
        default="control",
        help="environment to train in"
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
        "--warmup_transitions",
        type=int,
        default=-1,
        help="number of transitions to warmup the agent with, resets memory",
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
        "--batch_size", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument(
        "--fits_per_epoch", type=int, default=256, help="number of fits per epoch"
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
    parser.add_argument(
        "--start_tier", type=int, default=0, help="tier to start training from"
    )
    return parser.parse_args()

def get_constraint_rewards(env_info, agent=None):
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
                offset=link_lb,
                link='4',
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, 1]),
                offset=link_lb,
                link='7',
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, 1]),
                offset=z_lb,
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 0, -1]),
                offset=-z_ub,
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, 1, 0]),
                offset=y_lb,
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([0, -1, 0]),
                offset=-y_ub,
                agent=agent,
            ),
            PlaneAvoidanceReward(
                plane=np.array([1, 0, 0]),
                offset=x_lb,
                agent=agent,
            ),
        ],
        weights=[1, 1, 1, 1, 1, 1, 1],
    )

def get_mdp(args):
    if args.environment == 'control':
        mdp = IiwaPositionJointControl()

        custom_reward = RewardList(
            rewards=[
                PuckDistanceReward(lam=5),
                EffortReward(),
                get_constraint_rewards(mdp.env_info),
            ],
            weights=[1, 0.005, 0.1],
        )

        mdp.reward = lambda obs, action, next_obs, absorbing: \
            custom_reward(mdp, obs, action, next_obs, absorbing)
    else:
        mdp = AirHockeyChallengeWrapper(env='tournament')

        custom_reward_1 = RewardList(
            rewards=[
                ScoreReward(pov=1),
                ScoreDistanceReward(pov=1, lam=5),
                PuckVelocityReward(pov=1, lam=5),
                EffortReward(),
                get_constraint_rewards(mdp.env_info, agent=1),
            ],
            weights=[100, 1, 0.1, 0.005, 0.1],
        )

        custom_reward_2 = RewardList(
            rewards=[
                ScoreReward(pov=-1),
                ScoreDistanceReward(pov=-1, lam=5),
                PuckVelocityReward(pov=-1, lam=5),
                EffortReward(),
                get_constraint_rewards(mdp.env_info, agent=2),
            ],
            weights=[100, 1, 0.1, 0.005, 0.1],
        )

        mdp.base_env.reward = lambda obs, action, next_obs, absorbing: \
            (custom_reward_1(mdp.base_env, obs, action, next_obs, absorbing), \
             custom_reward_2(mdp.base_env, obs, action, next_obs, absorbing))

    mdp.env_info['rl_info'].action_space = Box(*mdp.env_info["robot"]["joint_vel_limit"])

    return mdp

def get_agent(args, mdp):
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
            use_cuda=args.use_cuda,
            is_joint_policy=True,
        )

    if args.alg == 'sac' and args.warmup_transitions != -1:
        agent.policy._replay_memory.reset()
        agent.policy._warmup_transitions = to_parameter(args.warmup_transitions)

    return agent

def get_core(args, mdp, agent):
    agent_wrapper = None
    if args.environment == 'control':
        core = Core(agent, mdp)
    else:
        agent_wrapper = TrainingTournamentAgentWrapper(mdp.env_info, agent, agent)
        core = ChallengeCore(agent_wrapper, mdp, is_tournament=True, init_state=mdp.base_env.init_state, time_limit=0.02)

    return agent_wrapper, core

def get_logger(args):
    if args.continue_from is None:
        return Logger(
            "train", results_dir="./logs", use_timestamp=True, log_console=True
        )
    else:
        return Logger(
            args.continue_from,
            results_dir="./logs",
            use_timestamp=False,
            log_console=True,
            append=True,
        )

if __name__ == "__main__":
    args = get_args()

    mdp = get_mdp(args)
    agent = get_agent(args, mdp)
    agent_wrapper, core = get_core(args, mdp, agent)
    logger = get_logger(args)

    logger.strong_line()
    logger.info(f"Training started at {logger._log_id}")

    def checkpoint(name):
        agent.save(f"checkpoints/{logger._log_id}/{name}.msh", full_save=True)
        logger.info(f"Saved model to checkpoints/{logger._log_id}/{name}.msh")

    if args.alg == "sac" and agent.policy._replay_memory.size < args.initial_replay_size:
        logger.info('Initialising replay buffer')
        core.learn(
            n_steps=args.initial_replay_size,
            n_steps_per_fit=args.initial_replay_size,
        )

    smoothed_J, alpha_J, env_tier = 0, 0.25, 0

    if args.environment == 'control' and args.start_tier > 0:
        for _ in range(args.start_tier):
            mdp.grow_start_range(0.25)
            env_tier += 1
        logger.info(f"Started training at tier {env_tier}")

    for epoch in range(args.start_epoch, args.end_epoch):
        if epoch % args.checkpoint_every == 0 and epoch != args.start_epoch:
            checkpoint(f"epoch_{epoch}")

        core.learn(
            n_steps=args.n_steps, n_steps_per_fit=args.n_steps // args.fits_per_epoch
        )

        dataset = core.evaluate(n_episodes=1)
        if agent_wrapper is not None:
            dataset = agent_wrapper.get_dataset_1(dataset)

        states, actions, rewards, next_states, abosorbing, _ = map(
            np.array, zip(*dataset)
        )

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        smoothed_J = alpha_J * J + (1 - alpha_J) * smoothed_J

        if args.alg == "sac":
            E = agent.policy.policy.entropy(np.array(states, dtype=np.float32))
            logger.epoch_info(epoch, J=J, smoothed_J=smoothed_J, R=R, E=E)
        else:
            logger.epoch_info(epoch, J=J, smoothed_J=smoothed_J, R=R)

        if args.environment == 'control' and smoothed_J >= 50:
            mdp.grow_start_range(0.25)
            env_tier += 1
            smoothed_J = 0
            logger.info(f"Increased size of start region, now at tier {env_tier}")

    checkpoint("final")

    print('Press [Enter] to test the agent')
    input()

    core.evaluate(n_episodes=10, render=True)