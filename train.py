import numpy as np
from argparse import ArgumentParser

from air_hockey_challenge.framework import AirHockeyChallengeWrapper, ChallengeCore
from air_hockey_challenge.utils.tournament_agent_wrapper import TrainingTournamentAgentWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent

from air_hockey_agent.agent_builder import build_agent
from air_hockey_agent.reward import *

from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.core import Logger

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--action', type=str, choices=['joint', 'ee'], default='joint')
    parser.add_argument('--alg', type=str, choices=['sac', 'ppo'], default='sac')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=100)
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[256, 128, 64, 32, 64, 128, 256])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--initial_replay_size', type=int, default=5000)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_steps_test', type=int, default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    mdp = AirHockeyChallengeWrapper(
        env='tournament',
        custom_reward_function=RewardList(
            rewards=[ScoreReward(), ConstraintReward('ee_constr'), ConstraintReward('link_constr')],
            weights=[1, 1, 1],
        ),
    )

    if args.continue_from is not None:
        agent_1 = build_agent(
            mdp.env_info,
            agent=args.action + '_policy',
            policy=args.alg,
            load_agent=f'checkpoints/{args.continue_from}/epoch_{args.start_epoch}.msh',
        )
    else:
        agent_1 = build_agent(
            mdp.env_info,
            agent=args.action + '_policy',
            policy=args.alg,
            layer_sizes=args.layer_sizes,
            batch_size=args.batch_size,
            initial_replay_size=args.initial_replay_size,
            is_off_policy=args.alg == 'sac',
            use_cuda=args.use_cuda,
            is_joint_policy=True,
        )

    agent = TrainingTournamentAgentWrapper(
        mdp.env_info,
        agent_1=agent_1,
        agent_2=BaselineAgent(mdp.env_info, 2),
    )

    core = ChallengeCore(agent, mdp, is_tournament=True, init_state=mdp.base_env.init_state, time_limit=0.02)

    if args.continue_from is None:
        logger = Logger('train', results_dir='./logs', use_timestamp=True, log_console=True)
    else:
        logger = Logger(args.continue_from, results_dir='./logs',use_timestamp=True,  log_console=True, append=True)

    logger.strong_line()
    logger.info(f'Training started at {logger._log_id}')

    if args.continue_from is None:
        if args.alg == 'sac':
            core.learn(n_steps=args.initial_replay_size, n_steps_per_fit=args.initial_replay_size)

    def checkpoint(name):
        agent_1.save(f'checkpoints/{logger._log_id}/{name}.msh', full_save=True)
        logger.info(f'Saved model to checkpoints/{logger._log_id}/{name}.msh')

    for epoch in range(args.start_epoch, args.end_epoch):
        if epoch % args.checkpoint_every == 0:
            checkpoint(f'epoch_{epoch}')

        core.learn(n_steps=args.n_steps, n_steps_per_fit=1 if args.alg == 'sac' else 1000)

        dataset = agent.get_dataset_1(core.evaluate(n_steps=args.n_steps_test))
        states, actions, rewards, next_states, abosorbing, _ = map(np.array, zip(*dataset))

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))

        if args.alg == 'sac':
            E = agent_1.policy.policy.entropy(states)
            logger.epoch_info(epoch, J=J, R=R, E=E)
        else:
            logger.epoch_info(epoch, J=J, R=R)

    checkpoint('final')