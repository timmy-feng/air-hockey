import numpy as np

class Reward:
    def reward(self, base_env, state, action, next_state, absorbing):
        raise NotImplementedError('Reward is an abstract class and should not be instantiated.')

    def __call__(self, base_env, state, action, next_state, absorbing):
        return self.reward(base_env, state, action, next_state, absorbing)

class RewardList(Reward):
    def __init__(self, rewards, weights):
        assert(len(rewards) == len(weights))
        self.rewards = rewards
        self.weights = weights

    def reward(self, base_env, state, action, next_state, absorbing):
        return sum(
            weight * reward(base_env, state, action, next_state, absorbing)
            for reward, weight in zip(self.rewards, self.weights)
        )

class ScoreReward(Reward):
    def reward(self, base_env, state, action, next_state, absorbing):
        puck_pos, _ = base_env.get_puck(next_state)
        score_reward = 0
        if absorbing:
            # Puck in Goal
            if (np.abs(puck_pos[1]) - base_env.env_info['table']['goal_width'] / 2) <= 0:
                # Score for home
                if puck_pos[0] > base_env.env_info['table']['length'] / 2:
                    score_reward += 1
                # Score for opponent
                elif puck_pos[0] < -base_env.env_info['table']['length'] / 2:
                    score_reward -= 1
        return np.array([score_reward, -score_reward])

class ConstraintReward(Reward):
    def __init__(self, name):
        self.name = name

    def reward(self, base_env, state, action, next_state, absorbing):
        loss = []
        for agent in [1, 2]:
            q, dq = base_env.get_joints(next_state, agent=agent)
            loss.append(np.maximum(0, base_env.env_info['constraints'].get(self.name).fun(q, dq)).sum())
        return -np.array(loss)