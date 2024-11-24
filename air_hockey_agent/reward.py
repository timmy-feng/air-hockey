from air_hockey_challenge.utils.kinematics import forward_kinematics
import numpy as np

class Reward:
    def reward(self, base_env, state, action, next_state, absorbing):
        raise NotImplementedError(
            "Reward is an abstract class and should not be instantiated."
        )

    def __call__(self, base_env, state, action, next_state, absorbing):
        return self.reward(base_env, state, action, next_state, absorbing)

class RewardList(Reward):
    def __init__(self, rewards, weights):
        assert len(rewards) == len(weights)
        self.rewards = rewards
        self.weights = weights

    def reward(self, base_env, state, action, next_state, absorbing):
        return sum(
            weight * reward(base_env, state, action, next_state, absorbing)
            for reward, weight in zip(self.rewards, self.weights)
        )

class PuckDistanceReward(Reward):
    def __init__(self, lam=1):
        self.lam = lam

    def reward(self, base_env, state, action, next_state, absorbing):
        puck_pos, _ = base_env.get_puck(next_state)
        ee_pos, _ = base_env.get_ee()
        return np.exp(-self.lam * np.linalg.norm(puck_pos - ee_pos) ** 2)

class EffortReward(Reward):
    def reward(self, base_env, state, action, next_state, absorbing):
        _, dq0 = base_env.get_joints(state)
        _, dq1 = base_env.get_joints(next_state)
        return -np.linalg.norm((dq1 - dq0) / base_env.dt)

class PlaneAvoidanceReward(Reward):
    def __init__(self, plane, offset, link='ee', d_max=0.01):
        self.link = link
        self.plane = plane
        self.offset = offset
        self.d_max = d_max

    def reward(self, base_env, state, action, next_state, absorbing):
        q, _ = base_env.get_joints(next_state)
        link_pos, _ = forward_kinematics(
            base_env.env_info['robot']['robot_model'],
            base_env.env_info['robot']['robot_data'],
            q,
            link=self.link
        )

        d = (np.dot(self.plane, link_pos) - self.offset) / np.linalg.norm(self.plane)

        return -max(0, 1 - d / self.d_max)
