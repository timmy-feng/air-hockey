from air_hockey_challenge.utils.kinematics import forward_kinematics
import numpy as np
import math

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
    def __init__(self, lam=1, agent=None):
        self.lam = lam
        self.agent = agent

    def reward(self, base_env, state, action, next_state, absorbing):
        puck_pos, _ = base_env.get_puck(next_state)
        ee_pos, _ = base_env.get_ee() if self.agent is None else base_env.get_ee(self.agent)
        return np.exp(-self.lam * np.linalg.norm(puck_pos - ee_pos) ** 2)
    
class PuckVelocityReward(Reward):
    def __init__(self, lam=1, pov=1):
        self.pov = pov
        self.lam = lam

    def reward(self, base_env, state, action, next_state, absorbing):
        _, puck_vel = base_env.get_puck(next_state)
        return math.tanh(self.lam * self.pov * puck_vel[0] ** 2)

class EffortReward(Reward):
    def __init__(self, agent=None):
        self.agent = agent

    def reward(self, base_env, state, action, next_state, absorbing):
        _, dq0 = base_env.get_joints(state) if self.agent is None else base_env.get_joints(state, self.agent)
        _, dq1 = base_env.get_joints(next_state) if self.agent is None else base_env.get_joints(next_state, self.agent)
        return -np.linalg.norm((dq1 - dq0) / base_env.dt)

class PlaneAvoidanceReward(Reward):
    def __init__(self, plane, offset, link='ee', d_max=0.01, agent=None):
        self.link = link
        self.plane = plane
        self.offset = offset
        self.d_max = d_max
        self.agent = agent

    def reward(self, base_env, state, action, next_state, absorbing):
        q, _ = base_env.get_joints(next_state) if self.agent is None else base_env.get_joints(next_state, self.agent)
        link_pos, _ = forward_kinematics(
            base_env.env_info['robot']['robot_model'],
            base_env.env_info['robot']['robot_data'],
            q,
            link=self.link
        )

        d = (np.dot(self.plane, link_pos) - self.offset) / np.linalg.norm(self.plane)

        return -max(0, 1 - d / self.d_max)

class ScoreReward(Reward):
    def __init__(self, pov=1):
        self.pov = pov

    def reward(self, base_env, state, action, next_state, abosorbing):
        puck_pos, _ = base_env.get_puck(next_state)
        score = 0
        if (np.abs(puck_pos[1]) - base_env.env_info['table']['goal_width'] / 2) <= 0:
            if puck_pos[0] > base_env.env_info['table']['length'] / 2:
                score += self.pov
            if puck_pos[0] < -base_env.env_info['table']['length'] / 2:
                score -= self.pov
        return score

class ScoreDistanceReward(Reward):
    def __init__(self, lam=1, pov=1):
        self.lam = lam
        self.pov = pov

    def reward(self, base_env, state, action, next_state, absorbing):
        puck_pos, _ = base_env.get_puck(next_state)
        score_dist = np.array([base_env.env_info['table']['length'] / 2 * self.pov, 0]) - puck_pos[:2]
        return np.exp(-self.lam * np.linalg.norm(score_dist) ** 2)