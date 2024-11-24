from air_hockey_challenge.framework import AgentBase
import numpy as np

class JointPolicyAgent(AgentBase):
    def __init__(self, env_info, alg, is_off_policy=False, **kwargs):
        super().__init__(env_info, **kwargs)
        mdp_info = env_info["rl_info"].copy()
        self.policy = alg(mdp_info, **kwargs)
        self.is_off_policy = is_off_policy
        self._add_save_attr(
            policy="mushroom",
        )

    def fit(self, dataset, alt_dataset=[], **kwargs):
        if self.is_off_policy:
            dataset += alt_dataset

        _dataset = []
        for state, action, reward, next_state, absorbing, last in dataset:
            _dataset.append(
                (state, action[1], reward, next_state, absorbing, last)
            )
        self.policy.fit(
            _dataset,
            **kwargs
        )

    def reset(self):
        pass

    def draw_action(self, observation):
        q = observation[self.env_info["joint_pos_ids"]]
        dq_desired = self.policy.draw_action(observation)
        q_desired = q + dq_desired * self.env_info["rl_info"].dt
        return np.vstack([q_desired, dq_desired])