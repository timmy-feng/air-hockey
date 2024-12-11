from air_hockey_challenge.framework import AgentBase
import numpy as np

class JointPolicyAgent(AgentBase):
    def __init__(self, env_info, alg, batches_per_fit=1, **kwargs):
        super().__init__(env_info, **kwargs)
        mdp_info = env_info["rl_info"].copy()
        self.policy = alg(mdp_info, **kwargs)
        self.batches_per_fit = batches_per_fit
        self._add_save_attr(
            policy="mushroom",
            batches_per_fit="primitive"
        )

    def fit(self, dataset, **kwargs):
        _dataset = []
        for state, action, reward, next_state, absorbing, last in dataset:
            _dataset.append(
                (
                    state.astype(np.float32),
                    action[1].astype(np.float32),
                    np.float32(reward),
                    next_state.astype(np.float32),
                    absorbing,
                    last
                )
            )

        self.policy.fit(_dataset, **kwargs)
        for _ in range(self.batches_per_fit - 1):
            self.policy.fit([], **kwargs)

    def reset(self):
        pass

    def draw_action(self, observation):
        q = observation[self.env_info["joint_pos_ids"]]
        dq_desired = self.policy.draw_action(observation.astype(np.float32))
        q_desired = q + dq_desired * self.env_info["rl_info"].dt
        return np.vstack([q_desired, dq_desired])