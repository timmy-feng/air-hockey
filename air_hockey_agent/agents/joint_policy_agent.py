from air_hockey_challenge.framework import AgentBase

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
                (state, action.flatten(), reward, next_state, absorbing, last)
            )
        self.policy.fit(
            [(sample[0], sample[1].flatten()) + sample[2:] for sample in dataset],
            **kwargs
        )

    def reset(self):
        pass

    def draw_action(self, observation):
        q = observation[self.env_info["joint_pos_ids"]]
        action = self.policy.draw_action(observation)
        action[:7] = q + action[7:] * self.env_info["rl_info"].dt
        return action.reshape(2, -1)
