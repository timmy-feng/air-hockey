from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.sac import get_sac_agent

class RawPolicyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.sac = get_sac_agent(env_info['rl_info'])
        self._add_save_attr(sac='mushroom')

    def fit(self, dataset, **kwargs):
        self.sac.fit([(sample[0], sample[1].flatten()) + sample[2:] for sample in dataset])

    def reset(self):
        pass

    def draw_action(self, observation):
        action = self.sac.draw_action(observation)
        q_pos = self.get_joint_pos(observation)
        # action[:7] = q_pos
        action[7:] = 0
        return action.reshape(2, 7)