import numpy as np

from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box

from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.sac import get_sac_agent
from air_hockey_challenge.utils.kinematics import inverse_kinematics

class RawPolicyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        mdp_info = env_info['rl_info'].copy()
        self.sac = get_sac_agent(mdp_info, **kwargs)
        # self.sac = get_sac_agent(
        #     MDPInfo(
        #         observation_space=mdp_info.observation_space,
        #         action_space=Box(
        #             np.array([-.5, .5]) * env_info['table']['length'],
        #             np.array([-.5, .5]) * env_info['table']['width']
        #         ),
        #         gamma=env_info['rl_info'].gamma,
        #         horizon=env_info['rl_info'].horizon,
        #         dt=env_info['rl_info'].dt
        #     ),
        #     **kwargs)
        self._add_save_attr(sac='mushroom')

    def fit(self, dataset, **kwargs):
        self.sac.fit([(sample[0], sample[1].flatten()) + sample[2:] for sample in dataset])

    def reset(self):
        pass

    def draw_action(self, observation):
        # ee_desired_pos = self.sac.draw_action(observation)
        # q_desired = inverse_kinematics(
        #     self.env_info['robot']['robot_model'],
        #     self.env_info['robot']['robot_data'],
        #     np.concatenate(ee_desired_pos, np.array([self.env_info['robot']['ee_desired_height']])),
        # )
        # dq_desired = np.zeros(7)
        # return np.vstack((q_desired, dq_desired))
        return self.sac.draw_action(observation).reshape(2, -1)