import numpy as np

from mushroom_rl.core import MDPInfo
from mushroom_rl.utils.spaces import Box

from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
)

class EEPolicyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        mdp_info = env_info["rl_info"].copy()
        self.policy = alg(
            MDPInfo(
                observation_space=mdp_info.observation_space,
                action_space=Box(
                    low=np.array(
                        [
                            0.3 * env_info["table"]["length"],
                            -0.1 * env_info["table"]["width"],
                        ]
                    ),
                    high=np.array(
                        [
                            0.4 * env_info["table"]["length"],
                            0.1 * env_info["table"]["width"],
                        ]
                    ),
                ),
                gamma=env_info["rl_info"].gamma,
                horizon=env_info["rl_info"].horizon,
                dt=env_info["rl_info"].dt,
            ),
            **kwargs
        )

        self._add_save_attr(
            policy="mushroom",
        )

    def fit(self, dataset, **kwargs):
        _dataset = []
        for state, action, reward, next_state, absorbing, last in dataset:
            q = state[self.env_info["puck_pos_ids"]]

            ee_desired_pos, _ = forward_kinematics(
                self.env_info["robot"]["robot_model"],
                self.env_info["robot"]["robot_data"],
                action[0],
            )

            ee_desired_vel = np.dot(
                jacobian(
                    self.env_info["robot"]["robot_model"],
                    self.env_info["robot"]["robot_data"],
                    q,
                ),
                action[1],
            )[:3]

            _action = np.concatentate((ee_desired_pos, ee_desired_vel))
            _dataset.append((state, _action, reward, next_state, absorbing, last))
        self.sac.fit(
            [(sample[0], sample[1].flatten()) + sample[2:] for sample in dataset],
            **kwargs
        )

    def reset(self):
        pass

    def draw_action(self, observation):
        q = self.get_joint_pos(observation)
        puck_pos = self.get_puck_pos(observation)
        ee_pos, _ = forward_kinematics(
            self.env_info["robot"]["robot_model"],
            self.env_info["robot"]["robot_data"],
            q,
        )

        ee_desired_pos = np.concatenate(
            (
                self.sac.draw_action(observation),
                np.array([self.env_info["robot"]["ee_desired_height"]]),
            )
        )

        ee_displacement = ee_desired_pos - ee_pos

        J = jacobian(
            self.env_info["robot"]["robot_model"],
            self.env_info["robot"]["robot_data"],
            q,
        )

        q_displacement = np.dot(
            np.linalg.pinv(J),
            np.concatenate(
                (np.array([1, 1, 2]) * ee_displacement, np.array([0, 0, 0]))
            ),
        )

        q_desired = q + q_displacement / self.env_info['rl_info'].dt
        dq_desired = -q_displacement

        action = np.clip(
            np.concatenate((q_desired, dq_desired)),
            self.env_info["rl_info"].action_space.low,
            self.env_info["rl_info"].action_space.high,
        )

        return action.reshape(2, -1)