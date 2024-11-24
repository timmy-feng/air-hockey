import numpy as np

from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle
from air_hockey_challenge.environments import position_control_wrapper

from mushroom_rl.utils.spaces import Box


class AirHockeyJointControl(AirHockeySingle):
    """
        Class for the joint control task.
        The agent should move the end-effector to the goal position.
    """
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}, lam_err=20, lam_eff=0.005, **kwargs):
        self.start_range = np.array([[-0.8, -0.6], [-0.1, 0.1]])
        self.table_range = np.array([[-0.9, -0.3], [-0.4, 0.4]])

        self.lam_err = lam_err
        self.lam_eff = lam_eff

        self.timer = 0

        self.joint_vel = np.zeros(7)
        self.joint_acc = np.zeros(7)

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        # puck is just a goal position, disable collision
        self._disable_puck_collision()

    def grow_start_range(self, alpha):
        self.start_range = self.table_range * alpha + self.start_range * (1 - alpha)

    def _disable_puck_collision(self):
        puck_geom_id = self._model.geom("puck").id
        self._model.geom(puck_geom_id).contype = 0
        self._model.geom(puck_geom_id).conaffinity = 0

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        # add a dummy observation for the ghost opponent
        observation_space = mdp_info.observation_space
        obs_low = np.concatenate([observation_space.low, [1.5, -1.5, -1.5]])
        obs_high = np.concatenate([observation_space.high, [4.5, 1.5, 1.5]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        # add a dummy observation for the ghost opponent
        return np.concatenate([obs, np.random.rand(3) * 3 + np.array([1.5, -1.5, -1.5])])

    def setup(self, obs):
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]
        puck_vel = np.zeros(3)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        self.timer = 0

        self.joint_vel = np.zeros(7)
        self.joint_acc = np.zeros(7)

        super(AirHockeyJointControl, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        puck_pos, _ = self.get_puck(state)
        ee_pos, _ = self.get_ee()

        puck_reward = np.exp(-self.lam_err * np.linalg.norm(puck_pos - ee_pos) ** 2)
        effort_reward = -self.lam_eff * np.linalg.norm(self.joint_acc)

        return puck_reward + effort_reward

    def is_absorbing(self, state):
        _, joint_vel = self.get_joints(state)
        self.joint_acc = (joint_vel - self.joint_vel) / self.dt
        self.joint_vel = joint_vel

        self.timer += self.dt
        if self.timer > 15.0:
            return True

        return super().is_absorbing(state)

class IiwaPositionJointControl(position_control_wrapper.PositionControlIIWA, AirHockeyJointControl):
    pass


if __name__ == '__main__':
    env = AirHockeyJointControl()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    env.render()
    while True:
        # action = np.random.uniform(-1, 1, env.info.action_space.low.shape) * 8
        action = np.zeros(7)
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
