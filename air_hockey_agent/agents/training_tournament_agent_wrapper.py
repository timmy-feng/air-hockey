from air_hockey_challenge.utils.tournament_agent_wrapper import TournamentAgentWrapper, timeit

class TrainingTournamentAgentWrapper(TournamentAgentWrapper):
    def __init__(self, env_info, agent_1, agent_2):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.episode_start_1 = self.agent_1.episode_start
        self.episode_start_2 = self.agent_2.episode_start

    def get_dataset_1(self, dataset):
        agent_1_dataset = []
        for state, action, reward, next_state, absorbing, last in dataset:
            agent_1_dataset.append((
                state[:len(state) // 2],
                action[0],
                reward[0],
                next_state[:len(next_state) // 2],
                absorbing,
                last
            ))
        return agent_1_dataset
    
    def get_dataset_2(self, dataset):
        agent_2_dataset = []
        for state, action, reward, next_state, absorbing, last in dataset:
            agent_2_dataset.append((
                state[len(state) // 2:],
                action[1],
                reward[1],
                next_state[len(next_state) // 2:],
                absorbing,
                last
            ))
        return agent_2_dataset

    def fit(self, dataset, **kwargs):
        try:
            self.agent_1.fit(self.get_dataset_1(dataset), **kwargs)
        except NotImplementedError:
            pass

        try:
            self.agent_2.fit(self.get_dataset_2(dataset), **kwargs)
        except NotImplementedError:
            pass

    @timeit
    def get_action_1(self, obs_1):
        return self.agent_1.draw_action(obs_1)

    @timeit
    def get_action_2(self, obs_2):
        return self.agent_2.draw_action(obs_2)

    def preprocessor_1(self, obs_1):
        for p in self.agent_1.preprocessors:
            obs_1 = p(obs_1)
        return obs_1

    def preprocessor_2(self, obs_2):
        for p in self.agent_2.preprocessors:
            obs_2 = p(obs_2)
        return obs_2