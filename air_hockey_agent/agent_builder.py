from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents import *
from air_hockey_agent.policy import *

agents = {
    'joint_policy': JointPolicyAgent,
    'ee_policy': EEPolicyAgent,
}

policies = {
    'sac': get_sac_agent,
    'ppo': get_ppo_agent
}

def build_agent(env_info, agent='joint_policy', policy='sac', **kwargs) -> AgentBase:
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    if "load_agent" in kwargs:
        return agents[agent].load_agent(kwargs["load_agent"], env_info)
    return agents[agent](env_info, alg=policies[policy], **kwargs)
