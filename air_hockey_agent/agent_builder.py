from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.raw_policy_agent import RawPolicyAgent

def build_agent(env_info, **kwargs) -> AgentBase:
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return RawPolicyAgent(env_info, **kwargs)