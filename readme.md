# DeceptiveRL
This code accompanies the paper - Deceptive Reinforcement Learning: A Policy-Based Approach for High-Dimensional Action-Spaces. 

The paper investigates a policy-based approach to deceptive reinforcement learning for privacy preservation in high-dimensional action-spaces. In reinforcement learning, the reward function defines the agent's objective. In adversarial scenarios, an agent may want to both maximise rewards and keep it's reward function private from observers. Recent approaches use the concept of entropy to select actions which are ambiguous over a set of possible reward functions. Despite promising results, they are inefficient to train, have poor state-space exploration properties, and are inapplicable in high-dimensional action-spaces. To resolve these issues, we combine entropy-based deception with an online actor-critic learning framework. We evaluate our model in discrete and continuous action-space, path-planning environments. Our model is more deceptive than an honest agent in both domains, extending entropy-based deception to high-dimensional action-spaces. The online learning framework leads to targeted exploration of the state-space. This improves the deceptive policy and reduces the number of environment interactions required to train the agent.

## Example agent behaviour gifs
Environment |Honest Agent                                     | Online Actor-Critic Ambiguity Agent
-----------:|:-----------------------------------------------:|:----------------------------------------------:
Discrete    | ![Alt Text](/assets/honest_discrete_map_16.gif) | ![Alt Text](/assets/ambiguity_discrete_map_16.gif)
Continuous  | 

## References
We used openAI's spinningup as the starting point for our subagent model. See the code-base here:

https://github.com/openai/spinningup

## Running

## Notice
I am still currently in the process of cleaning up some of the code and merging in changes from private repositories for publication. If anything appears to be strange, please let me know.
