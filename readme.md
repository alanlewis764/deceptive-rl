# deceptive-rl
This code accompanies the paper - Deceptive Reinforcement Learning: A Policy-Based Approach for High-Dimensional Action-Spaces. 

The paper investigates a policy-based approach to deceptive reinforcement learning for privacy preservation in high-dimensional action-spaces. In reinforcement learning, the reward function defines the agent's objective. In adversarial scenarios, an agent may want to both maximise rewards and keep it's reward function private from observers. Recent approaches use the concept of entropy to select actions which are ambiguous over a set of possible reward functions. Despite promising results, they are inefficient to train, have poor state-space exploration properties, and are inapplicable in high-dimensional action-spaces. To resolve these issues, we combine entropy-based deception with an online actor-critic learning framework. We evaluate our model in discrete and continuous action-space, path-planning environments. Our model is more deceptive than an honest agent in both domains, extending entropy-based deception to high-dimensional action-spaces. The online learning framework leads to targeted exploration of the state-space. This improves the deceptive policy and reduces the number of environment interactions required to train the agent.

## Installation
```
# create clean conda environment
conda create -n deceptiverl python=3.7
conda activate deceptiverl

# install dependencies
git clone git@github.com:alanlewis764/deceptive-rl.git
cd deceptive-rl
pip install -r requirements.txt
```

## Training the agent
To train the online actor-critic ambiguity and/or pre-trained actor-critic ambiguity use the following commands:
Agent       | Command
-----------:|:------------------------------------------------:
online actor-critic ambiguity | ```python run.py --agent 'online' --map_num 1 --action_space 'discrete'```
pre-trained actor-critic ambiguity | ```python run.py --agent 'pre-trained' --map_num 1 --action_space 'discrete'```
q-ambiguity | ```python run.py --agent 'q' map_num 1 --action_space 'discrete'```
honest | ```python run.py --agent 'honest' map_num 1 --action_space 'discrete'```

## Running the agent
We have added some agents that have already been trained which you can run with the following commands:
Agent       | Command
-----------:|:------------------------------------------------:
online actor-critic ambiguity | ```python run.py --agent 'online' --map_num 1 --action_space 'discrete'```
pre-trained actor-critic ambiguity | ```python run.py --agent 'pre-trained' --map_num 1 --action_space 'discrete'```

To run the continuous action-space environment change 'discrete' to 'continuous'.

Note: we only included already trained agents files for the first map as an example.


## References
- We used OpenAI's [Spinningup](https://github.com/openai/spinningup) as the starting point for our subagent model.
- We used [Gym MiniGrid](https://github.com/maximecb/gym-minigrid) as a base to create our discrete action-space environment.
- We used [Gym Extensions](https://github.com/Breakend/gym-extensions ) as a base to create our continuous action-space environment.

## Running

## Example agent behaviour gifs
Environment |Honest Agent                                      | Online Actor-Critic Ambiguity Agent
-----------:|:------------------------------------------------:|:----------------------------------------------:
Discrete    | ![Alt Text](/assets/honest_discrete_map_16.gif)  | ![Alt Text](/assets/ambiguity_discrete_map_16.gif)
Continuous  | ![Alt Text](/assets/honest_continuous_map_3.gif) | ![Alt Text](/assets/ambiguity_continuous_map_3.gif)

## Notice
I am still currently in the process of cleaning up some of the code and merging in changes from private repositories for publication. If anything appears to be strange, please let me know.
