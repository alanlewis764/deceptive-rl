# deceptive-rl
This code accompanies the paper - Deceptive Reinforcement Learning in Model-Free Domains. 

This paper investigates the problem of deceptive reinforcement learning for privacy preservation in model-free and continuous action space domains. In reinforcement learning, the reward function defines the agent's objective. In adversarial scenarios, an agent may need to both maximise rewards and keep its reward function private from observers. Recently, the \emph{ambiguity model} (AM) selects actions that are ambiguous over a set of possible reward functions, via pre-trained $Q$-functions. Despite promising results, AM is inefficient to train, ineffective in model-free domains due to misdirected state space exploration, and inapplicable in continuous action spaces. We propose the \emph{deceptive exploration ambiguity model} (DEAM), which learns using the deceptive policy during training, leading to targeted exploration of the state space. DEAM is also applicable in continuous action spaces. We evaluate DEAM in discrete and continuous action-space path planning environments. DEAM achieves similar performance to an optimal version of AM and outperforms a model-free version of AM in terms of path cost, deceptiveness and training efficiency. These results extend to the continuous domain.

## Installation
```
# create clean conda environment
conda create -n deceptiverl python=3.7
conda activate deceptiverl

# install dependencies
cd deceptive-rl
pip install -r requirements.txt
```

## Training the agent
To train ACA and/or pre-trained ACA use the following commands:
Agent       | Command
-----------:|:------------------------------------------------:
DEAM | ```python train.py --agent 'ACA' --map_num 1 --action_space 'discrete'```
AM | ```python train.py --agent 'pre-trained-ACA' --map_num 1 --action_space 'discrete'```

This will store the pyTorch networks for each subagent in a directory called 'data'.

## Running the agent
We have added some agents that have already been trained which you can run with the following commands:
Agent       | Command
-----------:|:------------------------------------------------:
DEAM | ```python run.py --agent 'ACA' --map_num 1 --action_space 'discrete'```
AM | ```python run.py --agent 'pre-trained-ACA' --map_num 1 --action_space 'discrete'```

To run the continuous action-space environment change 'discrete' to 'continuous'.

Note: we only included already trained agents files for the first map as an example.


## References
- We used OpenAI's [Spinningup](https://github.com/openai/spinningup) as the starting point for our subagent model.
- We used [Gym MiniGrid](https://github.com/maximecb/gym-minigrid) as a base to create our discrete action-space environment.
- We used [Gym Extensions](https://github.com/Breakend/gym-extensions ) as a base to create our continuous action-space environment.

## Running

## Example agent behaviour gifs
Environment |Honest Agent                                      | DEAM
-----------:|:------------------------------------------------:|:----------------------------------------------:
Discrete    | ![Alt Text](/assets/honest_discrete_map_16.gif)  | ![Alt Text](/assets/ambiguity_discrete_map_16.gif)
Continuous  | ![Alt Text](/assets/honest_continuous_map_3.gif) | ![Alt Text](/assets/ambiguity_continuous_map_3.gif)

## Notice
I am still currently in the process of cleaning up some of the code and merging in changes from private repositories for publication. If anything appears to be strange, please let me know.
