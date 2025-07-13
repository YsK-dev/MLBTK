# %%
# Q-Learning with Gymnasium Environment
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Use gymnasium instead of gym
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')

# Rest of your code remains the same
state, info = env.reset()

nb_states = env.observation_space.n
nb_actions = env.action_space.n
q_table = np.zeros((nb_states, nb_actions))

print("Number of states:", nb_states)
print("Number of actions:", nb_actions) 
print("Q-table shape:", q_table.shape)

episodes = 100
max_steps = 10
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01

outcomes = []
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    for step in range(max_steps):
        if random.uniform(0, 1) < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Handle both termination conditions
        total_reward += reward
        
        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += learning_rate * td_error
        
        state = next_state
        
        if done:
            break
    
    outcomes.append(total_reward)
    
    # Decay exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Print average reward over the last 100 episodes
average_reward = np.mean(outcomes[-100:])
print("Average reward over the last 100 episodes:", average_reward)
# Close the environment
env.close() 

print("Q table after training:")
print(q_table)  

plt.bar(range(len(episode)), outcomes)

print("Success rate:", np.mean(np.array(outcomes) > 0.0))
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Performance on FrozenLake')
plt.show()
# %%
