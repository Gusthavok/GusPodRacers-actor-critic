import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from math import exp

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward_atq', 'reward_dfs'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize(memory, actor_net, critic_net, critic_smooth_net, optimizer_actor, optimizer_critic, device, GAMMA = 0.95, BATCH_SIZE = 512):
    if len(memory) < BATCH_SIZE:
        return 0, 0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_atq_batch = torch.cat(batch.reward_atq).unsqueeze(1)
    reward_dfs_batch = torch.cat(batch.reward_dfs).unsqueeze(1)
    reward_batch = torch.cat((reward_atq_batch, reward_dfs_batch), dim=1)

    state_action_values = critic_net(action_batch, state_batch)

    next_state_values = torch.zeros((BATCH_SIZE, 2), device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = critic_smooth_net(actor_net(non_final_next_states), non_final_next_states)
    
    #### Critic optimization
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.MSELoss()
    loss_critic = criterion(state_action_values.squeeze(), expected_state_action_values.squeeze())

    # Optimize the model
    optimizer_critic.zero_grad()
    loss_critic.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(critic_net.parameters(), 100)
    optimizer_critic.step()

    #### Actor optimization
    values = critic_smooth_net(actor_net(state_batch), state_batch)
    q_value = values[:, 0] + values[:, 1]
    # Compute loss
    loss_actor = -q_value.mean()

    # Optimize the model
    optimizer_actor.zero_grad()
    loss_actor.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(actor_net.parameters(), 100)
    optimizer_actor.step()

    return loss_actor.clone().detach(), loss_critic.clone().detach()


def optimize_actor(memory, actor_net, critic_net, critic_smooth_net, optimizer_actor, optimizer_critic, device, GAMMA = 0.95, BATCH_SIZE = 512):
    if len(memory) < BATCH_SIZE:
        return 0
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)


    #### Actor optimization
    values = critic_smooth_net(actor_net(state_batch), state_batch)
    q_value = values[:, 0] + values[:, 1]
    
    # Compute loss
    loss_actor = -q_value.mean()

    # Optimize the model
    optimizer_actor.zero_grad()
    loss_actor.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(actor_net.parameters(), 100)
    optimizer_actor.step()   

    
def select_action(state, actor_net, choose_random_action, device, eps_threshold):
    sample_atq = float(random.random() > eps_threshold[0])
    sample_dfs = float(random.random() > eps_threshold[1])
    
    with torch.no_grad():
        actor_action = actor_net(state).squeeze()
    random_action=torch.tensor(choose_random_action(), device=device, dtype=torch.long)
    
    
    return torch.cat(((actor_action*sample_atq + random_action*(1-sample_atq))[:4], 
                      (actor_action*sample_dfs + random_action*(1-sample_dfs))[4:]), 
                     dim=0)
    
def soft_update(policy_net, target_net, TAU=0.005):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)
