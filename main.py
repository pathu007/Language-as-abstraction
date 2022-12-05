import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import argparse

from clevr_robot_env.env import ClevrEnv
from networks import DQN, HDQN, Encoder
from replay_buffer import ReplayBuffer
from util import *

import csv  
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPLAY_BUFFER_SIZE = 2e6
BATCH_SIZE = 32
CYCLE = 1
EPOCH = 1
EPISODES = 100
STEPS = 100
UPDATE_STEPS = 100

class DoubleDQN(nn.Module):
    def __init__(self, env, tau=0.05, gamma=0.9, epsilon=1.0):
        super().__init__()
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.embedding_size = 64
        self.hidden_size = 64
        self.obs_shape = self.env.get_obs().shape
        self.action_shape = 40 // 5
        self.encoder = Encoder(self.embedding_size, self.hidden_size).to(DEVICE)

        self.model = DQN(self.obs_shape, self.action_shape, self.encoder).to(DEVICE)
        self.target_model = DQN(self.obs_shape, self.action_shape, self.encoder).to(DEVICE)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epsilon = epsilon

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)


    def get_action(self, state, goal):
        assert len(state.shape) == 2 # This function should not be called during update

        if(np.random.rand() > self.epsilon):
            q_values = self.model.forward(state, goal)
            idx = torch.argmax(q_values).detach()
            obj_selection = idx // 8
            direction_selection = idx % 8
        else:
            action = self.env.sample_random_action()
            obj_selection = action[0]
            direction_selection = action[1]

        return int(obj_selection), int(direction_selection)

        
    def compute_loss(self, batch):     
        states, actions, goals, rewards, next_states, satisfied_goals, dones = batch

        rewards = torch.FloatTensor(rewards).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        curr_Q = self.model(states, goals) 

        curr_Q_prev_actions = [curr_Q[batch, actions[batch][0], actions[batch][1]] for batch in range(len(states))] # TODO: Use pytorch gather
        curr_Q_prev_actions = torch.stack(curr_Q_prev_actions)

        next_Q = self.target_model(next_states, goals) 
        
        next_Q_max_actions = torch.max(next_Q, -1).values
        next_Q_max_actions = torch.max(next_Q_max_actions, -1).values

        next_Q_max_actions = rewards + (1 - dones) * self.gamma * next_Q_max_actions

        loss = F.mse_loss(curr_Q_prev_actions, next_Q_max_actions.detach())

        return loss

    def update(self, replay_buffer, batch_size):
        for _ in range(UPDATE_STEPS):
            batch = replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_net(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def train(agent,env,state,instruction):
        replay_buffer_agent = ReplayBuffer(REPLAY_BUFFER_SIZE)

    #with open('episodic_log_k_6.csv', 'w') as csvfile:
        #csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(["Epoch","Cycle","Cycle_reward","Episode","Reward","Achieved_Goals"])

    #with open('instructions_k_6.csv', 'w') as csvfile_1:
        #csvwriter_1 = csv.writer(csvfile_1)
        #csvwriter_1.writerow(["Epoch","Cycle","Cycle_reward","Episode","Goal_Achieved","Steps"])


        #agent.train()
        #state = env.reset()
        #goal, goal_program = env.sample_goal()
        #env.set_goal(instruction)
        episode_reward = 0
        steps = 0
        
        #no_of_achieved_goals = 0
        #current_instruction_steps = 0
        trajectory = []

        for step in range(STEPS):
            action = agent.get_action(state, instruction)
            next_state, reward, done, _ = env.step(action, record_achieved_goal=True)
            transition = Transition(state, action, instruction, reward, next_state, [], done)
            trajectory.append(transition)

            episode_reward += reward

            if reward == 1.0:
                steps = step
            #with open('instructions.csv', 'a') as csvfile_1:
                #csvwriter_1 = csv.writer(csvfile_1)
                #csvwriter_1.writerow([str(epoch),str(cycle),str(cycle_reward),str(episode),goal,str(step)])
            #goal, goal_program = env.sample_goal()
            #env.set_goal(goal, goal_program)
                #no_of_achieved_goals += 1
            #current_instruction_steps = 0
                break
            if done:
                break

            #if current_instruction_steps == 10: # Early stop if stuck
                #break
                    
                current_instruction_steps += 1
                state = next_state

                # Hindsight Instruction Relabeling (HIR)
        for step in range(len(trajectory)): # 
            replay_buffer_agent.add(trajectory[step])
            for goal_prime in trajectory[step].satisfied_goals_t:
                transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, 1.0, trajectory[step].next_state, [], trajectory[step].done)
                replay_buffer_agent.add(transition)
                deltas = future_instruction_relabeling_strategy(trajectory, step, 0, 0.9)
                for delta in deltas:
                    goal_prime, reward_prime = delta
                    transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, reward_prime, trajectory[step].next_state, [], trajectory[step].done)
                    replay_buffer_agent.add(transition)  

            

        #print("[Episode] " + str(episode) + ": Reward " + str(episode_reward) + " Achieved Goals: " + str(no_of_achieved_goals))
                
        #with open('episodic_log.csv', 'a') as csvfile:
            #csvwriter = csv.writer(csvfile)
            #csvwriter.writerow([str(epoch),str(cycle),str(cycle_reward),str(episode),str(episode_reward),str(no_of_achieved_goals)])


        agent.update(replay_buffer_agent, BATCH_SIZE)   
        return next_state,episode_reward,steps
        #print("[Cycle] " + str(cycle) + ": Total Reward " + str(cycle_reward))

        #if cycle > 9:
            #agent.epsilon *= 0.993
            #if agent.epsilon < 0.1:
                #agent.epsilon = 0.1

if __name__ == "__main__":
    env = ClevrEnv(action_type="perfect", obs_type='order_invariant', direct_obs=True, use_subset_instruction=True)
    
    
    
    state = env.reset()
    #action = env.sample_random_action()
    goal,goal_p = env.sample_goal()

    
    description, full_description = env.get_description()
    print(description[0].split("? "))
    instruction_dict = []
    for i in range(len(description)):
        
        #print(description[i].split("? ")[0])
        instruction_dict.append(description[i].split("? ")[0] + "?")

    print(instruction_dict)
    
    #print(action)
    #obj_selection = action[0]
    #direction_selection = action[1]
    #action = int(obj_selection), int(direction_selection)
    #next_state, reward, done, _ = env.step(action, record_achieved_goal=True)

    #print(next_state)
    #print(reward)

    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #model = AutoModelForCausalLM.from_pretrained("gpt2")
    #observation = next_state
    #print(len(next_state[0]))
    #prompt = "There is a red rubber sphere.Generate a question to verify it."

    #input = tokenizer(goal, return_tensors="pt")
    #print(input)

    #input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    #imput_ids = input.input_ids

    #outputs = model.generate(input_ids, do_sample=False, max_length=30)
    #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    #obs_shape = env.get_obs().shape
    #hrl = HDQN(obs_shape, 64)

    #q_values = hrl.forward(state)
    #idx = torch.sum(q_values, dim=1)
    #print(idx)
    #outputs = model.generate(idx, do_sample=False, max_length=64)
    #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    #outputs = model.generate(next_state, do_sample=False, max_length=30)
    
    
    agent = DoubleDQN(env)
    q_values = agent.model.forward(state, goal)
    print(q_values )
    print(torch.argmax(q_values))
    print(torch.argmax(q_values).detach())
    print(torch.argmax(q_values).detach()//8,torch.argmax(q_values).detach()%8)
    #train(env, agent)
    
    
    #goal, goal_program = env.sample_goal()
    #description, full_description = env.get_description()
    #true_desc = []
    #print(len(description))
    #for i in description:
        #print(i.split("?"))
        #print("\n")
        #if i.split("?")[1] == " True":
            #print(i.split("?")[0].split(";")[0])
            #true_desc.append(i.split("?")[0].split(";")[0])
    #print(full_description)
    #print(goal,goal_program)
    #print(true_desc)