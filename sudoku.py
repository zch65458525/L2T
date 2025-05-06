import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.optim as optim
from torch.distributions import Normal
from st import SentenceEncoder
from llm import LLM
import copy
import re
import pandas as pd
from chatgpt import Chatgpt
import math
llm=Chatgpt()#LLM()
encoder=SentenceEncoder('test')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
prompt="""
solve the 5*5 sudoku puzzle where * represents a cell to be filled in.The requirement of Sudoku is that each row and each column should not have duplicate numbers, and the numbers should range from 1 to 5.
Based on the interaction history, determine the appropriate action to take:
(0) The current thought is incorrect and should be terminated.give the reason(just terminate,don't give another plan)
(1) The problem has been successfully solved.
(2) Design  and execute the next step of the solution plan by breaking the problem into multiple steps and provide only the next step here.Don't repeat what have done in history.
output example:
{{Operation:(2)
Contnet:
Input:[[*, 2, *, 1, 4], [2, *, 4, *, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]}}
{{Operation:(0)
Contnet:
Input:[[*, 2, 2, 1, 4], [2, *, 4, *, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:Stop,Duplicate numbers appear.}}
{{Operation:(0)
Contnet:
Input:[[*, 2, 6, 1, 4], [2, *, 4, *, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:Stop,The number '6' is out of range.}}
{{Operation:(1)
Contnet:
Iutput:[[5, 2, 3, 1, 4], [2, 3, 4, 5, 1], [4, 1, 5, 3, 2], [3, 4, 1, 2, 5], [1, 5, 2, 4, 3]]
Output:Successfully achieved the goal of sorting the array in ascending order..
}}
Determine which operation should be selected for the next step.
"""
prompt1="""Determine which operation should be selected for the next step.generate {input1} different answer!the input sequence now is {input}
just give one step plan.provide different plans through reasoning, without providing the reasoning process.
each answer should be in "{{}}".Output according to the format in the system prompt.The number of answer is {input1}.
Example:
{{Operation:(2)
Content:answer1}},
{{Operation:(2)
Content:answer2}},
...
{{Operation:(2)
Content:answern}},
"""
value_prompt="""Evaluate the plan below using {{s}} to score where s is an integer from 0 to 10,only show the score number,number should in{{}}.
'*'means the empty cell to fill in.The format of the Sudoku is a 5x5 two-dimensional matrix.
The rule that needs to be checked is that there should be no duplicate numbers in the same row or column in Sudoku.
example:5 in the second row,third column,and another 5 in the fourth row,third column.there are duplicate numbers in the same column.
If the number filled in reduces the number of possible choices for other empty cells, it scores higher.
show the score.
output example:
score:{{5}}
"""
value_prompt1="""plan input:{input}
"""
# and the reason
'''also examine if there is crucial mistake.if there is a crucial mistake,set the {{s}} to be 0
mistake example:
{{Operation:(2)
Contnet:input: 12,12
output: 12*2=24 ((left: 24))}}(mistake reason:using the number 2, which is not present in the input.)
score:[0]
'''
value_prompt2="""
The current goal is to fill a sudoku puzzle.'*'means the empty cell to fill in.The format of the Sudoku is a 5x5 two-dimensional matrix.
The rule that needs to be checked is that there should be no duplicate numbers in the same row or column in Sudoku.
Currently, a step plan has been generated, and now we need to evaluate its effectiveness. We need to determine if the remaining part can be filled correctly.
and use the result to evaluate the plan given in Input using |s| to score where s is an integer from 0 to 10,number should in ||
give the score and the reason
example:
{{Input:{{Operation:(2)
Contnet:
Input:[[*, 2, 3, 1, 4], [*, 3, 4, *, 1], [4, 1, 5, 3, 2], [3, 4, 1, 2, 5], [1, 5, 2, 4, 3]]
Output:[[*, 2, 3, 1, 4], [2, 3, 4, *, 1], [4, 1, 5, 3, 2], [3, 4, 1, 2, 5], [1, 5, 2, 4, 3]]}}
[[5, 2, 3, 1, 4], [2, 3, 4, 5, 1], [4, 1, 5, 3, 2], [3, 4, 1, 2, 5], [1, 5, 2, 4, 3]] row 2,column 4 can fill 5,and row 1,column 1 can fill 5
the task can be achieved
score:|10|
}}
{{Input:{{Operation:(2)
Contnet:
Input:[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:[[*, 2, 2, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]}}
duplicate 2 in row 1
score:|0|
}}
{{Input:{{Operation:(2)
Contnet:
Input:[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:[[*, 2, 5, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]}}
duplicate 5 in column 3
score:|0|
}}
{{
Input:{{Operation:(2)
Contnet:
Input:[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, *, *, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]]
Output:[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, *, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]]}}
[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, 1, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]] duplicate in column
[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, 2, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]] duplicate in row
[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, 3, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]] duplicate in row
[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, 4, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]] duplicate in row
[[*, 2, 3, 1, 4], [2, *, 4, 5, 1], [4, 5, 5, 3, 2], [3, 4, 1, 2, 5], [1, *, 2, 4, 3]] duplicate in row
score:|0|
impossible to reach the goal
}}
{{Input:{{Operation:(2)
Contnet:
Input:[[*, 2, *, 1, 4], [2, *, 4, *, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]
Output:[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]]}}
[[5, 2, 3, 1, 4], [2, 3, 4, 5, 1], [*, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]] 
[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [4, 1, 5, 3, 2], [*, 4, *, 2, *], [*, 5, *, 4, *]]
[[*, 2, *, 1, 4], [2, 3, 4, 5, 1], [3, 1, 5, 3, *], [*, 4, *, 2, *], [*, 5, *, 4, *]] duplicate in row
cannot achieve the goal now, but can generate several feasible plan.
score:|7|
}}

Input:{input}
"""
early_stop=0
success_num=0
class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=1e-4):
        super(CustomGCNConv, self).__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.alpha * x_j

    def update(self, aggr_out, x):
        return self.linear(aggr_out+x)
prompt0=prompt.format()
class ActorCriticNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_actions):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = CustomGCNConv(in_channels, hidden_channels)
        self.fc1=nn.Linear(in_channels,hidden_channels)
        self.mu_layer = nn.Linear(hidden_channels, num_actions)
        self.log_std_layer = nn.Linear(hidden_channels, num_actions)
        self.critic = nn.Linear(hidden_channels, 1)
            
    def forward(self, data, target_node_index):
        x = F.relu(self.conv1(data.x,data.edge_index))
        x = x[target_node_index]
        
        mu = torch.sigmoid(self.mu_layer(x))
        std=torch.tensor(1e-5).to(device)
        state_value = self.critic(x)
        return mu,std, state_value
class SimpleEnv:
    def __init__(self,initial_graph,initial_states,leafnode):
        self.initial_graph=initial_graph
        self.initial_states=initial_states
        self.initial_leafnode=leafnode
        self.reset()
    def reset(self):
        self.graph_data=Data(x=self.initial_graph.x, edge_index=self.initial_graph.edge_index).to(device)
        self.states=self.initial_states.copy()
        self.leafnode=self.initial_leafnode.copy()
        self.target_node_index = self.leafnode.pop(0)
        self.done = False
        self.steps = 0
        return self.graph_data, self.target_node_index
    def step(self, action_lst,mu):
        global success_num
        new_edge = torch.tensor([[self.target_node_index],
                                 [self.graph_data.x.size(0)]], dtype=torch.long).to(device)
        print("??????????",self.target_node_index)
        prompt=copy.deepcopy(self.states[self.target_node_index]["prompt"])
        curr=self.states[self.target_node_index]["current"]
        redo=0
        flag=0
        while(redo<2):
            pro=copy.deepcopy(prompt)
            if self.steps<=10:
                pro.append({"role": "user", "content":prompt1.format(input=curr,input1=3)})
            else:
                pro.append({"role": "user", "content":prompt1.format(input=curr,input1=math.ceil(mu[0]*10))})
            res=llm.query(pro,num_responses=1,temperature=mu[1].item(),top_p=mu[2].item())
            res = re.findall(r'\{(.*?)\}', res[0], re.DOTALL)
            value=[]
            for i in range(len(res)):
                if res[i].find('Operation:(0)')>-1:
                    print(res[i])
                    value.append(1)
                    continue
                if res[i].find('Operation:(1)')>-1:
                    self.done=1
                    success_num+=1
                    value.append(10)
                    print(res[i])
                    continue
                print(res[i])
                pro=[{"role": "system", "content":prompt0}]
                pro.append({"role": "user", "content":value_prompt2.format(input=res[i])})
                value_res=llm.query(pro,num_responses=1,max_tokens=4000)[0]
                print("value_res",value_res)
                if value_res.find("|")>-1:
                    sc=value_res.find("|")+1
                    score=int(value_res[sc:value_res.find("|",sc)])
                else:score=0
                value.append(score)
            
            max_index1 = value.index(max(value))
            temp_list = value.copy()
            value_=value.copy()
            temp_list[max_index1] = float('-inf')
            max_index2 = temp_list.index(max(temp_list))
            tmp=[]
            new_node_feat=[]
            es=[]
            et=[]
            tmp=[index for index,val in enumerate(value) if val>=4]
            if len(tmp)>0:
                break
            if redo==1:
                flag=1
            redo+=1
        

        if early_stop==0:
            if len(tmp)==0:
                if len(value)==1:
                    if value[max_index1]>0:
                        tmp=[max_index1]
                else:
                    if value[max_index1]>0:
                        tmp.append(max_index1)
                    if value[max_index2]>0:
                        tmp.append(max_index2)
            if len(tmp)==1:
                if len(value)>1:
                    if value[max_index2]>0:
                        tmp.append(max_index2)
        else:
            if len(tmp)==0:
                if value[max_index1]>0:
                    tmp=[max_index1]
        reward_lst=[]
        reward=0
        for i in value:
            reward+=0.1*(i-5)
        reward_lst.append([reward])
        print("action",action_lst)
        
        for index,value in enumerate(tmp):
            self.states.append({"prompt":prompt})
            left=res[value].find("Output:[")
            right=res[value].find("]]",left)
            curr=res[value][left+7:right+2]
            self.leafnode.append(self.graph_data.x.size(0)+index)
            
            self.states[self.graph_data.x.size(0)+index]["current"]=curr
            self.states[self.graph_data.x.size(0)+index]["value"]=value_[value]
            rr=res[value].find("Output:")
            new_node_feat.append(encoder.encode(res[value][rr:]).to(device))
            es.append(self.target_node_index)
            et.append(self.graph_data.x.size(0)+index)
            self.states[self.graph_data.x.size(0)+index]["prompt"].append({"role": "assistant", "content":res[value]})
            print("generate",self.graph_data.x.size(0)+index,value_[value])
        if flag==1:
            es.append(self.target_node_index)
            et.append(self.target_node_index)
        if len(tmp)>0:
            new_edge=torch.tensor([es,et]).to(device)
            new_node_feat=torch.stack(new_node_feat).to(device)
            self.graph_data.x = torch.cat([self.graph_data.x, new_node_feat], dim=0)
            self.graph_data.edge_index = torch.cat([self.graph_data.edge_index, new_edge], dim=1)
        target_node=self.target_node_index
        if self.steps>200:
            self.done=1
        if len(self.leafnode)==0:
            self.done=1
        else:
            self.target_node_index=self.leafnode.pop(0)
        self.steps += 1
        next_state = Data(x=self.graph_data.x, edge_index=self.graph_data.edge_index).to(device)
        return next_state, reward_lst, self.done,target_node,self.steps
    
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
        self.target_node_indices = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]
        del self.target_node_indices[:]

clip_param = 0.2
max_grad_norm = 0.5
ppo_epochs = 1

in_channels = 768
hidden_channels = 64
num_actions = 3

dataf=pd.read_csv('24.csv').iloc[900:1000]['Puzzles'].tolist()
dataf=["[[*, 3, 4, *], [4, 1, *, 2], [1, *, *, 3], [*, 2, 1, 4]]"]

for i in dataf:
    
    print(i)
    model = ActorCriticNetwork(in_channels, hidden_channels, num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    num_nodes = 5
    num_features = 3
    num_classes=3
    x0={"prompt":[{"role": "system", "content":prompt}],"current":i,"value":10}
    x=encoder.encode(str(x0)).unsqueeze(0).to(device)
    edge_index = torch.tensor([[0], [0]])
    leafnode=[0]

    initial_data = Data(x=x,edge_index=edge_index).to(device)
    env = SimpleEnv(initial_graph=initial_data,initial_states=[x0],leafnode=leafnode)
    num_updates = 1
    memory = Memory()

    for update in range(num_updates):
        print('\n\nnext:')
        state, target_node_index = env.reset()
        done = False
        num_repeat=1
        while not done:
            data = state
            mu,std, state_value = model(data, target_node_index)
            dist = Normal(mu, std)
            action_lst=[] 
            
            for i in range(num_repeat):
                action_lst.append(mu)
            next_state, reward, done, target_node_index,step = env.step(action_lst,mu)
            if step>=8:
                num_repeat=1
            log_prob=[dist.log_prob(act).sum(dim=-1, keepdim=True) for act in action_lst]
            
            for i in range(len(action_lst)):
                memory.states.append(copy.deepcopy(data))
                memory.actions.append(action_lst[i])
                memory.log_probs.append(log_prob[i])
                memory.rewards.append(torch.tensor(reward[i], dtype=torch.float))
                memory.is_terminals.append(done)
                memory.state_values.append(state_value)
                memory.target_node_indices.append(target_node_index)
            loss_=[]
            min_loss = float('inf')
            no_improvement_count = 0
            complete_count=0
            if step%1==0 and early_stop==0:
                rewards = memory.rewards
                dones = memory.is_terminals
                values = memory.state_values

                returns = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + 0.99 * discounted_reward
                    returns.insert(0, discounted_reward)
                print(returns)
                returns = torch.cat(returns).detach().to(device)
                print(returns.shape)
                values = torch.cat(values).detach().to(device)
                advantages = returns - values

                for _ in range(ppo_epochs):
                    batch_states = memory.states
                    print(memory.actions)
                    batch_actions = memory.actions
                    batch_log_probs = torch.tensor(memory.log_probs).to(device)
                    batch_returns = returns
                    batch_advantages = advantages
                    batch_target_node_indices = memory.target_node_indices
                    new_log_probs = []
                    state_values = []
                    for idx, data in enumerate(batch_states):
                        target_node_idx = batch_target_node_indices[idx]
                        mu,std, state_value = model(data, target_node_idx)
                        dist = Normal(mu, std)
                        new_log_prob = dist.log_prob(batch_actions[idx]).sum(dim=-1, keepdim=True)
                        new_log_probs.append(new_log_prob)
                        state_values.append(state_value)
                    new_log_probs = torch.stack(new_log_probs)
                    state_values = torch.stack(state_values).squeeze()
                    
                    ratios = torch.exp(new_log_probs - batch_log_probs)
                    
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    print(state_values,batch_returns)
                    value_loss = F.mse_loss(state_values.unsqueeze(0), batch_returns)
                    
                    loss = policy_loss + 0.5 * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    print(f'Loss: {loss.item()}')
                    loss_.append(loss.item())
                    if loss.item() < min_loss or step<20:
                        min_loss = loss.item()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    if loss.item()<1:
                        complete_count+=1
                    else:
                        complete_count=0
                    if no_improvement_count>=3 or complete_count>=3:
                        early_stop=1

            memory.clear()
            
            state = copy.deepcopy(next_state)

        if update % 10 == 0:
            print(f'Update {update}, Loss: {loss.item()}')
    print(i,'success')
print("success num",success_num)

