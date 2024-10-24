import torch.optim as optim
import ray
import random
import time
import torch
import numpy as np
from collections import deque
from agent.codesign_drqn import Q_net
from rl_env.codesign_energy_year_env import AllCodesignEnergyEnv
from torch.distributions.normal import Normal
from torch.distributions import Categorical

# Actor Class
class Actor:
    def __init__(self, num, algorithm, writer, device, state_dim, action_dim, agent_args, epsilon=None):
        self.num = num
        self.args = agent_args
        if epsilon is not None:
           self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
        else:
           self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args).to(device)
    
    def get_algorithm(self):
        return self.algorithm
    
    def get_weights(self):
        return self.algorithm.get_weights()
    
    def set_weights(self, weights):
        self.algorithm.set_weights(weights)

    def run(self, ps, global_buffer, epochs):
        env = AllCodesignEnergyEnv()
        print("actor start")
        i = 0
        while True:
            if i % self.args['actor_update_cycle'] == 0:
                weights = ray.get(ps.pull.remote())
                self.algorithm.set_weights(weights)
            run_env(env, self.algorithm, self.algorithm.device, self.args['traj_length'], True)
            data = self.algorithm.get_trajectories()
            td_error = self.algorithm.get_td_error(data)
            data['priority'] = td_error.detach().cpu().numpy()
            global_buffer.put_trajectories.remote(data)
            i += 1
        print('actor finish')
      
# Learner Class
class Learner:
    def __init__(self, algorithm, writer, device, state_dim, action_dim, agent_args):
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args).to(device)
        self.optimizer = optim.Adam(self.algorithm.parameters(), lr = self.args['lr'])
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_weights(self):
        return self.algorithm.get_weights()
    
#Prioritized Experience Replay    
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx]) 

# Memoly for Experience Buffer
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)    
        
@ray.remote
class ApexBuffer:
    def __init__(self, learner_memory_size, state_dim, num_action):
        self.append_buffer = deque(maxlen = learner_memory_size)
        self.update_buffer = deque(maxlen = learner_memory_size)
        self.buffer = Memory(learner_memory_size)
        self.max_iter = 50
        
    def put_trajectories(self, data):
        self.append_buffer.append(data)
        
    def put_idxs(self,idxs):
        self.update_buffer.append(idxs)
        
    def get_append_buffer(self):
        return self.append_buffer
    
    def get_update_buffer(self):
        return self.update_buffer
    
    def get_buffer(self):
        return self.buffer
    
    def sample(self,batch_size):
        return self.buffer.sample(batch_size)
    
    def stack_data(self):
        size = min(len(self.append_buffer), self.max_iter)
        data = [self.append_buffer.popleft() for _ in range(size)]
        for i in range(size):
            priority, state, action, reward, next_state, done = \
            data[i]['priority'], data[i]['state'], data[i]['action'], data[i]['reward'], data[i]['next_state'], data[i]['done']
            for j in range(len(data[i])):
                self.buffer.add(priority[j].item(), [state[j], action[j], reward[j], next_state[j], done[j]])
                
    def update_idxs(self):    
        size = min(len(self.update_buffer), self.max_iter)
        data = [self.update_buffer.popleft() for _ in range(size)]
        for i in range(size):
            idxs, td_errors = data[i]
            for j in range(len(idxs)):
                self.buffer.update(idxs[j], td_errors[j].item())

# APEXLearner Class
class APEXLearner(Learner):
    def __init__(self, algorithm, writer, device, state_dim, action_dim, agent_args, epsilon):
        self.args = agent_args
        self.algorithm = algorithm(writer, device, state_dim, action_dim, agent_args, epsilon).to(device)
    def run(self, ps, buffer):
        data = ray.get(buffer.sample.remote(self.args['learner_batch_size']))
        idx, td_error = self.algorithm.train_network(data)
        ray.wait([ps.push.remote(self.get_weights())])
        ray.wait([buffer.put_idxs.remote([idx, td_error])])
        
def make_transition(state, action, reward, next_state, done, log_prob):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['done'] = done
    transition['log_prob'] = log_prob
    return transition  
   
def run_env(env, algorithm, device, traj_length=0, get_traj=False, reward_scaling=0.1):
    score = 0
    if traj_length == 0:
        traj_length = env._max_episode_steps
    state = env.state if env.can_run else env.reset()
    for t in range(traj_length):
        if algorithm.args['value_based']:
            action = algorithm.get_action(torch.from_numpy(state).float().to(device))
            log_prob = np.zeros((1, 1))
        else:
            if algorithm.args['discrete']:
                prob = algorithm.get_action(torch.from_numpy(state).float().to(device))
                dist = Categorical(prob)
                action = dist.sample()
                log_prob = torch.log(prob.reshape(1, -1).gather(1, action.reshape(1, -1))).detach().cpu().numpy()
                action = action.item()
            else:
                mu, std = algorithm.get_action(torch.from_numpy(state).float().to(device))
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).detach().cpu().numpy()
                action = action.clamp(-1, 1).detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)
        next_state = np.zeros_like(state) if done else next_state
        transition = make_transition(state, action, reward * reward_scaling, next_state, done, log_prob)
        algorithm.append_sample(transition)
        state = next_state
        score += reward
        if done:
            break
    env.time_steps += 1
    return score

def run_setting(num):
    if num == 0:
        return 1
    elif num == 1:
        return 12
    else:
        return num - 1
    
@ray.remote
class ParameterServer:
    def __init__(self, weights):
        self.weights = weights
        
    def push(self, weights):
        self.weights = weights
        
    def pull(self):
        return self.weights   
    
class TestAgent:
    def __init__(self, args, state_dim, action_dim, writer, algorithm, epsilon):
        self.args = args
        self.device = 'cpu'
        self.agent = Actor(0, algorithm, writer, 'cpu', state_dim, action_dim, args, epsilon)
    
    def test(self, epoch, run_num, total_epochs):
        test_rewards = []
        for _ in range(run_num):
            test_env = AllCodesignEnergyEnv()
            score = run_env(test_env, self.agent.algorithm, 'cpu', 0)
            test_rewards.append(score)
            time.sleep(1)
        return np.mean(test_rewards)
    
def run(buffer, ps, num):
    epsilon = run_setting(num)
    agent = APEXLearner(Q_net, None, 'cuda', 365, 15, {'gamma': 0.99, 'lr': 0.0001}, epsilon)
    while True:
        agent.run(ps, buffer)
        time.sleep(0.01)
        
def buffer_run(buffer):
    print('buffer_start')
    while 1:
        ray.wait([buffer.stack_data.remote()])
        #synchronize issue check
        #lock it if learner added data to buffer
        ray.wait([buffer.update_idxs.remote()])
        time.sleep(0.1)
    print("buffer finished")