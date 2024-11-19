import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
from multiprocessing import Process
# Environment
from rl_env.energy import RenewableEnergyEnv

###############################################################################
# Main func with pytorch
from agent.drqn import Q_net, EpisodeMemory, EpisodeBuffer, train

def run_training(seed,battery_times):
    random.seed(seed)
    np.random.seed(seed)
    
    env = RenewableEnergyEnv(battery_times)
    env.battery_times = battery_times
    device = torch.device("cpu")
    observation_space = env.observation_space
    action_space = env.action_space
    gamma = env.gamma
    log_dir = f'logs/test_{datetime.now().strftime("%m%d%H%M")}_seed_{seed}_battery_{round(battery_times,1)}'
    writer = SummaryWriter(log_dir=log_dir)

    # Set parameters
    batch_size = 1
    learning_rate = 1e-4
    buffer_len = int(100000)
    min_epi_num = 20 # Start moment to train the Q network
    episodes = 800
    print_per_iter = 30
    target_update_period = 4
    eps_start = 0.75
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 1000
    
    # DRQN param
    random_update = False # If you want to do random update instead of sequential update
    lookup_step = 24 * 1# If you want to do random update instead of sequential update
    max_epi_len = 800
    max_epi_step = max_step

    # Create Q functions
    Q = Q_net(observation_space, action_space).to(device)
    Q_target = Q_net(state_space=observation_space,action_space=action_space).to(device)
    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    
    episode_memory = EpisodeMemory(random_update=random_update, 
                                   max_epi_num=100, max_epi_len=600, 
                                   batch_size=batch_size, 
                                   lookup_step=lookup_step)
    
    reward_list = []
    q_value_list = []
    solar_generation_data = []
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
    bidding_history = []  # Initialize bidding_history
    scaling_history = []  # Initialize scaling_history
    soc_history = []
    step_rewards_list = [] 
    Pc_t_history = []
    Pd_t_history = []
    penalty_history = []
    battery_penalty_history = []
    xD_t_history = []
    
    # Train
    for i in range(episodes):
        obs = env.reset()
        
        env.scaling = 1.0
        env.bidding = 0.0
        done = False
 
        episode_bidding = []  # List to collect bidding data
        episode_scaling = []  # List to collect scaling data
        episode_step_rewards = [] 
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=batch_size, training=False)
        episode_soc_history = [] 
        episode_reward = 0
        episode_reward_discount = 0
        episode_q_values = [] 
        episode_Pc_t_history = []
        episode_Pd_t_history = []
        episode_penalty_history = []
        episode_battery_penalty_history = []
        episode_xD_t_history = []
        
        for t in range(max_step):

            # Get action
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                              h.to(device), c.to(device),
                                              epsilon)

            # Do action
            s_prime, r, done = env.step(a)
            obs_prime = s_prime
            episode_bidding.append(env.bidding)  # Collect bidding data
            episode_scaling.append(env.scaling)  # Collect scaling data
            episode_soc_history.append(env.soc)
            episode_step_rewards.append(r)
            episode_Pc_t_history.append(env.Pc_t)
            episode_Pd_t_history.append(env.Pd_t)
            episode_penalty_history.append(env.penalty)
            episode_battery_penalty_history.append(env.battery_penalty)
            episode_xD_t_history.append(env.xD_t)
            
            with torch.no_grad():
                q_values, _, _ = Q(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), h, c)
                max_q_value = q_values.max().item()
                episode_q_values.append(max_q_value)
                
                if t == 0:
                   initial_q_value = max_q_value
            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r, obs_prime, done_mask])

            obs = obs_prime
            
            episode_reward += r
            
            episode_reward_discount = r + gamma*episode_reward_discount
            
            if len(episode_memory) >= min_epi_num:
                train(Q, Q_target, episode_memory, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        epsilon = max(eps_end, epsilon * eps_decay) # Linear annealing
        
        episode_memory.put(episode_record)

        bidding_history.append(episode_bidding)  # Save episode bidding data
        scaling_history.append(episode_scaling)  # Save episode scaling data
        step_rewards_list.append(episode_step_rewards)
        soc_history.append(episode_soc_history)  # Extend the global soc_history list
        Pc_t_history.append(episode_Pc_t_history)
        Pd_t_history.append(episode_Pd_t_history)
        penalty_history.append(episode_penalty_history)
        battery_penalty_history.append(episode_battery_penalty_history)
        xD_t_history.append(episode_xD_t_history)

        
        reward_list.append(episode_reward)
        q_value_list.append((episode_q_values))
        
        print(f"Episode {i + 1}: Reward : {episode_reward}")
        # print(f"Episode {i + 1}: Reward_discount : {episode_reward_discount}")
        
        # Log the reward
        writer.add_scalar('Initial_Q_value', initial_q_value, i)
        writer.add_scalar('Rewards', episode_reward, i)
        writer.add_scalar('Q_value', max_q_value, i)
        writer.add_scalar('discount_Rewards', episode_reward_discount, i)
        
    writer.close()
    
    # Save step rewards to a CSV file
    step_rewards_df = pd.DataFrame(step_rewards_list)
    step_rewards_df.to_csv(f'action/step_rewards_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    # torch.save(Q.state_dict(),f'Q_net/Q_net_{current_datetime}.pth')
    # torch.save(Q_target.state_dict(),f'Q_net/Q_target_net_{current_datetime}.pth')
    
    bidding_df = pd.DataFrame(bidding_history)
    scaling_df = pd.DataFrame(scaling_history)
    soc_df = pd.DataFrame(soc_history)
    Pc_df = pd.DataFrame(Pc_t_history)
    Pd_df = pd.DataFrame(Pd_t_history)
    penalty_df = pd.DataFrame(penalty_history)
    battery_penalty_df = pd.DataFrame(battery_penalty_history)
    xD_t_history_df =pd.DataFrame(xD_t_history)
    
    # Save each DataFrame to CSV files
    bidding_df.to_csv(f'action/episode_bidding_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    scaling_df.to_csv(f'action/episode_scaling_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    soc_df.to_csv(f'action/episode_soc_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    Pc_df.to_csv(f'action/episode_Pc_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    Pd_df.to_csv(f'action/episode_Pd_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    penalty_df.to_csv(f'action/episode_penalty_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    battery_penalty_df.to_csv(f'action/episode_battery_penalty_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)
    xD_t_history_df.to_csv(f'action/episode_battery_xDt_{current_datetime}_seed_{seed}_battery_{round(battery_times,1)}.csv', index=False)

if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]  

    battery_times = 0.1  # 初期battery_times
    battery_times_max = 1.0 # battery_timesの上限
    
    while battery_times <= battery_times_max:
        processes = []
        for seed in seeds:
            p = Process(target=run_training, args=(seed,battery_times))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        battery_times += 0.3