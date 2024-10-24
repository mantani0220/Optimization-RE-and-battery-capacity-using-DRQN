import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Environment
from rl_env.codesign_energy_env import CodesignEnergyEnv

###############################################################################
# Main func with pytorch
from agent.codesign_drqn import Q_net, EpisodeMemory, EpisodeBuffer, train

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
  
    log_dir = 'logs/logs'+datetime.now().strftime('%m%d%H%M')
    writer  = SummaryWriter(log_dir=log_dir)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = CodesignEnergyEnv()
    observation_space = env.observation_space
    action_space      = env.action_space
    rho_space         = 1
    
    # battery_price = 1000 # 6000が60000円/kwhに相当
    
    ############################################
    # DRQN setting parameters
    ############################################
    batch_size = 8
    learning_rate = 1e-4
    buffer_len = int(100000)
    min_epi_num  = 20 # Start moment to train the Q network
    episodes     = 5000
    print_per_iter = 30
    target_update_period = 4
    eps_start = 0.5
    eps_end   = 0.001
    eps_decay = 0.995
    tau       = 1e-2
    max_step  = 1000
    gamma     = 0.99

    # DRQN param
    random_update = True  # If you want to do random update instead of sequential update
    lookup_step   = 24 * 1 # If you want to do random update instead of sequential update
    max_epi_len   = 700
    max_epi_step = max_step

    
    ###############################################
    # Codesign learning parameters
    ###############################################
    batch_size_Phi      = 10
    learning_rate_mu    = 1e-5 
    learning_rate_sigma = 0
    min_epi_codesign    = 300

    ##################################################
    # Create Q functions and optimizer
    device = torch.device("cpu")
    Q = Q_net(observation_space, action_space, rho_space).to(device)
    Q_target = Q_net(state_space=observation_space,action_space=action_space,rho_space=rho_space).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start    
    episode_memory = EpisodeMemory(random_update=random_update, 
                                    max_epi_num=100, max_epi_len=600, 
                                    batch_size=batch_size, 
                                    lookup_step=lookup_step)
    
    reward_list = []
    q_value_list = []
    solar_generation_data = []
    rho_list = []
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
    bidding_history = []  # Initialize bidding_history
    scaling_history = []  # Initialize scaling_history
    soc_history = []
    step_rewards_list = [] 
    Pc_t_history = []
    Pd_t_history = []
    penalty_history = []
    
    # Initialize phi = [mu, sigma]
    mu    = 0.1
    sigma = 0.2
    phi   = [mu, sigma]
    # Train
    for i in range(episodes):
        
        # Initialize environment
        rho = max(0.05, np.random.normal(mu,sigma))
        writer.add_scalar('Mu', mu, i)
        obs = env.reset(rho)
        env.scaling = 1.0
        env.bidding = 0.0
        done = False        # Initialize buffer     
        if i <= 300:
            battery_price = 0
        elif i <= 1000:
            battery_price = 6000* (i - 300) / (1000 - 300)
        else:
            battery_price = 6000
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=batch_size, training=False)
         
        episode_bidding = []  # List to collect bidding data
        episode_scaling = []  # List to collect scaling data
        episode_step_rewards = [] 
        episode_soc_history = [] 
        episode_reward = 0
        episode_reward_discount = 0
        episode_q_values = [] 
        episode_Pc_t_history = []
        episode_Pd_t_history = []
        episode_penalty_history = []


        # For each time step i       
        for t in range(max_step):
             
            ##########################################################
            # Simulate one step 
            ##########################################################
            # Get action
            obs_rho = np.concatenate([obs, np.array([rho])])    #[obs, rho].numpy
            obs_rho = torch.from_numpy(obs_rho).float().to(device).unsqueeze(0).unsqueeze(0) #[[[obs,rho]]].tensor
            a, h, c = Q.sample_action(obs_rho, h.to(device), c.to(device),
                                      epsilon, np.array([rho]))

            # Do action
            obs_prime, r, done = env.step(a)
            episode_reward += r            
            episode_reward_discount = r + gamma*episode_reward_discount

            # Store (s, a, r, s') to replay buffer
            done_mask = 0.0 if done else 1.0
            episode_record.put([obs, a, r, obs_prime, done_mask, rho])

            # from k-1 to k
            obs = obs_prime
            
            episode_record = EpisodeBuffer()

            with torch.no_grad():
                q_values, _, _ = Q(obs_rho, h, c)
                max_q_value = q_values.max().item()
                episode_q_values.append(max_q_value)
                
                if t == 0:
                    initial_q_value = max_q_value
            
            ##########################################################
            # Learning step for DRQN
            ##########################################################
            if len(episode_memory) >= min_epi_num:
                train(Q, Q_target, episode_memory, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        rho_space = rho_space)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
        
        # update epsilon
        epsilon = max(eps_end, epsilon * eps_decay) # Linear annealing
        
        # store current episode to "episode memory" 
        episode_memory.put(episode_record)
        
        # データをセーブする
        # bidding_history.append(episode_bidding)  
        # scaling_history.append(episode_scaling) 
        # step_rewards_list.append(episode_step_rewards)
        # soc_history.append(episode_soc_history)  
        # Pc_t_history.append(episode_Pc_t_history)
        # Pd_t_history.append(episode_Pd_t_history)
        # penalty_history.append(episode_penalty_history)
        
        reward_list.append(episode_reward)
        q_value_list.append((episode_q_values))
        rho_list.append(rho)
        
        print(f"Episode {i + 1}: Reward : {episode_reward}")
        print(f"Episode {i + 1}: Reward_discount : {episode_reward_discount}")
        


        # Log the reward
        writer.add_scalar('Initial_Q_value', initial_q_value, i)
        writer.add_scalar('Rewards', episode_reward, i)
        writer.add_scalar('Q_value', max_q_value, i)
        writer.add_scalar('discount_Rewards', episode_reward_discount, i)
        
        

        ###########################################################
        # Update phi
        ###########################################################
        # バッチデータから states と rho を抽出
        
        if i >= min_epi_codesign:
             if i % 10 ==0:
                
                # batch_data, seq_len = episode_memory.sample(random_sample=True, batch_size=batch_size_Phi)
                # statesop = [ batch_data[i]['obs'] for i in range(batch_size_Phi) ]
                # rhos   = [ batch_data[i]['rho'] for i in range(batch_size_Phi) ]
                # rewards = [batch_data[i]['rews'] for i in range(batch_size_Phi)]
                # states = np.array(states, dtype=np.float32)
                # rhos   = np.array(rhos, dtype=np.float32)
                # rewards = np.array(rewards, dtype=np.float32)
       
                # calculate Q-value 
                # h, c = Q.init_hidden_state(batch_size=batch_size_Phi, training=True)
                # state_tensor = torch.tensor(states, dtype=torch.float32)            
                # rhos         = np.reshape(rhos, (batch_size_Phi,lookup_step,1))
                # rho_tensor   = torch.tensor(rhos, dtype=torch.float32)
                # state_rho    = torch.cat([state_tensor, rho_tensor], dim=-1) # Add batch dimension
                # q_out, h, c  = Q.forward(state_rho, h, c)
                # q_max, _     = q_out.max(axis=2)
                # q_max        = q_max.mean(axis=1)
                
                # update mu
                # final_rewards = rewards[:, -1]
                # final_rewards = torch.tensor(final_rewards, dtype=torch.float32)
                rhos    = np.array(rho_list[-10:])
                
                G = np.array(reward_list[-10:])*365/7 - rhos * battery_price
                # G       = q_max - rhos * battery_price  # Profit - battery_capacity * battery_price
                mu_grad =  (((rhos - mu) / (sigma ** 2)) * (G -G.mean())).mean()
                mu      = mu     + learning_rate_mu * mu_grad
    
                # update sigma            
                sigma_grad = 0
                sigma = sigma + learning_rate_sigma * sigma_grad
                #  (rho_tensor - mu) ** 2  - sigma ** 2) / (sigma ** 3)
        
                # release phi = [mu, simga]
                phi = [mu, sigma]
                
    
        
        ####################################################################
       
        
    writer.close()
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")
    # Save step rewards to a CSV file
    # step_rewards_df = pd.DataFrame(step_rewards_list)
    # step_rewards_df.to_csv(f'action/step_rewards_{current_datetime}.csv', index=False)
    # # torch.save(Q.state_dict(),f'Q_net/Q_net_{current_datetime}.pth')
    # # torch.save(Q_target.state_dict(),f'Q_net/Q_target_net_{current_datetime}.pth')
    
    # bidding_df = pd.DataFrame(bidding_history)
    # scaling_df = pd.DataFrame(scaling_history)
    # soc_df = pd.DataFrame(soc_history)
    # Pc_df = pd.DataFrame(Pc_t_history)
    # Pd_df = pd.DataFrame(Pd_t_history)
    # penalty_df = pd.DataFrame(penalty_history)
    
    # # Save each DataFrame to CSV files
    # bidding_df.to_csv(f'action/episode_bidding_{current_datetime}_run_{run_id+1}_.csv', index=False)
    # scaling_df.to_csv(f'action/episode_scaling_{current_datetime}_run_{run_id+1}_.csv', index=False)
    # soc_df.to_csv(f'action/episode_soc_{current_datetime}_run_{run_id+1}_.csv', index=False)
    # Pc_df.to_csv(f'action/episode_Pc_{current_datetime}_run_{run_id+1}_.csv', index=False)
    # Pd_df.to_csv(f'action/episode_Pd_{current_datetime}_run_{run_id+1}_.csv', index=False)
    # penalty_df.to_csv(f'action/episode_penalty_{current_datetime}_run_{run_id+1}_.csv', index=False)
    
    # パラメータを辞書にまとめる
parameters = {
    "drqn_params": {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "buffer_len": buffer_len,
        "min_epi_num": min_epi_num,
        "episodes": episodes,
        "print_per_iter": print_per_iter,
        "target_update_period": target_update_period,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "tau": tau,
        "max_step": max_step,
        "gamma": gamma
    },
    "codesign_params": {
        "batch_size_Phi": batch_size_Phi,
        "learning_rate_mu": learning_rate_mu,
        "min_epi_codesign": min_epi_codesign
    }
}


# 日時を含めたファイル名を設定
filename = f'parameters/parameters_{current_datetime}.npy'

# np.save()で日時入りのファイル名に保存
np.save(filename, parameters)