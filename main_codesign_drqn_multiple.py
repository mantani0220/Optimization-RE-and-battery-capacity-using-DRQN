import concurrent.futures
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
from multiprocessing import Process
# Environment
from rl_env.codesign_energy_env import CodesignEnergyEnv

###############################################################################
# Main func with pytorch
from agent.codesign_drqn import Q_net, EpisodeMemory, EpisodeBuffer, train

# 関数を定義して、その中でrun_idごとの処理を行う
def run_training(seed, battery_price):
    random.seed(seed)
    np.random.seed(seed)
    
    log_dir = f'codesign_logs/logs_{datetime.now().strftime("%m%d%H%M")}_seed_{seed}_price_{battery_price}'
    writer = SummaryWriter(log_dir=log_dir)
    
    ###########################################    
    # Environment setting
    ##########################################
    env = CodesignEnergyEnv()
    observation_space = env.observation_space
    action_space = env.action_space
    rho_space = 1
    # battery_price = 3000#要検討
    
    ############################################
    # DRQN setting parameters
    ############################################
    batch_size = 8
    learning_rate = 1e-4
    buffer_len = int(100000)
    min_epi_num = 20  # Start moment to train the Q network
    episodes = 5000
    print_per_iter = 30
    target_update_period = 4
    eps_start = 0.5
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 1000
    gamma = 0.9

    # DRQN param
    random_update = True  # If you want to do random update instead of sequential update
    lookup_step = 24 * 1  # If you want to do random update instead of sequential update
    max_epi_len = 700
    max_epi_step = max_step

    ###############################################    
    # Codesign learning parameters
    ###############################################    
    batch_size_Phi = 10
    learning_rate_mu = 1e-6
    learning_rate_sigma = 0
    min_epi_codesign = 300

    ##################################################
    # Create Q functions and optimizer
    device = torch.device("cpu")
    Q = Q_net(observation_space, action_space, rho_space).to(device)
    Q_target = Q_net(state_space=observation_space, action_space=action_space, rho_space=rho_space).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start    
    episode_memory = EpisodeMemory(random_update=random_update, max_epi_num=100, max_epi_len=600, batch_size=batch_size, lookup_step=lookup_step)

    reward_list = []
    q_value_list = []
    rho_list = []
    # solar_generation_data = []
    # current_datetime = datetime.now().strftime('%Y%m%d_%H%M')
    # bidding_history = []  # Initialize bidding_history
    # scaling_history = []  # Initialize scaling_history
    # soc_history = []
    # step_rewards_list = [] 
    # Pc_t_history = []
    # Pd_t_history = []
    # penalty_history = []
    
    # Initialize phi = [mu, sigma]
    mu = 0.1
    sigma = 0.2
    phi = [mu, sigma]

    # Train
    for i in range(episodes):
        rho = max(0.01, np.random.normal(mu, sigma))
        print('Mu',mu)
        writer.add_scalar('Mu', mu, i)
        obs = env.reset(rho)
        env.scaling = 1.0
        env.bidding = 0.0
        done = False

        # Initialize buffer        
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
            obs_rho = np.concatenate([obs, np.array([rho])])
            obs_rho = torch.from_numpy(obs_rho).float().to(device).unsqueeze(0).unsqueeze(0)
            a, h, c = Q.sample_action(obs_rho, h.to(device), c.to(device), epsilon, np.array([rho]))

            # Do action
            obs_prime, r, done = env.step(a)
            episode_reward += r            
            episode_reward_discount = r + gamma * episode_reward_discount

            # Store (s, a, r, s') to replay buffer
            done_mask = 0.0 if done else 1.0
            episode_record.put([obs, a, r, obs_prime, done_mask, rho])

            obs = obs_prime
            
            # Record sampled data
            episode_bidding.append(env.bidding)  # Collect bidding data
            episode_scaling.append(env.scaling)  # Collect scaling data
            episode_soc_history.append(env.soc)
            episode_step_rewards.append(r)
            episode_Pc_t_history.append(env.Pc_t)
            episode_Pd_t_history.append(env.Pd_t)
            episode_penalty_history.append(env.penalty)

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
                train(Q, Q_target, episode_memory, device, optimizer=optimizer, batch_size=batch_size, learning_rate=learning_rate, rho_space=rho_space)

                if (t + 1) % target_update_period == 0:
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):
                           target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                
            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)
        episode_memory.put(episode_record)
        
                
        reward_list.append(episode_reward)
        q_value_list.append(episode_q_values)
        rho_list.append(rho)
        
        print(f"Episode {i + 1}: Reward : {episode_reward}")
        print(f"Episode {i + 1}: Reward_discount : {episode_reward_discount}")

        # Log the reward
        writer.add_scalar('Initial_Q_value', initial_q_value, i)
        writer.add_scalar('Rewards', episode_reward, i)
        writer.add_scalar('Q_value', max_q_value, i)
        writer.add_scalar('discount_Rewards', episode_reward_discount, i)
        

        # Update phi
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

    writer.close()

    # parameters = {
    #     "drqn_params": {
    #         "batch_size": batch_size,
    #         "learning_rate": learning_rate,
    #         "buffer_len": buffer_len,
    #         "min_epi_num": min_epi_num,
    #         "episodes": episodes,
    #         "print_per_iter": print_per_iter,
    #         "target_update_period": target_update_period,
    #         "eps_start": eps_start,
    #         "eps_end": eps_end,
    #         "eps_decay": eps_decay,
    #         "tau": tau,
    #         "max_step": max_step,
    #         "gamma": gamma
    #     },
    #     "codesign_params": {
    #         "batch_size_Phi": batch_size_Phi,
    #         "learning_rate_mu": learning_rate_mu,
    #         "learning_rate_sigma": learning_rate_sigma,
    #         "min_epi_codesign": min_epi_codesign
    #     },
    #     "learning_results": {
    #         "reward_list": reward_list,
    #         "q_value_list": q_value_list
    #     }
    # }

    # Save parameters and results
    # df_scaling = pd.DataFrame({'scaling': episode_scaling})
    # df_bidding = pd.DataFrame({'bidding': episode_bidding})
    # df_step_rewards = pd.DataFrame({'step_rewards': episode_step_rewards})
    # df_soc = pd.DataFrame({'soc': episode_soc_history})
    # df_penalty = pd.DataFrame({'penalty': episode_penalty_history})
    # df_Pc_t = pd.DataFrame({'Pc_t': episode_Pc_t_history})
    # df_Pd_t = pd.DataFrame({'Pd_t': episode_Pd_t_history})

    # df_scaling.to_csv(f'{log_dir}/scaling.csv', index=False)
    # df_bidding.to_csv(f'{log_dir}/bidding.csv', index=False)
    # df_step_rewards.to_csv(f'{log_dir}/step_rewards.csv', index=False)
    # df_soc.to_csv(f'{log_dir}/soc.csv', index=False)
    # df_penalty.to_csv(f'{log_dir}/penalty.csv', index=False)
    # df_Pc_t.to_csv(f'{log_dir}/Pc_t.csv', index=False)
    # df_Pd_t.to_csv(f'{log_dir}/Pd_t.csv', index=False)
    
if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]  
    processes = []
    battery_price = 5000
    battery_price_max = 6000
    
    while battery_price <= battery_price_max:
        processes = []
        for seed in seeds:
            p = Process(target=run_training, args=(seed,battery_price))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        battery_price += 1000