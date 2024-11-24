# Optimization-RE-and-battery-capacity-using-DRQN

Python implementation of "Renewable Energy Bidding and Battery Size Co-optimization Using Reinforcement Learning". 

## Training of Standard DRQN agent

1. Training of a single DRQN agent for each BESS size setting:
   ```console
    ./main_drqn_multiple.py
   ```
1. Training of multiple DRQN agents in parallel:
   ```console
    ./main_drqn_multiple.py
   ```

<div align=left> 
<img src="./plot/bar_graph_average_random_update_year.png" width=300 alt="Total net revenue F for various BESS sizes"/>
<img src="./plot/box_graph_average_random_update_year_3.png" width=300 alt="Net profit G considering BESS installation costs"/>
</div>


## Co-design (Verification Using Short-term Data)

1. Training of a single co-design agent:
   ```console
    ./main_codesign_drqn.py
   ```
1. Training of multiple co-design agents in parallel:
   ```console
    ./main_codesign_drqn_multiple.py
   ```

<div align=left> 
<img src="./plot/update_mu.png" width=300 alt="Change in the value $\mu$ during learning process"/>
</div>

## Co-design (Applicability to Long-Term Data)

1. Training of co-desing agent with parallel workers:
   ```console
    ./main_codesign_drqn_apex.py
   ```

<div align=left> 
<img src="./plot/plot_bar_update_mu.png" width=300 alt="Total revenues with fixed and optimized BESS sizes"/>
</div>

