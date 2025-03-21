tf:  1.13.2
baselines: 0.1.6
python: 3.7.0
torch: 1.1.0
gfootball安装需要使用 
````
pip install -e . 
````
并在football/gfootball/scenarios下添加11_vs_11_005.py， 11_vs_11_035.py， 11_vs_11_065.py， 11_vs_11_095.py后使用（参考11_vs_11_easy_stochastic.py改难度即可）


Ours: 
````
python run_ppo2.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'impala_cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 10000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512
````

PPO+CR: 
````
python run_ppo2.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 10000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512
````

impala: 
````
python run_ppo_baseline.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'impala_cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 40000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512
````

PPO: 
````
python run_ppo_baseline.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 40000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512
````
