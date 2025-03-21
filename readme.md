
Ours: python run_ppo2.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'impala_cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 10000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512

PPO+CR: python run_ppo2.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 10000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512

impala: python run_ppo_baseline.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'impala_cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 40000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512


PPO: python run_ppo_baseline.py   --level '11_vs_11_easy_stochastic'   --reward_experiment 'scoring,checkpoints'   --policy 'cnn'   --cliprange 0.08   --gamma 0.993   --ent_coef 0.003   --num_timesteps 40000000   --save_interval 100   --max_grad_norm 0.64   --lr 0.000343   --num_envs 16   --noptepochs 2   --nminibatches 8   --nsteps 512
