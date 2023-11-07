python train.py  --env_name miner  --exp_name ppo  --use_which_gae fixed  --flag gamma0.999 &
python train.py  --env_name miner  --exp_name ppo  --use_which_gae average   &
python train.py  --env_name miner  --exp_name ppo  --use_which_gae normal --gamma_type increase --start_gamma 0.95 --end_gamma 0.99 --flag inc9599 &
python train.py  --env_name miner  --exp_name ppo  --use_which_gae normal --gamma_type decrease --start_gamma 0.99 --end_gamma 0.95 --flag dec9599 &
python train.py  --env_name miner  --exp_name ppo  --use_which_gae normal --gamma_type random   --start_gamma 0.95 --end_gamma 0.99 --flag ran9599 &

