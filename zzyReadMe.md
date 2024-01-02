## race
统一budget 0.05
`python main.py --step_size 0.03 --eps_g 0.05 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-05 --uniform_eps`
`python main.py --step_size 0.03 --eps_g 0.05 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-05 --uniform_eps --bootstrap 100 --valid True --load_epoch 2299`
统一budget 0.01
`python main.py --step_size 0.03 --eps_g 0.01 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-01 --uniform_eps`
`python main.py --step_size 0.03 --eps_g 0.01 --sensitive_attr race --max_epoch_stage1 800 --max_epoch_stage2 1500 --seed 1 --target_dir_name race_DP_0-01 --uniform_eps --bootstrap 100 --valid True --load_epoch 2299`
比例budget
先算原始dis
`python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --sensitive_attr race --seed 1 --target_dir_name race_specific_1-0 --uniform --uniform_eps`
`python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --sensitive_attr race --seed 1 --target_dir_name race_specific_1-0 --uniform --uniform_eps --valid True --bootstrap 100 --load_epoch 1799`
再加个权重
python main.py --weight_eps 0.9 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.10 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-9
python main.py --weight_eps 0.7 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.06 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-7
python main.py --weight_eps 0.6 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.06 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-6
python main.py --weight_eps 0.5 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.05 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-5
python main.py --weight_eps 0.4 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.04 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-4
python main.py --weight_eps 0.3 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.04 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-3
python main.py --weight_eps 0.2 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.03 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-2
python main.py --weight_eps 0.1 --max_epoch_stage1 800 --max_epoch_stage2 1000 --step_size 0.02 --sensitive_attr race --seed 1 --target_dir_name race_specific_0-1

## bank
python main.py --step_size 1.0 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset bank --sensitive_attr noloan --seed 1 --target_dir_name bank_noloan_specific_1-0_1-0 --uniform --uniform_eps
python main.py --step_size 1.0 --max_epoch_stage1 800 --max_epoch_stage2 1000 --dataset bank --sensitive_attr noloan --seed 1 --target_dir_name bank_noloan_specific_0-1_1-0 --weight_eps 0.1
## compas
python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset compas --sensitive_attr race --seed 1 --target_dir_name compas_African_specific_1-0 --uniform --uniform_eps
python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1000  --dataset compas --sensitive_attr race --seed 1 --target_dir_name compas_African_specific_0-1 --weight_eps 0.1
一直到0.9

要想做bootstrap，需要添加：--valid True --bootstrap 100 --load_epoch 1799

## eicu
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0_0-5 --uniform --uniform_eps
python main.py --step_size 1.0 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0_1-0 --uniform --uniform_eps
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0_0-5_half --uniform --uniform_eps
python main.py --step_size 1.0 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0_1-0_half --uniform --uniform_eps

python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1000 --eps_g 1.0 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_1-0_0-5_bigdisparity --uniform --uniform_eps
python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1000 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name eicu_race_specific_0-1_0-5 --weight_eps 0.1


--valid True --bootstrap 100 --load_epoch 2299

python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-9_0-50_single_weekplus_newsort --weight_eps 0.9

python main.py --step_size 0.30 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-7_0-30_single_weekplus_newsort --weight_eps 0.7

python main.py --step_size 0.23 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-6_0-23_single_weekplus_newsort --weight_eps 0.6

python main.py --step_size 0.15 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-5_0-15_single_weekplus_newsort --weight_eps 0.5

python main.py --step_size 0.12 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-4_0-12_single_weekplus_newsort --weight_eps 0.4

python main.py --step_size 0.09 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-3_0-09_single_weekplus_newsort --weight_eps 0.3

python main.py --step_size 0.07 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-2_0-07_single_weekplus_newsort --weight_eps 0.2

python main.py --step_size 0.17 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-4_0-17_single_weekplus_newsort --weight_eps 0.4

python main.py --step_size 0.13 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-3_0-13_single_weekplus_newsort --weight_eps 0.3

python main.py --step_size 0.05 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name weekplus_newsort/eicu_race_specific_0-2_0-05_single_weekplus_newsort --weight_eps 0.2

python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --seed 1 --target_dir_name new_trial/eicu_DP_0-01_0-50_multi --eps_g 0.01 --uniform_eps --bootstrap 100 --load_epoch 2299 --new_trial 50

python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --target_dir_name new_trial/eicu_DP_0-01_0-50_multi_another_seed_5 --eps_g 0.01 --uniform_eps --bootstrap 100 --load_epoch 2299 --new_trial 50 --seed 5


python main.py --step_size 0.50 --max_epoch_stage1 800 --max_epoch_stage2 1500 --dataset eicu --sensitive_attr race --target_dir_name new_trial_train_rate/eicu_DP_0-01_0-50_multi_0-7_1 --eps_g 0.01 --uniform_eps --bootstrap 100 --load_epoch 2299 --new_trial 200 --seed 1 --new_trial_train_rate 0.7
<!-- 2024.1.2 -->
python main.py --step_size 0.50 --max_epoch_stage1 100 --max_epoch_stage2 100 --dataset eicu --sensitive_attr race --target_dir_name new_trial/eicu_DP_0-01_0-50_seed_1 --eps_g 0.01 --uniform_eps --new_trial 10 --bootstrap 100 --seed 1
python main.py --step_size 0.50 --max_epoch_stage1 100 --max_epoch_stage2 100 --dataset eicu --sensitive_attr race --target_dir_name new_trial/eicu_DP_0-01_0-50_seed_1 --eps_g 0.01 --uniform_eps --new_trial 10 --new_trial_test_rate 0.135 --valid --seed 1
...
python main.py --step_size 0.50 --max_epoch_stage1 100 --max_epoch_stage2 100 --dataset eicu --sensitive_attr race --target_dir_name new_trial/eicu_DP_0-01_0-50_seed_1 --eps_g 0.01 --uniform_eps --new_trial 10 --new_trial_test_rate 0.015 --valid --seed 1