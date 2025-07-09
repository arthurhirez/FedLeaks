import os
import glob

from utils.conf import set_random_seed


# PROCESSES VARIABLES
run_simulation = False
federated_training = True
process_latents = True

generate_viz = False
save_extras = False


# HYPERPARAMETERS
comm_epoch = 5
local_epoch = 2
infoNCET = 0.015
LSTM_units = 20

interval = 86400
window = 84
step_size = 1

i = 0
set_random_seed(i*12)

file_paths = glob.glob(f'data_leaks/benchmark/energy/P*')

for window in [14, 7, 3]:
    for file in file_paths:
        exp_id = file[-14:]
        n_feats = int(exp_id[-6]) + 1
    
        exp_comments = f'BENCHMARK_ENERGY___{window}_{step_size}_{interval}___{i}'
        cmd = (
            f"python Run_Experiment_Bench.py "
            f"--experiment_id {exp_id} "
            f"--extra_coments {exp_comments} "
            f"--id_network Benchmark_energy "
            f"--dataset benchmark "
            f"--seed {i*12} "
            f"--communication_epoch {comm_epoch} "
            f"--local_epoch {local_epoch} "
            f"--interval_agg {interval} "
            f"--window_size {window} "
            f"--step_size {step_size} "
            f"--infoNCET {infoNCET} "
            f"--input_size {n_feats} "
            f"--output_size {n_feats} "
            f"--lstm_units {LSTM_units} "
            f"--run_simulation {run_simulation} "
            f"--federated_training {federated_training} "
            f"--process_latents {process_latents} "
            f"--generate_viz {generate_viz} "
            f"--save_extras {generate_viz} "
        )

        print(f"\nRunning command:\n{cmd}\n")
        os.system(cmd)



