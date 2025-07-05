import os

from utils.conf import set_random_seed



# PROCESSES VARIABLES
run_simulation = False
federated_training = False
process_latents = True

generate_viz = True
save_extras = True


# HYPERPARAMETERS
comm_epoch = 10
local_epoch = 2
infoNCET = 0.015
LSTM_units = 20

interval = 2*3600
window = 84
step_size = 1

i = 0
set_random_seed(i*12)


for local_epoch in [2, 5]:
    for window in [6, 12, 24]:
        for interval in [1800, 3600]:
            exp_comments = f'BENCHMARK_INITIAL___{window}_{step_size}_{interval}___{i}'

            cmd = (
                f"python Run_Experiment.py "
                f"--experiment_id Benchmark "
                f"--extra_coments {exp_comments} "
                f"--seed {i*12} "
                f"--communication_epoch {comm_epoch} "
                f"--local_epoch {local_epoch} "
                f"--interval_agg {interval} "
                f"--window_size {window} "
                f"--step_size {step_size} "
                f"--infoNCET {infoNCET} "
                f"--input_size 2 "
                f"--output_size 2 "
                f"--lstm_units {LSTM_units} "
                f"--run_simulation {run_simulation} "
                f"--federated_training {federated_training} "
                f"--process_latents {process_latents} "
                f"--generate_viz {generate_viz} "
                f"--save_extras {generate_viz} "
            )

            print(f"\nRunning command:\n{cmd}\n")
            os.system(cmd)

    run_simulation = False




# experiment_ids = ['Pipeline_Full_medium_E']
# communication_epochs = [15, 20, 30]
# local_epochs = [1, 2, 3]
#
#
# for comm_epoch in communication_epochs:
#     for local_epoch in local_epochs:
#         cmd = (
#             f"python Run_Experiment.py "
#             f"--experiment_id {experiment_ids[0]} "
#             f"--communication_epoch {comm_epoch} "
#             f"--local_epoch {local_epoch}"
#         )
#         print(f"Running: {cmd}")
#         os.system(cmd)
