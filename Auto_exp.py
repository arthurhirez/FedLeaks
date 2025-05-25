import os

experiment_ids = ['Pipeline_Full_medium_E_loss']
infoNCET = [0.20]
LSTM_units = [20]

for info in infoNCET:
    for lstm in LSTM_units:
        exp_id = 'proto_month' + f'_{str(info)}' + f'_{str(lstm)}_history'
        cmd = (
            f"python Run_Experiment.py "
            f"--extra_coments {exp_id} "
            f"--infoNCET {info} "
            f"--lstm_units {lstm}"
        )
        print(f"Running: {cmd}")
        os.system(cmd)


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
