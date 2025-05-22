import os

experiment_ids = ['Pipeline_Full']
communication_epochs = [5, 7, 10, 15, 20, 30, 50]
local_epochs = [1, 2, 3, 4, 5]


for comm_epoch in communication_epochs:
    for local_epoch in local_epochs:
        cmd = (
            f"python Run_Experiment.py "
            f"--experiment_id {experiment_ids[0]} "
            f"--communication_epoch {comm_epoch} "
            f"--local_epoch {local_epoch}"
        )
        print(f"Running: {cmd}")
        os.system(cmd)
