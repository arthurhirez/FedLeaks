import os

from utils.conf import set_random_seed



def drift_id(tgt_district, seed_node, income_density_mapping, drift_income, drift_density):
    # 1. First part: first letter of tgt_district (e.g., "District_D" → "D")
    district_code = tgt_district.split('_')[-1]

    # 2. Second part: Get the N-th value of income_density_mapping, where N is the index of district_code in alphabet
    mapping_values = income_density_mapping.split('_')
    idx = ord(district_code.upper()) - ord('A')  # A=0, B=1, ..., D=3
    if idx < len(mapping_values):
        density_code = mapping_values[idx]
    else:
        density_code = "??"

    # 3. Third part: zip drift pairs and generate codes like "LM", "LH", etc.
    drift_code = drift_income[0].upper() + drift_density[0].upper()

    # Combine parts into final ID string
    exp_id = f"{district_code}_{seed_node}_{density_code}_{drift_code}"
    return exp_id



# HYPERPARAMETERS
comm_epoch = 10
local_epoch = 1
infoNCET = 0.20
LSTM_units = 30

drift_income = ['low']
drift_density = ['medium']


clients = ['District_D']
nodes = ['2']
drifts = [['LM_LL_LH_LL_ML']]


interval = 2*3600
step_size = 1

# HYPERPARAMETERS
communication = [20]
local = [1]


# PROCESSES VARIABLES
run_simulation = False
federated_training = True
process_latents = True
generate_viz = True


for i in range(1):
    set_random_seed(i*12)
    for window in [252]:
        for comm_epoch, local_epoch in zip(communication, local):
            for LSTM_units in [35]: #[30]:
                for infoNCET in [0.02, 0.1]: #[0.4]:
                    for DI, DD in zip(drift_income, drift_density):
                        drift_ID = DI[0].capitalize() + DD[0].capitalize()
                        for tgt_district, seed_node, income_density_mapping in zip(clients, nodes, drifts):
                            for mapping in income_density_mapping:
                                options = mapping.split('_')
                                if drift_ID not in options:
                                    continue

                                exp_id = drift_id(tgt_district, seed_node, mapping, DI, DD)
                                exp_id += f'__{mapping}'
                                # exp_comments = f'proto_NCET{str(infoNCET).replace(".", "")}_LSTM{LSTM}'
                                exp_comments = f'FIGURES3D_{mapping}__{window}_{step_size}_{interval}_{LSTM_units}_{str(infoNCET).replace(".", "")}___{i}'

                                cmd = (
                                    f"python Run_Experiment.py "
                                    f"--experiment_id {exp_id} "
                                    f"--extra_coments {exp_comments} "
                                    f"--seed {i*12} "
                                    f"--communication_epoch {comm_epoch} "
                                    f"--local_epoch {local_epoch} "
                                    f"--interval_agg {interval} "
                                    f"--window_size {window} "
                                    f"--step_size {step_size} "
                                    f"--infoNCET {infoNCET} "
                                    f"--lstm_units {LSTM_units} "
                                    f"--run_simulation {run_simulation} "
                                    f"--federated_training {federated_training} "
                                    f"--process_latents {process_latents} "
                                    f"--generate_viz {generate_viz} "
                                    f"--tgt_district {tgt_district} "
                                    f"--seed_node {seed_node} "
                                    f"--drift_income {DI} "
                                    f"--drift_density {DD}"
                                )

                                print(f"\nRunning command:\n{cmd}\n")
                                os.system(cmd)

                        run_simulation = False



