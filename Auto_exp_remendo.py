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
infoNCET = 0.015
LSTM_units = 20



drift_income = ['low', 'low']
drift_density = ['medium', 'high']


clients = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
nodes = ['62', '84', '65', '2', '15']
drifts = [
    ['LL_LM_LH_LL_LL','LL_LH_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LL_LH_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL'],
    ['LM_LL_LH_LL_LL','LH_LL_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL'],
    ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LM_LL_LH_LL','LL_LH_LL_LM_LL'],
    ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LL_LH','LL_LL_LH_LL_LM','LH_LL_LL_LL_LM','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL'],
    ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL','LH_LL_LL_LM_LL','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL']]


interval = 4*3600
step_size = 2

# HYPERPARAMETERS
communication = [10]
local = [1]


# PROCESSES VARIABLES
run_simulation = False
federated_training = True
process_latents = True
generate_viz = False


for i in range(1):
    set_random_seed(i*12)
    for window in [12]:
        for comm_epoch, local_epoch in zip(communication, local):
            for LSTM_units in [40, 60]: #[30]:
                for infoNCET in [0.05, 0.1, 0.2, 0.4]: #[0.4]:
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
                                exp_comments = f'EXPERIMENT_HYPER_{mapping}__{window}_{step_size}_{interval}_{LSTM_units}_{infoNCET}___{i}'

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



