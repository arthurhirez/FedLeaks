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


# DRIFT VARIABLES
tgt_district = 'District_D'
seed_node = '2'
# income_density_mapping = 'ML_LM_LH_LL_LL'
# drift_income = ['low', 'low', 'medium', 'high', 'medium']
# drift_density = ['medium', 'high', 'low', 'low', 'high']


income_density_mapping = ['ML_LM_LH_LL_LL', 'HL_LL_LL_LL_LH', 'LL_MH_LL_LL_LM', 'LM_LL_LH_LL_ML', 'ML_MM_HL_LL_LM'] #D

income_density_mapping = ['LM_LH_LL_LL_LL', 'LH_LL_ML_HL_LL', 'HL_LH_LL_LM_LL', 'LL_MM_HL_LM_LL', 'HH_LH_ML_LM_LL']

drift_income = ['low', 'low', 'medium', 'medium', 'high', 'high', 'high']
drift_density = ['medium', 'high', 'low', 'high', 'low', 'medium', 'high']


income_density_mapping = ['LL_MH_LL_LL_LM']
drift_income = ['medium']
drift_density = ['high']


# PROCESSES VARIABLES
run_simulation = False
federated_training = True
process_latents = True
generate_viz = False

# HYPERPARAMETERS
comm_epoch = 20
local_epoch = 2
infoNCET = 0.20
LSTM_units = 20

for i in range(10):
    set_random_seed(i)
    for DI, DD in zip(drift_income, drift_density):
        for mapping in income_density_mapping:
            exp_id = drift_id(tgt_district, seed_node, mapping, DI, DD)
            exp_id += f'__{mapping}'
            # exp_comments = f'proto_NCET{str(infoNCET).replace(".", "")}_LSTM{LSTM}'
            exp_comments = f'proto_REP_{mapping}__{i}'
            
            cmd = (
                f"python Run_Experiment.py "
                f"--experiment_id {exp_id} "
                f"--extra_coments {exp_comments} "
                f"--seed {i*7} "
                f"--communication_epoch {comm_epoch} "
                f"--local_epoch {local_epoch} "
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
            
    # run_simulation = False




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
