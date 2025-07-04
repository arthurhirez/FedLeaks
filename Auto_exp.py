import os

from utils.conf import set_random_seed

# parser.add_argument('--n_segments', type=int, default=24,
#                     help='Number of urban growth simulation segments.')
# parser.add_argument('--warm_up', type=int, default=4,
#                     help='Number of rounds without changes.')
# parser.add_argument('--epochs_lenght', type=int, default=6,
#                     help='Number of epochs per simulation run.')
# parser.add_argument('--days_lenght', type=int, default=5,
#                     help='Number of days in each epoch.')
# parser.add_argument('--n_intervals', type=int, default=24,
#                     help='Number of time intervals per day.')


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

#
# # DRIFT VARIABLES
# tgt_district = 'District_D'
# seed_node = '2'
# # income_density_mapping = 'ML_LM_LH_LL_LL'
# # drift_income = ['low', 'low', 'medium', 'high', 'medium']
# # drift_density = ['medium', 'high', 'low', 'low', 'high']
#
#
# income_density_mapping = ['ML_LM_LH_LL_LL', 'HL_LL_LL_LL_LH', 'LL_MH_LL_LL_LM', 'LM_LL_LH_LL_ML', 'ML_MM_HL_LL_LM'] #D
# income_density_mapping = ['LM_LH_LL_LL_LL', 'LH_LL_ML_HL_LL', 'HL_LH_LL_LM_LL', 'LL_MM_HL_LM_LL', 'HH_LH_ML_LM_LL']
# drift_income = ['low', 'low', 'medium', 'medium', 'high', 'high', 'high']
# drift_density = ['medium', 'high', 'low', 'high', 'low', 'medium', 'high']
#
#
#
# tgt_district = 'District_D'
# seed_node = '2'
# income_density_mapping = ['LM_LH_LL_LL_LL', 'LL_LL_LM_LH_LL', 'LM_LL_LM_LL_LL']
#
# income_density_mapping = ['HL_ML_LM_LH_LL', 'LH_HM_LM_MH_LL', 'MM_HL_LM_ML_LL', 'ML_HL_HM_LM_LL']
#
# income_density_mapping = ['HL_ML_LM_LH_LL', 'LH_HM_LM_MH_LL', 'MM_HL_LM_ML_LL', 'ML_HL_HM_LM_LL', 'LH_LL_LM_LL_LL', 'LM_LH_LL_LL_LL', 'LL_LM_LH_LL_LL',  'LL_LL_LM_LH_LL',]
#
#
#
#
# tgt_district = 'District_A'
# seed_node = '62'
#
# income_density_mapping = ['LL_LM_LH_LL_LL','LL_LH_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LL_LH_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL',
#                           'LL_HL_ML_LH_LL', 'LL_LH_HM_MH_ML', 'LL_MM_HL_ML_LM', 'LL_ML_HL_LM_HM']
#
#
# tgt_district = 'District_B'
# seed_node = '84'
#
# income_density_mapping = ['LM_LL_LH_LL_LL','LH_LL_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL',
#                           'HL_LL_ML_LH_LL', 'LH_LL_HM_MH_ML', 'MM_LL_HL_ML_LM', 'ML_LL_HL_LM_HM']
#
#
# tgt_district = 'District_C'
# seed_node = '65'
#
# income_density_mapping = ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LM_LL_LH_LL','LL_LH_LL_LM_LL',
#                           'HL_ML_LL_LH_LL', 'LH_HM_LL_MH_ML', 'MM_HL_LL_ML_LM', 'ML_HL_LL_LM_HM']
#
#
# tgt_district = 'District_D'
# seed_node = '2'
#
# income_density_mapping = ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LL_LH','LL_LL_LH_LL_LM','LH_LL_LL_LL_LM','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL',
#                           'HL_ML_LH_LL_LL', 'LH_HM_MH_LL_ML', 'MM_HL_ML_LL_LM', 'ML_HL_LM_LL_HM']
#
#
# tgt_district = 'District_E'
# seed_node = '15'
#
# income_density_mapping = ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL','LH_LL_LL_LM_LL','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL',
#                           'HL_ML_LH_LL_LL', 'LH_HM_MH_ML_LL', 'MM_HL_ML_LM_LL', 'ML_HL_LM_HM_LL']
#
# drift_income = ['low', 'low', 'medium']
# drift_density = ['medium', 'high', 'low']
#
#
#
#
#
# # NEW_TESTING
# tgt_district = 'District_A'
# seed_node = '62'
#
# income_density_mapping = [
#                             'MM_HL_LH_ML_LL',
#                             'HL_LH_LM_HH_HM',
#                             'MH_MM_LL_HL_HH',
#                             'LL_HM_LM_MM_LL',
#                             'HH_ML_LM_HL_LH',
#                             'ML_HH_LL_HM_MH',
#                          ]
#
#
# drift_income = ['high', 'high', 'high', 'medium', 'medium', 'medium', 'low']
# drift_density = ['low', 'medium', 'high', 'low', 'medium', 'high', 'low']
#
#
#
# # NEW_WINDOW / NEW_WINDOW_WEEKS
# tgt_district = 'District_A'
# seed_node = '62'
#
# income_density_mapping = ['LL_LM_LH_LL_LL','LL_LH_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LL_LH_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL',
#                           'LL_HL_ML_LH_LL', 'LL_LH_HM_MH_ML', 'LL_MM_HL_ML_LM', 'LL_ML_HL_LM_HM']
#
#
# drift_income = ['low', 'low', 'medium']
# drift_density = ['medium', 'high', 'low']
#
#
#
#
# interval_agg = 2 * 3600
# window_size = 6
# step_size = 28


# for i in range(1):
#     set_random_seed(i*12)
#     for interval in [2 * 3600, 6 * 3600, 12 * 3600]:
#         for window in [12, 48, 144]:
#             for step_size in [int(window / 4), int(window / 2), int(window * 0.75)]:

#
# for i in range(1):
#     set_random_seed(i*12)
#     for interval in [2 * 3600, 3 * 3600, 4 * 3600]:
#         for window in [84, 168, 360, 720, 1080]:
#             for step_size in [int(window / 4), int(window * 0.75)]:
#                 for DI, DD in zip(drift_income, drift_density):
#                     drift_ID = DI[0].capitalize() + DD[0].capitalize()
#                     for mapping in income_density_mapping:
#                         options = mapping.split('_')
#                         if drift_ID not in options

# HYPERPARAMETERS
comm_epoch = 10
local_epoch = 1
infoNCET = 0.015
LSTM_units = 20


# PROCESSES VARIABLES
run_simulation = False
federated_training = True
process_latents = True
generate_viz = True

federated_training = False

# drift_income = ['low', 'low']
# drift_density = ['medium', 'high']
#
#
# clients = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
# nodes = ['62', '84', '65', '2', '15']
# drifts = [['LL_LM_LH_LL_LL','LL_LH_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LL_LH_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL'],
# ['LM_LL_LH_LL_LL','LH_LL_LM_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL'],
# ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LL_LM_LH','LL_LL_LL_LH_LM','LH_LL_LL_LL_LM','LL_LM_LL_LH_LL','LL_LH_LL_LM_LL'],
# ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LL_LH','LL_LL_LH_LL_LM','LH_LL_LL_LL_LM','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL'],
# ['LM_LH_LL_LL_LL','LH_LM_LL_LL_LL','LL_LL_LM_LH_LL','LL_LL_LH_LM_LL','LH_LL_LL_LM_LL','LL_LM_LH_LL_LL','LL_LH_LM_LL_LL']]
#
# for i in range(1):
#     set_random_seed(i*12)
#     for interval in [2 * 3600, 8 * 3600, 12 * 3600]:
#         for window in [12, 24, 48]:
#             for step_size in [int(window / 2)]:



clients = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
nodes = ['62', '84', '65', '2', '15']
# drifts = [
#     ['LL_HL_ML_LH_LL', 'LL_LH_HM_MH_ML', 'LL_MM_HL_ML_LM', 'LL_ML_HL_LM_HM'],
#     ['HL_LL_ML_LH_LL', 'LH_LL_HM_MH_ML', 'MM_LL_HL_ML_LM', 'ML_LL_HL_LM_HM'],
#     ['HL_ML_LL_LH_LL', 'LH_HM_LL_MH_ML', 'MM_HL_LL_ML_LM', 'ML_HL_LL_LM_HM'],
#     ['HL_ML_LH_LL_LL', 'LH_HM_MH_LL_ML', 'MM_HL_ML_LL_LM', 'ML_HL_LM_LL_HM'],
#     ['HL_ML_LH_LL_LL', 'LH_HM_MH_ML_LL', 'MM_HL_ML_LM_LL', 'ML_HL_LM_HM_LL']
# ]


drifts = [
    ['LL_ML_HL_LL_LL','LL_HL_ML_LL_LL','LL_LL_LL_ML_HL','LL_LL_LL_HL_ML','LL_HL_LL_LL_ML','LL_LL_ML_HL_LL','LL_LL_HL_ML_LL'],
['ML_LL_HL_LL_LL','HL_LL_ML_LL_LL','LL_LL_LL_ML_HL','LL_LL_LL_HL_ML','HL_LL_LL_LL_ML','LL_LL_ML_HL_LL','LL_LL_HL_ML_LL'],
['ML_HL_LL_LL_LL','HL_ML_LL_LL_LL','LL_LL_LL_ML_HL','LL_LL_LL_HL_ML','HL_LL_LL_LL_ML','LL_ML_LL_HL_LL','LL_HL_LL_ML_LL'],
['ML_HL_LL_LL_LL','HL_ML_LL_LL_LL','LL_LL_ML_LL_HL','LL_LL_HL_LL_ML','HL_LL_LL_LL_ML','LL_ML_HL_LL_LL','LL_HL_ML_LL_LL'],
['ML_HL_LL_LL_LL','HL_ML_LL_LL_LL','LL_LL_ML_HL_LL','LL_LL_HL_ML_LL','HL_LL_LL_ML_LL','LL_ML_HL_LL_LL','LL_HL_ML_LL_LL']]


drift_income = ['medium', 'high']
drift_density = ['low', 'low']

drift_income = ['medium']
drift_density = ['low']

for i in range(1):
    set_random_seed(i*12)
    for interval in [2*3600]:
        for window in [12]:
            for step_size in [2]:
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
                            exp_comments = f'EXPERIMENT_INCOMESIMPLE_{mapping}__{window}_{step_size}_{interval}___{i}'

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
