import pandas as pd

from utils.Toolbox_detection import combined_value_counts
from utils.Toolbox_results import summary_experiment, experiment_detection

exp_id='HYPER _REFACTOR'
dir_id='densit_base'
hyper_id='2_4_12__10_2_40'

# hyper_id='2_4_12__5_2_30' TA FALTANDO INFO 0.4
# hyper_id='2_4_12__10_1_30' TA FALTANDO INFO 0.4 -> density base


df = summary_experiment(exp_id=exp_id, dir_id=dir_id, hyper_id=hyper_id)

print('Experiments summarized - check df_summary on the experiment directory.')

folders_experiment = df['path'].tolist()
total_epochs = df['com_epoch'].unique().tolist()
if len(total_epochs) != 1:
    raise ValueError('The number of unique epochs should be 1.')

dict_params = {
    'periods_params': {
        'periods': [{
            'first_4_months': list(range(0, 5)),
            'first_year': list(range(0, 13)),
            'second_year': list(range(11, 24)),
            'all_periods': list(range(0, 24)),
        }, {
            'first_4_months': list(range(0, 5)),
            'all_periods': list(range(0, 24)),
        }],
        'periods_tags': ['Overlapped', 'Targeted']
    },
    'skip_epochs': [True, False],
    'epochs_params': {
        'epochs': [[0, 1, 2], [0, 1, 2, 3, 4], None],
        'epochs_tags': ['init', 'selected', 'all'],
        'total_epochs' : total_epochs[0],
    },
    'metrics_similarity' : ['cosine', 'wavelet', 'dft', 'autocorr']
}

if exp_id == 'HYPER':
    if hyper_id is None:
        raise NameError('definir hyperparameter')
    path = f'results/{dir_id}/traceback/{exp_id}/{hyper_id}'
else:
    path = f'results/{dir_id}/traceback/{exp_id}'


experiment_detection(folders_experiment=folders_experiment, params_dict = dict_params,
                     exp_id=exp_id, dir_id=dir_id, hyper_id=hyper_id)