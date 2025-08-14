import pandas as pd
import numpy as np
import glob
import os
import pickle
from tqdm import tqdm

from utils.Toolbox_detection import analyze_experiments_by_epoch, evaluate_extraction_methods
from utils.Toolbox_detection import extract_best_clients_by_max_cosine_distance, \
    extract_best_clients_by_ranked_summary_initial, extract_best_clients_by_summary_metrics
from utils.Toolbox_similarity import compute_results


def summary_experiment(exp_id = 'INCOMESIMPLE', dir_id = 'income_base', hyper_id = None):
    folders_experiment = [f for f in glob.glob(os.path.join(f'results/{dir_id}', f'*{exp_id}*')) if os.path.isdir(f)]
    # folders_experiment = [f.replace('___0', '').replace('results/density_base\\', '') for f in folders_experiment]

    ids_exp = folders_experiment[0].replace(f'results/{dir_id}\\', '').split('__')
    exp_path = 'datasets/leaks/Graeme/' + ids_exp[0] + '__' + ids_exp[1][:14] + '/'

    rows = []

    for filename in folders_experiment:
        parts = filename.replace('___0', '').replace(f'results/{dir_id}\\', '').split('__')

        if len(parts) == 3:
            prefix, window_part, suffix = parts
            before_window, after_window = window_part.split(f'_EXPERIMENT_{exp_id}_')

            row = {
                'prefix': prefix,
                'before_window': before_window[15:],
                'after_window': after_window,
                'suffix': suffix,
                'path' : filename
            }
            rows.append(row)
        else:
            print(f"Skipping invalid format: {filename}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    cols_exp = ['window', 'step_size', 'interval', 'lstm', 'infonet'] if 'HYPER' in exp_id else ['window', 'step_size', 'interval']
    df[cols_exp] = df['suffix'].str.split('_', expand = True).astype(float)
    df[['com_epoch', 'loc_epoch', 'agg', 'window_check']] = df['before_window'].str.split('_', expand = True).astype(int)

    df['step_ratio'] = round(df['step_size']  / df['window'], 3)
    df['init_state'] = df['prefix'].str[5:7]
    df['final_state'] = df['prefix'].str[-2:]

    df['exp_id'] = df['init_state']  + '_' + df['final_state']  + '__' + df['after_window']

    df[['A', 'B', 'C', 'D', 'E']] = df['after_window'].str.split('_', expand = True)

    if 'HYPER' in exp_id:
        df = df[['init_state', 'final_state', 'A', 'B', 'C', 'D', 'E', 'agg', 'window',
           'step_size', 'step_ratio', 'lstm', 'infonet','com_epoch', 'loc_epoch', 'path']]
    else:
        df = df[['init_state', 'final_state', 'A', 'B', 'C', 'D', 'E', 'agg', 'window',
           'step_size', 'step_ratio', 'path']]

    df['exp_id'] = df['path'].str.split('__').str[0].str[-5:] + '__' + df['path'].str.split('__').str[1].str[:14]
    df['ground_truth'] = df['path'].str.replace(f'results/{dir_id}\\', '').str.split('__').str[0].str[0]


    target_cols = ['A', 'B', 'C', 'D', 'E']
    df['match_init_state'] = df[target_cols].eq(df['init_state'], axis=0).sum(axis=1)
    df['match_final_state'] = df[target_cols].eq(df['final_state'], axis=0).sum(axis=1)

    def find_similar(row):
        match_cols = [col for col in ['A', 'B', 'C', 'D', 'E'] if row[col] == row['final_state']]
        return '_'.join(match_cols)

    df['similar_districts'] = df.apply(find_similar, axis=1)

    # Get last character of final_state
    df['final_income'] = df['final_state'].str[0]
    df['final_density'] = df['final_state'].str[-1]

    # Get last character of each target column and compare
    df['match_final_income'] = (
        df[target_cols].apply(lambda x: x.str[0], axis=1)
        .eq(df['final_income'], axis=0)
        .sum(axis=1)
    )

    df['match_final_density'] = (
        df[target_cols].apply(lambda x: x.str[-1], axis=1)
        .eq(df['final_density'], axis=0)
        .sum(axis=1)
    )

    df = df[df['match_final_state'] == 1]

    # df_density = df[df['final_state'] != 'ML']
    if 'HYPER' in exp_id:
        if hyper_id is None:
            raise NameError('definir hyperparameter')
        path = f'results/{dir_id}/traceback/{exp_id}/{hyper_id}'
    else:
        path = f'results/{dir_id}/traceback/{exp_id}'

    os.makedirs(path, exist_ok=True)
    df.to_csv(f'{path}/df_summary.csv', index = False)
    return df




def experiment_detection(folders_experiment, params_dict, compile_results = True,
                         exp_id='HYPER', dir_id='density_base', hyper_id = None,
                         ):

    periods = params_dict['periods_params']['periods']
    periods_tags = params_dict['periods_params']['periods_tags']

    epochs = params_dict['epochs_params']['epochs']
    epochs_tag = params_dict['epochs_params']['epochs_tags']

    if 'HYPER' in exp_id:
        if hyper_id is None:
            raise NameError('definir hyperparameter')
        path = f'results/{dir_id}/traceback/{exp_id}/{hyper_id}'
    else:
        path = f'results/{dir_id}/traceback/{exp_id}'

    print('\nDetection experiments:')

    for skip_tgt in tqdm(params_dict['skip_epochs'], desc="Skip Epochs"):
        for epochs_tgt, ep_tag in tqdm(zip(epochs, epochs_tag), desc="Epochs"):
            for period_tgt, pr_tag in tqdm(zip(periods, periods_tags), desc="Periods"):

                results = analyze_experiments_by_epoch(data = folders_experiment, epochs = epochs_tgt,
                                                       periods = period_tgt, skip_middle = skip_tgt)

                possible_exps = set([exp.split('__')[-2] for exp in list(results.keys())])

                setup_dict = {}
                for setup in possible_exps:
                    exps_setup = [exp for exp in list(results.keys()) if setup in exp]
                    setup_dict[setup] = exps_setup

                extraction_methods = [
                    ("max cosine", extract_best_clients_by_max_cosine_distance),
                    ("rank cosine_mean", extract_best_clients_by_ranked_summary_initial),
                    ("rank cosine_sum", extract_best_clients_by_ranked_summary_initial),
                    ("summary", extract_best_clients_by_summary_metrics),
                ]

                results_evaluation, summary_df = evaluate_extraction_methods(setup_dict = setup_dict,
                                                                             results = results,
                                                                             extractors = extraction_methods,
                                                                             periods_list = list(period_tgt.keys()),
                                                                             exp_directory = f'results/{dir_id}\\')

                save_results = {
                    'data_analysis' : results,
                    'evaluation' : results_evaluation,
                    'data_evaluation' : summary_df
                }

                with open(f'{path}/detect__{str(skip_tgt).lower()}_{pr_tag}_{ep_tag}.pkl', 'wb') as f:
                    pickle.dump(save_results, f)


    if compile_results:
        print('\nDone - compiling results.')
        data = []
        for skip_tgt in params_dict['skip_epochs']:
            for ep_tag in epochs_tag:
                for pr_tag in periods_tags:
                    with open(f'{path}/detect__{str(skip_tgt).lower()}_{pr_tag}_{ep_tag}.pkl', 'rb') as f:
                        save_results = pickle.load(f)

                    aux = save_results['evaluation'].rename(columns={'exp_id': 'state_map'})
                    aux['skip_tgt'] = skip_tgt
                    aux['epochs_tgt'] = ep_tag
                    aux['period_tgt'] = pr_tag
                    aux['detection'] = np.where(aux['correct_%'] >= 0.75, 1, 0)

                    aux['exp_id'] = exp_id
                    data.append(aux)
        compiled = pd.concat(data)
        compiled.to_csv(f'{path}/compiled_detection.csv', index=False)

    print('\nSimilarity experiments:')

    for metric in tqdm(params_dict['metrics_similarity'], desc="Similarity Metrics"):
        for epochs_tgt, ep_tag in tqdm(zip(epochs, epochs_tag), desc="Epochs", leave=False):
            similar = compute_results(exp_folders = folders_experiment, metrics = [metric], epochs_tgt = epochs_tgt)

            with open(f'{path}/similar__{metric}_{ep_tag}.pkl', 'wb') as f:
                    pickle.dump(similar, f)

    if compile_results:
        print('\nDone - compiling results.')
        data = []
        for metric in params_dict['metrics_similarity']:
            for epochs_tgt, ep_tag in zip(epochs, epochs_tag):
                with open(f'{path}/similar__{metric}_{ep_tag}.pkl', 'rb') as f:
                    similar = pickle.load(f)

                best_results = similar[2]
                valid_epochs = params_dict['epochs_params']['total_epochs'] if epochs_tgt is None else len(epochs_tgt)
                best_results['correct_epochs'] = best_results['total_final_matches'] / 24
                best_results['correct_percent'] = best_results['correct_epochs'] / valid_epochs

                cols_exp = ['window', 'step_size', 'agg', 'lstm', 'infonet'] if 'HYPER' in exp_id else ['window',
                                                                                                        'step_size',
                                                                                                        'agg']

                best_results[cols_exp] = best_results['exp_id'].str.replace('__0', '').str.split('_',
                                                                                                 expand=True).astype(
                    float)

                best_results['agg'] = (best_results['agg'] / 3600).astype(int)
                best_results['step_ratio'] = round(best_results['step_size'] / best_results['window'], 3)

                best_results['metric'] = metric
                best_results['ep_tag'] = ep_tag
                best_results['detection'] = np.where(best_results['correct_percent'] >= 0.75, 1, 0)

                data.append(best_results)

        compiled = pd.concat(data)
        compiled.to_csv(f'{path}/compiled_similarity.csv', index=False)

    print('FINISHED!!!')


def compile_results_detection(exp_id='NEW_WINDOW', exp_dir='density_base', hyper_id=None):
    if 'HYPER' in exp_id:
        if hyper_id is None:
            raise NameError('definir hyperparameter')
        path = f'results/{exp_dir}/traceback/{exp_id}/{hyper_id}'
    else:
        path = f'results/{exp_dir}/traceback/{exp_id}'

    data = []
    for skip_tgt in [True, False]:
        for ep_tag in ['init', 'all']:
            for period_tag in ['Overlapped', 'Targeted']:
                with open(f'{path}/detect__{str(skip_tgt).lower()}_{period_tag}_{ep_tag}.pkl', 'rb') as f:
                    save_results = pickle.load(f)

                aux = save_results['evaluation'].rename(columns={'exp_id': 'state_map'})
                aux['skip_tgt'] = skip_tgt
                aux['epochs_tgt'] = ep_tag
                aux['period_tgt'] = period_tag
                aux['detection'] = np.where(aux['correct_%'] >= 0.75, 1, 0)

                aux['exp_id'] = exp_id
                data.append(aux)
    compiled = pd.concat(data)
    compiled.to_csv(f'{path}/compiled_detection.csv', index=False)

    return compiled


def compile_results_similarity(exp_id='NEW_WINDOW', exp_dir='density_base', hyper_id=None, total_epochs=10):
    if 'HYPER' in exp_id:
        if hyper_id is None:
            raise NameError('definir hyperparameter')
        path = f'results/{exp_dir}/traceback/{exp_id}/{hyper_id}'
    else:
        path = f'results/{exp_dir}/traceback/{exp_id}'

    data = []
    for metric in ['cosine', 'wavelet', 'dft', 'autocorr']:
        for epochs_tgt, ep_tag in zip([[0, 1, 2], None], ['init', 'all']):
            with open(f'{path}/similar__{metric}_{ep_tag}.pkl', 'rb') as f:
                similar = pickle.load(f)

            best_results = similar[2]
            valid_epochs = total_epochs if epochs_tgt is None else len(epochs_tgt)
            best_results['correct_epochs'] = best_results['total_final_matches'] / 24
            best_results['correct_percent'] = best_results['correct_epochs'] / valid_epochs

            cols_exp = ['window', 'step_size', 'agg', 'lstm', 'infonet'] if 'HYPER' in exp_id else ['window',
                                                                                                    'step_size', 'agg']

            best_results[cols_exp] = best_results['exp_id'].str.replace('__0', '').str.split('_', expand=True).astype(
                float)

            best_results['agg'] = (best_results['agg'] / 3600).astype(int)
            best_results['step_ratio'] = round(best_results['step_size'] / best_results['window'], 3)

            best_results['metric'] = metric
            best_results['ep_tag'] = ep_tag
            best_results['detection'] = np.where(best_results['correct_percent'] >= 0.75, 1, 0)

            data.append(best_results)

    compiled = pd.concat(data)
    compiled.to_csv(f'{path}/compiled_similarity.csv', index=False)

    return compiled