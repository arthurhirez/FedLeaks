import pandas as pd
import os
import glob
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def get_similar_districts_from_path(path_string):
    try:
        # Split the experiment name
        parts = path_string.split('__')

        # Get target info
        header = parts[0]  # e.g., 'E_15_LL_LM'
        target_district = header.split('_')[0]  # 'E'
        target_init_state = header.split('_')[2]  # 'LL'
        target_final_state = header.split('_')[3]  # 'LM'

        # Get district state mapping
        state_mapping_str = parts[1]  # e.g., 'LM_LL_LM_LL_LL'
        district_states = state_mapping_str.split('_')

        # Districts in order
        district_names = ['A', 'B', 'C', 'D', 'E']

        if len(district_states) < 5:
            raise ValueError("Expected at least 5 states for districts A–E. Got: " + str(district_states))

        # Map district to state
        district_state_map = dict(zip(district_names, district_states[:5]))

        # Find similar districts (excluding the target itself)
        similar_initial = [d for d, s in district_state_map.items() if s == target_init_state and d != target_district]
        similar_final = [d for d, s in district_state_map.items() if s == target_final_state and d != target_district]

        return {
            "target_district": target_district,
            "target_init_state": target_init_state,
            "target_final_state": target_final_state,
            "initial_similar_districts": similar_initial,
            "final_similar_districts": similar_final
        }

    except Exception as e:
        print(f"Error parsing: {e}")
        return None


def analyze_experiment(results_df, results_id):
    try:
        info_state = get_similar_districts_from_path(results_id)
        tgt_client = results_id[0]

        top_ranked = results_df[results_df['rank'] == 1]

        summary = top_ranked.groupby(['epoch', 'label', 'client_2']).size().reset_index(name='count')
        summary['district_letter'] = summary['client_2'].str.extract(r'District_([A-Z])')

        summary['epoch'] = pd.to_numeric(summary['epoch'], errors='coerce')
        summary['label'] = pd.to_numeric(summary['label'], errors='coerce')
        summary = summary.dropna(subset=['district_letter', 'epoch', 'label'])

        # Final-only match
        summary['match_final_only'] = summary['district_letter'].isin(info_state['final_similar_districts'])

        # Warm-up rule
        def match_with_warmup(row):
            if row['label'] <= 3:
                return row['district_letter'] in info_state['initial_similar_districts']
            else:
                return row['district_letter'] in info_state['final_similar_districts']

        summary['match_warmup_rule'] = summary.apply(match_with_warmup, axis=1)

        # Summarize results
        total_final = summary['match_final_only'].sum()
        total_warmup = summary['match_warmup_rule'].sum()

        matches_per_epoch = summary.groupby('epoch')[['match_final_only', 'match_warmup_rule']].sum().reset_index()
        max_final = matches_per_epoch['match_final_only'].max()
        max_warmup = matches_per_epoch['match_warmup_rule'].max()

        matches_per_epoch['experiment'] = results_id
        matches_per_epoch['max_final_matches'] = max_final
        matches_per_epoch['max_warmup_matches'] = max_warmup

        return {
            'experiment': results_id,
            'total_final_matches': total_final,
            'total_warmup_matches': total_warmup,
            'max_final_matches': max_final,
            'max_warmup_matches': max_warmup
        }

    except Exception as e:
        return {'experiment': results_id, 'error': str(e)}


def extract_exp_info(exp_id):
    try:
        parts = exp_id.split('__')
        target_header = parts[0]                     # 'E_15_LL_LM'
        mapping = parts[1]                           # 'LM_LL_LM_LL_LL'

        target_district = target_header.split('_')[0]     # 'E'
        node_id = target_header.split('_')[1]             # '15'
        init_state = target_header.split('_')[2]          # 'LL'
        final_state = target_header.split('_')[3]         # 'LM'

        aux_exp1 = exp_id.split('__')[-1]
        aux_exp2 = exp_id.split('__')[-2]

        return pd.Series([
            f"{target_district}_{node_id}",         # 'E_15'
            f"{init_state}_{final_state}",          # 'LL_LM'
            mapping[:14],                                 # 'LM_LL_LM_LL_LL'
            f"{aux_exp2}_{aux_exp1}"
        ])
    except Exception as e:
        print(f"Failed parsing exp_id: {exp_id} — {e}")
        return pd.Series([None, None, None])


def find_similar(row):
    match_cols = [col for col in ['A', 'B', 'C', 'D', 'E'] if row[col] == row['target_final_state']]
    return '_'.join(match_cols)


def compute_best_cases(df_results):

    df_results[['target_district_node', 'target_init_final_state', 'initial_state_mapping', 'exp_id']] = (
        df_results['experiment'].apply(extract_exp_info)
    )

    # Get max warmup match for each unique config
    idx = df_results.groupby(
        ['target_district_node', 'target_init_final_state', 'initial_state_mapping']
    )['total_final_matches'].idxmax()

    df_best_per_config = df_results.loc[idx].reset_index(drop=True)

    df_best_per_config['correct_epochs'] = df_best_per_config['total_final_matches'] / 24

    df_best_per_config['target_district'] = df_best_per_config['target_district_node'].str.split('_').str[0]
    df_best_per_config['target_initial_state'] = df_best_per_config['target_init_final_state'].str.split('_').str[-2]
    df_best_per_config['target_final_state'] = df_best_per_config['target_init_final_state'].str.split('_').str[-1]

    df_best_per_config[['A', 'B', 'C', 'D', 'E']] = df_best_per_config['initial_state_mapping'].str.split('_', expand = True)

    df_best_per_config.drop(columns = ['target_district_node', 'initial_state_mapping', 'target_init_final_state'], inplace = True)

    df_best_per_config['similar_districts'] = df_best_per_config.apply(find_similar, axis=1)

    # Optional: sort by warmup performance
    df_best_per_config = df_best_per_config[df_best_per_config['similar_districts'] != '']
    df_best_per_config = df_best_per_config.sort_values(by=['total_final_matches', 'max_final_matches', 'similar_districts'], ascending=[False, False, True])

    return df_best_per_config



def compute_results(exp_folders = [], metrics = ['autocorr'], epochs_tgt = None):

    # Run analysis for each experiment folder
    epoch_results = []
    all_results = []

    for path in exp_folders:
        exp_name = os.path.basename(path)

        # Load data
        path_data = f"{path}/Baseline_proto.parquet"
        results_proto = pd.read_parquet(path_data)
        results_proto['epoch'] = results_proto['epoch'].astype(int)

        results_tgt = results_proto[results_proto['client_1'].str.contains(f'_{exp_name[0]}_')].drop(columns=['client_1'])

        # Compute rank based on the metrics
        results_tgt['test_metric'] = results_tgt[metrics[0]]

        if len(metrics) != 1:
            for metric in metrics[1:]:
                results_tgt['test_metric'] += results_tgt[metric]

        # Filter epochs
        if epochs_tgt is None: epochs_tgt = results_proto['epoch'].unique()
        results_tgt = results_tgt[results_tgt['epoch'].isin(epochs_tgt)]

        results_tgt['rank'] = results_tgt.groupby(['epoch', 'label'])['test_metric'].rank(method='dense', ascending=True)

        # Analyse the results over experiments
        exp_result = analyze_experiment(results_df = results_tgt, results_id = exp_name)
        all_results.append(exp_result.copy())

        # Analyse the results per epoch
        case_epochs = []

        for epoch in epochs_tgt:
            data_epoch = results_tgt[results_tgt['epoch'] == epoch]
            result = analyze_experiment(results_df = data_epoch, results_id = exp_name)
            result['epoch'] = epoch

            case_epochs.append(result)

            result_df = pd.DataFrame(result, index = [0])
            best_epochs = compute_best_cases(result_df)
            epoch_results.append(best_epochs.copy())

    exp_df = pd.DataFrame(all_results)
    best_cases = compute_best_cases(exp_df)


    return pd.concat(epoch_results, ignore_index = True), best_cases, exp_df, epoch_results