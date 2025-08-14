import matplotlib
import matplotlib.colors as mcolors

import plotly.graph_objects as go


def combined_value_counts(df, normalize=False, sort_index=True):
    """
    Compute value counts (and optionally normalized percentages) of combinations
    of multiple columns in a DataFrame.

    Parameters:
    - df: DataFrame with the columns to group by.
    - normalize: If True, return relative frequencies.
    - sort_index: If True, sort the result by index.

    Returns:
    - A DataFrame with a MultiIndex (column combinations) and columns ['count', 'percent'].
    """
    # Raw counts
    counts = df.value_counts(normalize=False)
    percents = df.value_counts(normalize=True)

    # Combine into one DataFrame
    result = pd.DataFrame({
        'count': counts,
        'percent': percents
    })

    if normalize:
        result.drop(columns='count', inplace=True)

    if sort_index:
        result = result.sort_index()

    return result


import os
from collections import defaultdict

import numpy as np

from sklearn.metrics.pairwise import cosine_distances


def combined_value_counts(df, normalize=False, sort_index=True, figsize=None):
    """
    Compute value counts (and optionally normalized percentages) of combinations
    of multiple columns in a DataFrame.

    Parameters:
    - df: DataFrame with the columns to group by.
    - normalize: If True, return relative frequencies.
    - sort_index: If True, sort the result by index.
    - figsize: Tuple (width, height) for plotting. If None, no plot is generated.

    Returns:
    - A DataFrame with a MultiIndex (column combinations) and columns ['count', 'percent'].
    """
    # Raw counts
    counts = df.value_counts(normalize=False)
    percents = df.value_counts(normalize=True)

    # Combine into one DataFrame
    result = pd.DataFrame({
        'count': counts,
        'percent': percents
    })

    if normalize:
        result.drop(columns='count', inplace=True)

    if sort_index:
        result = result.sort_index()

    # Optional plot
    if figsize is not None:
        plot_col = 'percent' if normalize else 'count'
        result_to_plot = result[plot_col]
        result_to_plot.plot(kind='bar', figsize=figsize, title='Combined Value Counts')
        plt.ylabel(plot_col.capitalize())
        plt.tight_layout()
        plt.show()

    return result


def client_self_cosine_drift(data_latent, skip_middle = False):
    features = [col for col in data_latent.columns if col.startswith('x_')]
    results = []

    for client_id, group in data_latent.groupby('client_id'):
        group_sorted = group.sort_values('label')  # Ensure chronological order
        vectors = group_sorted[features].values
        labels = group_sorted['label'].values

        if len(vectors) < 2:
            continue  # Skip clients with only one label

        # First and last vectors
        first_vector = vectors[0].reshape(1, -1)

        # 2. Distance of first label to all later labels
        for i in range(1, len(vectors)):
            if skip_middle and i == int(len(labels)*0.5):
                continue
            res = {
                'client_id': client_id,
                'label': labels[i],
                'compared_to': labels[0],
                'drift_type': 'first_vs_later',
                'cosine_distance': cosine_distances(vectors[i].reshape(1, -1), first_vector)[0][0],
            }
            results.append(res)

        # 3. Rolling (Consecutive)
        for i in range(len(vectors) - 1):
            if skip_middle and i == int(len(labels)*0.5 - 1):
                continue
            results.append({
                'client_id': client_id,
                'label': labels[i],
                'compared_to': labels[i + 1],
                'drift_type': 'rolling',
                'cosine_distance': cosine_distances(vectors[i].reshape(1, -1), vectors[i + 1].reshape(1, -1))[0][0],
            })

    return pd.DataFrame(results)


def rank_clients_by_drift(df, periods):
    # 1. Filter out unwanted labels
    # df = df[~df['label'].isin([11, 12])].copy()

    # 3. Store results in a dictionary
    ranked_results = {}

    for period_name, valid_labels in periods.items():
        df_period = df[df['label'].isin(valid_labels)].copy()

        # === a) Per-label RANKING ===
        df_period['rank'] = df_period.groupby(['drift_type', 'label'])['cosine_distance']\
                                     .rank(method='dense', ascending=False)

        # === b) Summary STATS (mean and sum per client) ===
        agg_stats = df_period.groupby(['drift_type', 'client_id'])['cosine_distance']\
                             .agg(['mean', 'sum'])\
                             .rename(columns={'mean': 'mean_distance', 'sum': 'sum_distance'})\
                             .reset_index()

        # === c) Rank clients by mean and sum ===
        agg_stats['rank_by_mean'] = agg_stats.groupby('drift_type')['mean_distance']\
                                             .rank(method='dense', ascending=False)

        agg_stats['rank_by_sum'] = agg_stats.groupby('drift_type')['sum_distance']\
                                            .rank(method='dense', ascending=False)

        # Store both the detailed and summary DataFrames
        ranked_results[period_name] = {
            'detailed': df_period,
            'summary': agg_stats
        }

    return ranked_results


def analyze_experiments_by_epoch(data, periods = None, epochs = None, skip_middle = False, top_n = 1):
    if periods is None:
        periods = {
            'all_periods': list(range(0, 12))
        }

    results = {}

    for experiment in data:
        results[experiment] = {}

        data_path = os.path.join(experiment, 'Baseline_latent_space.parquet')
        new_latent = pd.read_parquet(data_path)
        new_latent['epoch'] = new_latent['epoch'].astype(int)

        experiment_summary_rows = []

        epochs_loop = epochs if epochs is not None else new_latent['epoch'].sort_values().unique()
        for epoch in epochs_loop:
            results[experiment][epoch] = {}

            data_epoch = new_latent[
                (new_latent['epoch'] == epoch) &
                (new_latent['client_id'].str.contains('_proto'))
            ]

            if data_epoch.empty:
                print(f'    ⚠ No data for epoch {epoch}, skipping...')
                continue

            df_drift = client_self_cosine_drift(data_epoch, skip_middle = skip_middle)
            ranked = rank_clients_by_drift(df_drift, periods)

            compiled_summaries = {}

            for period_name, result in ranked.items():
                summary = result['summary'].copy()

                summary['experiment'] = experiment
                summary['epoch'] = epoch
                summary['period'] = period_name

                # Rank delta
                summary['rank_diff'] = (summary['rank_by_mean'] - summary['rank_by_sum']).abs()

                # Collect for experiment-level summary
                experiment_summary_rows.append(summary)

                # Top-N for both rank types
                top_by_mean = summary.sort_values(['drift_type', 'rank_by_mean']).groupby('drift_type').head(top_n)
                top_by_mean['rank_type'] = 'mean'

                top_by_sum = summary.sort_values(['drift_type', 'rank_by_sum']).groupby('drift_type').head(top_n)
                top_by_sum['rank_type'] = 'sum'

                compiled = pd.concat([top_by_mean, top_by_sum], ignore_index=True)
                compiled.sort_values(by=['drift_type', 'rank_type'], inplace=True)

                compiled_summaries[period_name] = compiled

            results[experiment][epoch]['ranked'] = ranked
            results[experiment][epoch]['compiled_summary'] = compiled_summaries


        # 🔄 Per-Experiment Global Analysis
        if experiment_summary_rows:
            exp_df = pd.concat(experiment_summary_rows, ignore_index=True)

            # === 1. Average Ranks + Rank Delta ===
            avg_ranks = exp_df.groupby(['client_id', 'drift_type'])[
                ['rank_by_mean', 'rank_by_sum', 'rank_diff']
            ].mean().reset_index()
            avg_ranks.rename(columns={
                'rank_by_mean': 'avg_rank_by_mean',
                'rank_by_sum': 'avg_rank_by_sum',
                'rank_diff': 'avg_rank_delta'
            }, inplace=True)

            # === 2. Stability Index (Top-N appearances) ===
            stability_counts = defaultdict(lambda: {'mean': 0, 'sum': 0})
            for _, row in exp_df.iterrows():
                if row['rank_by_mean'] <= top_n:
                    stability_counts[(row['client_id'], row['drift_type'])]['mean'] += 1
                if row['rank_by_sum'] <= top_n:
                    stability_counts[(row['client_id'], row['drift_type'])]['sum'] += 1

            stability_df = pd.DataFrame([
                {
                    'client_id': cid,
                    'drift_type': drift_type,
                    'stability_topN_mean': counts['mean'],
                    'stability_topN_sum': counts['sum']
                }
                for (cid, drift_type), counts in stability_counts.items()
            ])

            # === 3. Combine metrics
            experiment_metrics = avg_ranks.merge(stability_df, on=['client_id', 'drift_type'], how='outer')
            results[experiment]['summary_metrics'] = experiment_metrics
            results[experiment]['all_summary_df'] = exp_df

        else:
            results[experiment]['summary_metrics'] = pd.DataFrame()
            results[experiment]['all_summary_df'] = pd.DataFrame()

    return results


def extract_best_clients_by_summary_metrics(results_dict, periods_list, exp_directory = 'results\\'):
    """
    Extract the top client (lowest avg_rank_by_mean) per drift_type for each experiment,
    based on the summary_metrics table.

    Parameters:
        results_dict (dict): Structure like:
            results[experiment]['summary_metrics']

    Returns:
        pd.DataFrame: DataFrame with top clients per drift_type per experiment,
                      plus drift detection diagnostics.
    """
    top_clients_all_experiments = []

    for experiment_name, experiment_data in results_dict.items():
        summary_metrics = experiment_data.get('summary_metrics')

        if summary_metrics is None or summary_metrics.empty:
            continue

        # For each drift_type, get the client with the lowest avg_rank_by_mean
        top_clients_df = (
            summary_metrics
            .sort_values(['drift_type', 'avg_rank_by_mean'])
            .groupby('drift_type')
            .head(1)
            .copy()
        )

        top_clients_df['experiment'] = experiment_name
        top_clients_df['period'] = 'all_year'

        # Retain only necessary columns
        top_clients_all_experiments.append(top_clients_df[['client_id', 'drift_type', 'experiment', 'period']])

    # Combine all experiments into one DataFrame
    summary_df = pd.concat(top_clients_all_experiments, ignore_index=True)

    # Add derived/diagnostic columns
    summary_df['client_drift'] = summary_df['client_id'].str.split('_').str[1]
    summary_df['ground_truth'] =  summary_df['experiment'].str.replace(exp_directory, '', regex=False).str[0]
    summary_df['check'] = summary_df['client_drift'] == summary_df['ground_truth']

    return summary_df


def extract_best_clients_by_ranked_summary_initial(results_dict, periods_list, exp_directory = 'results\\', rank_func = 'mean'):
    """
    For each experiment, extract clients that most frequently rank highest (rank_by_mean == 1)
    across epochs, per drift_type.

    Parameters:
        results_dict (dict): Nested structure:
            results[experiment][epoch]['ranked']['first_4_months']['summary']

    Returns:
        pd.DataFrame: DataFrame with top clients per drift_type per experiment, plus diagnostics.
    """
    top_clients_all_experiments = []

    for period in periods_list: #['first_4_months','middle_year', 'final_year']:
        for experiment_name, experiment_data in results_dict.items():
            top_rows_per_epoch = []

            for epoch in list(experiment_data.keys())[:-2]:

                summary_df = experiment_data[epoch]['ranked'][period]['summary']
                if summary_df.empty:
                    continue

                top_rank_df = summary_df[summary_df[f'rank_by_{rank_func}'] == 1].copy()
                top_rank_df['epoch'] = epoch
                top_rows_per_epoch.append(top_rank_df)

            # Skip experiment if no qualifying rows
            if not top_rows_per_epoch:
                continue

            # Combine across epochs
            combined_epoch_df = pd.concat(top_rows_per_epoch, ignore_index=True)

            # Count appearances of each client at rank 1 per drift_type
            client_rank_counts = (
                combined_epoch_df.groupby(['drift_type', 'client_id'])
                .size()
                .reset_index(name='count')
            )

            # Select clients with the highest count per drift_type
            max_counts_per_drift = client_rank_counts.groupby('drift_type')['count'].transform('max')
            top_clients_df = client_rank_counts[client_rank_counts['count'] == max_counts_per_drift].copy()
            top_clients_df['experiment'] = experiment_name
            top_clients_df['period'] = period

            top_clients_all_experiments.append(top_clients_df[['client_id', 'drift_type', 'experiment', 'period']])

    # Combine results from all experiments
    summary_df = pd.concat(top_clients_all_experiments, ignore_index=True)

    # Add diagnostic columns
    summary_df['client_drift'] = summary_df['client_id'].str.split('_').str[1]
    summary_df['ground_truth'] =  summary_df['experiment'].str.replace(exp_directory, '', regex=False).str[0]
    summary_df['check'] = summary_df['client_drift'] == summary_df['ground_truth']

    return summary_df



def extract_best_clients_by_max_cosine_distance(results_dict, periods_list, exp_directory = 'results\\'):
    """
    For each experiment and epoch, identify clients with the maximum cosine_distance
    per drift_type, and count which clients appear most often across epochs.

    Parameters:
        results_dict (dict): Nested results dictionary with structure:
            results[experiment][epoch]['ranked']['first_4_months']['detailed']

    Returns:
        pd.DataFrame: Summary DataFrame with top clients per drift_type per experiment,
                      and additional columns for analysis.
    """
    best_clients_all_experiments = []

    for period in periods_list: #['first_4_months','middle_year', 'final_year']:

        for experiment_name, experiment_data in results_dict.items():
            max_cosine_rows_per_epoch = []

            for epoch in list(experiment_data.keys())[:-2]:
                detailed_df = experiment_data[epoch]['ranked'][period]['detailed']

                # Keep only rank 1 entries
                rank_1_df = detailed_df[detailed_df['rank'] == 1]

                # For each drift_type, select the row with max cosine_distance
                max_cosine_df = rank_1_df.loc[
                    rank_1_df.groupby('drift_type')['cosine_distance'].idxmax()
                ].copy()

                max_cosine_df['epoch'] = epoch
                max_cosine_rows_per_epoch.append(max_cosine_df)

            # Combine epochs for this experiment
            combined_epochs_df = pd.concat(max_cosine_rows_per_epoch)

            # Count how many times each client appears as top cosine_distance per drift_type
            client_counts = (
                combined_epochs_df.groupby(['drift_type', 'client_id'])
                .size()
                .reset_index(name='count')
            )

            # Keep only clients with the highest count per drift_type
            max_count_per_drift = client_counts.groupby('drift_type')['count'].transform('max')
            top_clients = client_counts[client_counts['count'] == max_count_per_drift].copy()
            top_clients['experiment'] = experiment_name
            top_clients['period'] = period

            best_clients_all_experiments.append(top_clients[['client_id', 'drift_type', 'experiment', 'period']])


    # Combine across all experiments
    summary_df = pd.concat(best_clients_all_experiments, ignore_index=True)

    # Add derived/diagnostic columns
    summary_df['client_drift'] = summary_df['client_id'].str.split('_').str[1]
    summary_df['ground_truth'] =  summary_df['experiment'].str.replace(exp_directory, '', regex=False).str[0]
    summary_df['check'] = summary_df['client_drift'] == summary_df['ground_truth']

    return summary_df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_extraction_methods(setup_dict, results, extractors, periods_list, exp_directory = 'results\\'):
    """
    Evaluate multiple extraction methods on different experiment setups,
    return a DataFrame with normalized accuracy results, and plot them.

    Parameters:
        setup_dict (dict): Mapping from setup name to list of experiment names.
        results (dict): Main results dictionary.
        extractors (list of tuples): List of (label, extraction_function).

    Returns:
        pd.DataFrame: Combined summary of correct/wrong % per drift_type, setup, and method.
    """


    all_ct_results = []
    summary_list = []
    for setup_name, experiment_list in setup_dict.items():
        experiment_subset = {exp: results[exp] for exp in experiment_list}

        for label, extract_func in extractors:
            # print(f'{setup_name} - {label.upper()}')
            if 'rank' in label:
                rank_func = label.split('_')[-1]
                summary_df = extract_func(results_dict = experiment_subset,
                                          periods_list = periods_list,
                                          exp_directory = exp_directory,
                                          rank_func = rank_func)
            else:
                summary_df = extract_func(results_dict = experiment_subset,
                                          periods_list = periods_list,
                                          exp_directory = exp_directory)

            summary_df['exp_id'] = summary_df['experiment'].str.split('__').str[0].str[-5:] + '__' + summary_df['experiment'].str.split('__').str[1].str[:14]

            summary_list.append(summary_df)
            # Crosstab: % correct and wrong per drift_type
            crosstab = (
                pd.crosstab([summary_df['exp_id'], summary_df['period'], summary_df['drift_type']], summary_df['check'], normalize='index') * 100
            ).round(2)

            crosstab.rename(columns={True: 'correct_%', False: 'wrong_%'}, inplace=True)
            # crosstab['drift_type'] = crosstab.index
            crosstab['setup'] = setup_name
            crosstab['method'] = label
            all_ct_results.append(crosstab.reset_index())

    # Combine into single DataFrame
    ct_summary_df = pd.concat(all_ct_results, ignore_index=True)

    return ct_summary_df, summary_list







####----//----########----//----########----//----########----//----########----//----########----//----####

# PLOT DEFINITIONS

####----//----########----//----########----//----########----//----########----//----########----//----####


def generate_colors(num_colors, palette = 'tab20'):
    """Generate a list of distinct colors."""
    colors = plt.colormaps[palette](np.linspace(0, 1, num_colors))  # Use a colormap with distinct colors
    return [f'rgba({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)}, 0.8)' for i in
            range(num_colors)]


def plot_palettes(palette1, palette2):
    plt.figure(figsize = (8, 2))
    for i, color in enumerate(palette1.values()):
        plt.fill([i, i + 1, i + 1, i], [0, 0, 1, 1], color = color)
        plt.text(i + 0.5, 0.5, color, ha = 'center', va = 'center', color = 'black', fontsize = 10)
    plt.title('Palette NODE')
    plt.xticks(range(len(palette1)), [])
    plt.yticks([])
    plt.show()

    plt.figure(figsize = (8, 2))
    for i, color in enumerate(palette2.values()):
        plt.fill([i, i + 1, i + 1, i], [0, 0, 1, 1], color = color)
        plt.text(i + 0.5, 0.5, color, ha = 'center', va = 'center', color = 'black', fontsize = 10)
    plt.title('Palette LINK')
    plt.xticks(range(len(palette2)), [])
    plt.yticks([])
    plt.show()


def reduce_intensity(color_hex, factor):
    # Convert hex to RGB
    r, g, b = tuple(int(color_hex[i:i + 2], 16) for i in (1, 3, 5))

    # Reduce intensity
    # r = int(r * factor)
    # g = int(g * factor)
    # b = int(b * factor)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    # Ensure RGB values are within the valid range
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    # Convert RGB back to hex
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def generate_pallete(variables = [], pallete = 'Accent', factor = 0.6, show = False):
    n_colors = len(variables)
    if not n_colors: return None, None

    cmap = matplotlib.colormaps.get_cmap(pallete)
    # norm = mcolors.Normalize(vmin=0, vmax=1)

    values = np.linspace(0, 1, n_colors)

    colors = [cmap(value) for value in values]
    colors_hex = [mcolors.to_hex(color) for color in colors]

    reduced_colors_hex = [reduce_intensity(c, factor) for c in colors_hex]

    node_dict = {}
    link_dict = {}

    for i, var in enumerate(variables):
        node_dict[var] = colors_hex[i]
        link_dict[var] = reduced_colors_hex[i]

    if show: plot_palettes(node_dict, link_dict)

    return node_dict, link_dict


####----//----########----//----########----//----########----//----########----//----########----//----####

# EXPLORATORY ANALYSIS

####----//----########----//----########----//----########----//----########----//----########----//----####

####****////****########****////****########****////****####
# BAR PLOT
####****////****########****////****########****////****####

def create_barplot(ax, crosstab_melted, var_x, var_hue, features_map, plot_map):
    """Creates a barplot with normalized values."""
    # Normalize data
    crosstab_melted['Normalized'] = crosstab_melted.groupby(var_x)['Count'].transform(lambda x: x / x.sum())

    # Plot the barplot
    sns.barplot(
        data = crosstab_melted,
        x = var_x,
        y = 'Normalized',
        hue = var_hue,
        hue_order = features_map[var_hue]['order'],
        palette = features_map[var_hue]['palette'],
        ax = ax
        )

    # Apply plot map settings
    ax.set_title(
        plot_map['title_text'].format(var_x = features_map[var_x]['label'], var_hue = features_map[var_hue]['label']),
        fontsize = plot_map['title_fontsize'])
    ax.set_xlabel(features_map[var_x]['label'], fontsize = plot_map['label_fontsize'])
    ax.set_ylabel(plot_map['y_label_text'], fontsize = plot_map['label_fontsize'])

    # Customize tick font sizes
    ax.tick_params(axis = 'x', labelsize = plot_map['tick_fontsize'])
    ax.tick_params(axis = 'y', labelsize = plot_map['tick_fontsize'])

    # Customize legend font size
    ax.legend(title = features_map[var_hue]['label'], title_fontsize = plot_map['legend_fontsize'] + 4,
              fontsize = plot_map['legend_fontsize'])
    # legend.set_title(var_hue)  # Set legend title explicitly


def create_crosstab_barplot(df, var_x, var_hue, features_map, plot_map, filename = None):
    """
    Creates a cross-tabulation and plots barplots using two variables.
    """
    # Step 1: Create a crosstab and transform to long format
    crosstab = pd.crosstab(df[var_x], df[var_hue])
    crosstab_reset = crosstab.reset_index()
    crosstab_melted = pd.melt(crosstab_reset, id_vars = [var_x], var_name = var_hue, value_name = 'Count')

    # Step 2: Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize = (14, 6), sharey = True)

    # Step 3: Plot using create_barplot function
    create_barplot(axs[0], crosstab_melted, var_x, var_hue, features_map, plot_map)

    create_barplot(axs[1], crosstab_melted, var_hue, var_x, features_map, plot_map)

    # Save and show plot
    plt.tight_layout()

    if filename is not None: plt.savefig(filename, dpi = 300, bbox_inches = 'tight', transparent = True)

    plt.show()


####****////****########****////****########****////****####
# SANKLEY PLOT
####****////****########****////****########****////****####

def define_flow(df, color_dict, nodes, links = None):
    # TODO implement default link color case

    flow = df.copy().sort_values(by = links)

    for case in ['link', 'node']:
        flow[case + '_color'] = flow[links].apply(
            lambda x: color_dict.get(x.split('__')[0], {})[case].get(x.split('__')[1], 'rgba(128, 128, 128, 0.8)')
            )

    # flow = flow.sort_values(by = links) .drop(columns = links)
    # flow.rename(columns = {nodes[0]: 'source', nodes[1]: 'target'}, inplace = True)

    flow['source'] = flow[nodes[0]]
    flow['target'] = flow[nodes[1]]

    return flow  #[['source', 'target', 'Count', 'link_color', 'node_color']]


def create_sankley(df, dict_mapping, features, unique_nodes, nodes, links, agg = False):
    # TODO implement logic to handle the flow definition automatically

    cross_A = define_flow(df = df, color_dict = dict_mapping, nodes = [features[c] for c in nodes[0]],
                          links = features[links[0]])
    cross_B = define_flow(df = df, color_dict = dict_mapping, nodes = [features[c] for c in nodes[1]],
                          links = features[links[1]])

    df_cross = pd.concat([cross_A, cross_B], axis = 0)
    # return df_cross, cross_A, cross_B

    # unique_nodes = pd.Series(df_cross[['source', 'target']].values.ravel()).unique()
    node_indices = {node: index for index, node in enumerate(unique_nodes)}
    color_nodes = [
        dict_mapping.get(n.split('__')[0], {})['node'].get(n.split('__')[1], 'rgba(128, 128, 128, 0.8)').replace('.8)',
                                                                                                                 '1)')
        for n in unique_nodes]

    # return df_cross

    # Create the source-target format
    df_cross['Source'] = df_cross['source'].map(node_indices)
    df_cross['Target'] = df_cross['target'].map(node_indices)
    # df_cross['link_color'] = df_cross['link_color'].str.replace('.8)', '.5)')

    if agg:
        df_cross = df_cross[['Source', 'Target', 'link_color', 'Count']]
        df_cross = df_cross.groupby(df_cross.columns[:-1].tolist()).sum().reset_index()

    fig = go.Figure(go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = unique_nodes,
            color = color_nodes,
            ),
        link = dict(
            source = df_cross['Source'],  # Source nodes
            target = df_cross['Target'],  # Target nodes
            value = df_cross['Count'],  # Flow values
            color = df_cross['link_color']
            )
        ))

    fig.update_layout(
        # title_text = "Sankey Diagram",
        font_size = 16,
        width = 1000,  # Set width in pixels
        height = 500,  # Set height in pixels
        font_color = "black",
        paper_bgcolor = 'rgba(0,0,0,0)',  # Background of the entire figure
        plot_bgcolor = 'rgba(0,0,0,0)'  # Background of the plotting area
        )
    fig.show()

    # if agg:
    #     fig.write_image("sankey_diagram_agg.png")
    # else:
    #     fig.write_image("sankey_diagram.png")
    return df_cross


def plot_sankley(df, features_map):
    features = list(features_map.keys())
    crosstab = df[features].copy()

    # Create unique labels by combining column names and their values
    for feature in features:
        crosstab[feature] = crosstab[feature].astype(str)  # Ensure all values are string for concatenation
        crosstab[feature] = feature + '__' + crosstab[feature]  # Append column name to each value

    crosstab = crosstab[features].value_counts().reset_index(name = 'Count')

    dict_colors = {}
    unique_nodes = []
    for feature, params in features_map.items():
        # TODO improve this pseudologic
        cat_order = [feature + '__' + v for v in params['order']]
        unique_nodes += cat_order
        cat_type = pd.CategoricalDtype(categories = cat_order, ordered = True)
        crosstab[feature] = crosstab[feature].astype(cat_type)
        crosstab = crosstab.sort_values(feature)

        node_color, link_color = generate_pallete(variables = params['order'],
                                                  pallete = params['palette'],
                                                  factor = .25,
                                                  # show = True
                                                  )

        dict_colors[feature] = {'node': node_color,
                                'link': link_color}

    _ = create_sankley(df = crosstab, dict_mapping = dict_colors, features = features, unique_nodes = unique_nodes,
                      nodes = [(0, 1), (1, 2)], links = [1, 0])
    _ = create_sankley(df = crosstab, dict_mapping = dict_colors, features = features, unique_nodes = unique_nodes,
                      nodes = [(0, 1), (1, 2)], links = [0, 2])
    _ = create_sankley(df = crosstab, dict_mapping = dict_colors, features = features, unique_nodes = unique_nodes,
                      nodes = [(0, 1), (1, 2)], links = [0, 1], agg = True)


# ## PROCESSAMENTO LATENTES SE PRECISAR
#
# import os
# import pickle
# import warnings
#
# import pandas as pd
# from tqdm import tqdm
# from collections import defaultdict
# from datasets import get_private_dataset
# from utils.Server import local_evaluate
# from utils.Toolbox_analysis import create_latent_df
#
#
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# new_list = folders_density_P2[651:]
# iterator = tqdm(new_list)
#
# for results_dir in iterator:
#
#     results_path = f"{results_dir}/results.pkl"
#     with open(results_path, 'rb') as f:
#         results = pickle.load(f)
#
#
#     args = results['Baseline']['args']
#     label_clients = [
#         'District_A', 'District_B', 'District_C', 'District_D', 'District_E',
#         'District_2A', 'District_2B', 'District_2C'
#     ]
#
#     priv_dataset = get_private_dataset(args)
#     backbones_list = priv_dataset.get_backbone(
#         parti_num=args.parti_num,
#         names_list=None,
#         n_series=args.input_size
#     )
#
#     train_DL = priv_dataset.get_data_loaders()
#     base_index = train_DL[0]['X_index']
#     latent_dfs = {}
#     scenarios = ['Baseline']
#
#     all_latents = defaultdict(list)
#
#     for scenario in scenarios:
#         global_model_history = results[scenario]['model']['global_weights_history']
#
#         for epoch in range(args.communication_epoch):
#             aux_latents = []
#             state_dict = global_model_history[epoch]
#             for net in backbones_list:
#                 net.load_state_dict(state_dict)
#
#             latent_spaces = local_evaluate(model=backbones_list,
#                                            train_dl=train_DL,
#                                            private_dataset=priv_dataset,
#                                            group_detections=False,
#                                            detect_anomalies=False)
#
#             for i, client in enumerate(latent_spaces):
#                 client_lat = create_latent_df(
#                     X_index=base_index,
#                     x_lat=client,
#                     label=f"{label_clients[i]}__{epoch}",
#                     is_unix=True
#                 )
#                 aux_latents.append(client_lat)
#
#             data_latent = pd.concat(aux_latents)
#             data_latent[['client_id', 'epoch']] = data_latent['label'].str.split('__', expand=True)
#
#             # Extract year and month
#             years = data_latent['timestamp'].dt.year
#             months = data_latent['timestamp'].dt.month
#
#             # Compute base year from the minimum timestamp
#             base_year = data_latent['timestamp'].min().year
#
#             # Continuous month index
#             data_latent['label'] = (years - base_year) * 12 + (months - 1)
#
#             id_cols = ['client_id', 'label', 'epoch']
#             feat_cols = [col for col in data_latent.columns if 'x_' in col]
#             aux_agg = data_latent[id_cols + feat_cols]
#             aux_agg = aux_agg.groupby(id_cols).mean().reset_index()
#             aux_agg = aux_agg.sort_values(by=['epoch', 'label', 'client_id']).reset_index(drop=True)
#             aux_agg['client_id'] += '__proto'
#
#             merged_latent = pd.concat([data_latent[['timestamp'] + id_cols + feat_cols], aux_agg])
#             merged_latent = merged_latent.reset_index(drop=True)
#
#             all_latents[scenario].append(merged_latent)
#
#
#
#     for scenario in scenarios:
#         pd.concat(all_latents[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_latent_space.parquet", index=False)
