import glob
import os
import pandas as pd
import altair as alt
from sklearn.preprocessing import MinMaxScaler

alt.data_transformers.enable("vegafusion")

# def format_latent_dict(latent_dfs):
#     """Adds timestamp columns and renames latent dimensions for PCA/UMAP outputs."""
#     for case in latent_dfs.values():
#         for epoch in case.values():
#             date_cols = epoch['latent_space'].iloc[:, :4].reset_index(drop=True)
#             for space in epoch.keys():
#                 if ('pca' in space) or ('umap' in space):
#                     epoch[space] = pd.concat([date_cols, epoch[space]], axis=1)
#                     epoch[space].columns = date_cols.columns.tolist() + ['latent_x', 'latent_y']

def load_and_scale_data(id_network, id_experiment, tgt_district = 'District_A'):
    """Loads client data, parses timestamps, and scales features."""
    df = pd.read_csv(f'datasets/leaks/{id_network}/{id_experiment}/{tgt_district.replace("District_", "Client")}_Baseline.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(df.iloc[:, 1:])
    scaled_df = pd.DataFrame(scaled, columns=df.columns[1:])
    scaled_df.insert(0, 'timestamp', df['timestamp'])

    return scaled_df


def combine_latents(results_dir):
    """Combines all epoch latent data into a single DataFrame with metadata."""

    parquet_files = glob.glob(os.path.join(results_dir, '**', '*.parquet'), recursive=True)
    pca_umap_files = [f for f in parquet_files if 'pca' in os.path.basename(f).lower() or 'umap' in os.path.basename(f).lower()]
    

    df_all = []

    for file in pca_umap_files:
        df = pd.read_parquet(file)
        df_all.append(df)

    df_combined = pd.concat(df_all, ignore_index=True)
    
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.rename(columns = {'label' : 'month', 'client_id' : 'label'}, inplace = True)
    # df_combined['month'] = df_combined['timestamp'].dt.month

    int_cols = ['epoch', 'month']
    for c in int_cols:
        df_combined[c] = df_combined[c].astype(int)

    df_combined['hour'] = df_combined['timestamp'].dt.hour
    df_combined['hour_filter'] = df_combined['hour'].apply(lambda x: x - 12 if x >= 12 else x)

    return df_combined


def plot_latent_heatmap(df_combined, results_dir):
    """Creates and saves an interactive heatmap of the latent space.
    """
    epoch_slider = alt.binding_range(min=df_combined['epoch'].min(), max=df_combined['epoch'].max(), step=1, name='Epoch: ')
    epoch_select = alt.selection_point(fields=['epoch'], bind=epoch_slider, value=0)

    hour_options = [None] + sorted(df_combined['hour_filter'].unique().tolist())
    hour_selection = alt.selection_point(fields=['hour_filter'], bind=alt.binding_select(options=hour_options, name=f'Hour: '), value=None)

    month_options = [None] + sorted(df_combined['month'].unique().tolist())
    month_selection = alt.selection_point(fields=['month'], bind=alt.binding_select(options=month_options, name=f'Month: '), value=None)

    method_options = df_combined['method'].unique().tolist()
    method_selection = alt.selection_point(fields=['method'], bind=alt.binding_select(options=method_options, name='Method: '), value=method_options[0])

    color_field_selection = alt.selection_point(fields=['key'], bind=alt.binding_select(options=['label', 'hour_filter', 'month'], name='Color by: '), value='label')

    folded = alt.Chart(df_combined).transform_filter(
        epoch_select & month_selection & hour_selection & method_selection
    ).transform_fold(
        ['label', 'hour_filter', 'month'], as_=['key', 'value']
    ).transform_filter(
        color_field_selection
    ).mark_rect().encode(
        x=alt.X('latent_x:Q', bin=alt.Bin(maxbins=100), title="Latent x"),
        y=alt.Y('latent_y:Q', bin=alt.Bin(maxbins=100), title="Latent y"),
        color=alt.Color('value:N', scale=alt.Scale(scheme='tableau20'), title='Color'),
        tooltip=['label:N', 'month:Q', 'hour:Q', 'epoch:Q']
    ).add_params(
        epoch_select, month_selection, hour_selection, method_selection, color_field_selection
    ).properties(
        width=350,
        height=350,
        title="Interactive Latent Space Heatmap (Color-coded)"
    ).interactive()

    folded.save(f'{results_dir}/Heatmap.html')


def plot_time_series_and_latents(df_combined, scaled_df,  results_dir, batch_temporal=14):
    """Creates and saves the combined plot of time-series and filtered latent space.
    :param period:
    """
    melted_df = scaled_df.melt(id_vars='timestamp', var_name='feature', value_name='value')
    melted_df['timestamp'] = pd.to_datetime(melted_df['timestamp'])
    melted_df['month'] = melted_df['timestamp'].dt.month

    melted_df['hour'] = melted_df['timestamp'].dt.hour
    melted_df['hour_filter'] = melted_df['hour'].apply(lambda x: x - 12 if x >= 12 else x)

    # Offset time-series values for stacking
    unique_features = melted_df['feature'].unique()
    offset_dict = {feature: i * 2 for i, feature in enumerate(unique_features)}
    melted_df['offset_value'] = melted_df.apply(lambda row: row['value'] + offset_dict[row['feature']], axis=1)

    # --- Selections ---
    start_ts = melted_df['timestamp'].min()
    end_ts = start_ts + pd.Timedelta(days=batch_temporal)
    brush = alt.selection_interval(encodings=['x'], value={'x': (start_ts, end_ts)})
    latent_selection = alt.selection_point(fields=['timestamp'], value=melted_df['timestamp'].min())

    hour_options = [None] + sorted(df_combined['hour_filter'].unique().tolist())
    hour_selection = alt.selection_point(
        name=f"Select Hour:",
        fields=['hour_filter'],
        bind=alt.binding_select(options=hour_options, name=f'Hour: '),
        value=None)

    month_options = [None] + sorted(df_combined['month'].unique().tolist())
    month_selection = alt.selection_point(
        name=f"Select Month:",
        fields=['month'],
        bind=alt.binding_select(options=month_options, name=f'Month: '),
        value=None)

    method_options = df_combined['method'].unique().tolist()
    method_selection = alt.selection_point(fields=['method'], bind=alt.binding_select(options=method_options, name='Method: '), value=method_options[0])

    # --- Time Series Chart ---
    base = alt.Chart(melted_df).mark_line().encode(
        x='timestamp:T',
        y='offset_value:Q',
        color='feature:N'
    ).properties(width=500)

    points = alt.Chart(melted_df).mark_circle(color='black', size=5).encode(
        x='timestamp:T',
        y='offset_value:Q',
        tooltip=['feature:N', 'timestamp:T']
    ).transform_filter(month_selection & hour_selection)

    upper = (base + points).encode(
        x=alt.X('timestamp:T', scale=alt.Scale(domain=brush))
    ).properties(height=200)

    lower = base.properties(height=60).add_params(brush)

    time_series_chart = upper & lower

    # --- Latent Space Chart ---
    x_range = [int(df_combined['latent_x'].min()) - 3, int(df_combined['latent_x'].max()) + 3]
    y_range = [int(df_combined['latent_y'].min()) - 3, int(df_combined['latent_y'].max()) + 3]
    
    epoch_slider = alt.binding_range(min=df_combined['epoch'].min(),
                                 max=df_combined['epoch'].max(),
                                 step=1,
                                 name='Epoch: ')
    epoch_select = alt.selection_point (fields=['epoch'], bind=epoch_slider, value = 0)

    color_field_selection = alt.selection_point(fields=['key'], bind=alt.binding_select(options=['label', 'hour_filter', 'month'], name='Color by: '), value='label')

    latent = alt.Chart(df_combined).transform_filter(
        brush & epoch_select & month_selection & hour_selection & method_selection
    ).transform_fold(
        ['label', 'hour_filter', 'month'], as_=['key', 'value']
    ).transform_filter(
        color_field_selection
    ).mark_rect().encode(
        x=alt.X('latent_x:Q', bin=alt.Bin(maxbins=100), title="Latent x"),
        y=alt.Y('latent_y:Q', bin=alt.Bin(maxbins=100), title="Latent y"),
        color=alt.Color('value:N', scale=alt.Scale(scheme='tableau20'), title='Color'),
        tooltip=['label:N', 'month:Q', 'hour:Q', 'epoch:Q']
    ).add_params(
        latent_selection, epoch_select, month_selection, hour_selection, method_selection, color_field_selection
    ).properties(
        width=350,
        height=350,
        title="Interactive Latent Space Heatmap (Color-coded)"
    ).interactive()


    # --- Layout and Save ---
    final_plot = latent | time_series_chart
    final_plot.save(f'{results_dir}/Series.html')


def plot_proto_similar(df_exp_proto, results_dir):
    df = df_exp_proto.copy()
    
    for c in ['client_1', 'client_2']:
        df[c] = df[c].str.replace('__proto', '')
    # Add pair column
    df['pair'] = df['client_1'] + ' vs ' + df['client_2']
    
    # Sliders for metric weights
    cosine_slider = alt.binding_range(min=0, max=4, step=0.05, name="cosine")
    manhattan_slider = alt.binding_range(min=0, max=4, step=0.05, name="manhattan")
    wavelet_slider = alt.binding_range(min=0, max=4, step=0.05, name="wavelet")
    dft_slider = alt.binding_range(min=0, max=4, step=0.05, name="dft")
    autocorr_slider = alt.binding_range(min=0, max=4, step=0.05, name="autocorr")
    epoch_slider = alt.binding_range(min=df['epoch'].min(), max=df['epoch'].max(), step=1, name="Epoch")
    month_slider = alt.binding_range(min=df['label'].min(), max=df['label'].max(), step=1, name="Month")
    client_dropdown = alt.binding_select(options=sorted(df['client_1'].unique()), name="Client")
    
    # Define interactive parameters
    cosine_param = alt.param(name='cosine', value=1, bind=cosine_slider)
    manhattan_param = alt.param(name='manhattan', value=1, bind=manhattan_slider)
    wavelet_param = alt.param(name='wavelet', value=1, bind=wavelet_slider)
    dft_param = alt.param(name='dft', value=1, bind=dft_slider)
    autocorr_param = alt.param(name='autocorr', value=1, bind=autocorr_slider)
    epoch_select = alt.selection_point(fields=['epoch'], bind=epoch_slider, value=0)
    month_select = alt.selection_point(fields=['label'], bind=month_slider, value=1)
    client_select = alt.selection_point(fields=['client_1'], bind=client_dropdown, value = 'District_A')
    
    # --- Base chart with weights ---
    base = alt.Chart(df).add_params(
        cosine_param,
        manhattan_param,
        wavelet_param,
        dft_param,
        autocorr_param,
        epoch_select,
        month_select
    ).transform_calculate(
        weighted_score="""
            datum.cosine * cosine +
            datum.manhattan * manhattan +
            datum.wavelet * wavelet +
            datum.dft * dft +
            datum.autocorr * autocorr
        """
    ).transform_filter(
        epoch_select & month_select
    )
    
    # --- Heatmap ---
    heatmap = base.mark_rect().encode(
        x=alt.X('client_1:N', title='Client 1'),
        y=alt.Y('client_2:N', title='Client 2'),
        color=alt.Color('weighted_score:Q', title='Weighted Dissimilarity', scale=alt.Scale(scheme='blueorange')),
        tooltip=['client_1', 'client_2', 'epoch', 'label', 'weighted_score:Q']
    ).properties(
        width=300,
        height=300,
        title='Client Dissimilarity Matrix (Weighted)'
    )
    
    text_labels = base.mark_text(
        align='center',
        baseline='middle',
        fontSize=14,
        fontWeight='bold'
    ).encode(
        x='client_1:N',
        y='client_2:N',
        text=alt.Text('weighted_score:Q', format='.2f'),
        color=alt.condition('datum.weighted_score > 1.0', alt.value('red'), alt.value('blue'))
    )
    
    heatmap_combined = heatmap + text_labels
    
    # --- Line chart for weighted score over time ---
    # Add all time data with weighting
    line_base = alt.Chart(df).add_params(
        cosine_param,
        manhattan_param,
        wavelet_param,
        dft_param,
        autocorr_param,
        client_select,
        epoch_select
    ).transform_calculate(
        weighted_score="""
            datum.cosine * cosine +
            datum.manhattan * manhattan +
            datum.wavelet * wavelet +
            datum.dft * dft +
            datum.autocorr * autocorr
        """
    ).transform_filter(
       client_select & epoch_select
    )
    
    line_chart = line_base.mark_line(point=True).encode(
        x=alt.X('label:N', title='Month Label'),
        y=alt.Y('weighted_score:Q', title='Weighted Score'),
        color=alt.Color('pair:N', title='Client Pair'),
        tooltip=['client_1', 'client_2', 'label:N', 'epoch:Q', 'weighted_score:Q']
    ).properties(
        width=500,
        height=300,
        title='Weighted Distance Over Time (Client Focused)'
    )
    
    # --- Combine heatmap and line chart ---
    final_chart = (heatmap_combined | line_chart).configure_title(anchor='start')
    
    final_chart.save(f'{results_dir}/Proto_Similarities.html')

    
def plot_distribution_similar(df_exp_latent, results_dir):
    # --- Prepare Data ---
    df = df_exp_latent.copy()
    df['pair'] = df['target_client'] + ' vs ' + df['other_client']
    
    # Convert similarity to dissimilarity
    df['SubspaceDiss'] = 1 - df['SubspaceAlignment']
    df['MutualInfoDiss'] = 1 - df['MutualInfo']
    
    # --- Define Metric Weights ---
    metrics = ['MMD', 'Wasserstein', 'Energy', 'SubspaceDiss', 'DTW', 'MutualInfoDiss', 'KL', 'JSD']
    bindings = {m: alt.binding_range(min=0, max=2, step=0.05, name=m) for m in metrics}
    params = [alt.param(name=m, value=0.2, bind=b) for m, b in bindings.items()]
    
    # --- Filters ---
    epoch_slider = alt.binding_range(min=df['epoch'].min(), max=df['epoch'].max(), step=1, name="Epoch")
    month_slider = alt.binding_range(min=df['label'].min(), max=df['label'].max(), step=1, name="Month")
    metric_selector = alt.binding_select(options=[None] + metrics, name="Metrics")
    
    epoch_select = alt.selection_point(fields=['epoch'], bind=epoch_slider, value=0)
    month_select = alt.selection_point(fields=['label'], bind=month_slider, value=1)
    metric_select = alt.selection_point(fields=['metric'], bind=metric_selector, value=None)
    
    # --- Base Chart with Weighted Score ---
    base = alt.Chart(df).add_params(
        *params,
        epoch_select,
        month_select
    ).transform_filter(
        epoch_select & month_select
    ).transform_calculate(
        weighted_score=" + ".join([f"datum.{m} * {m}" for m in metrics])
    )
    
    # --- Heatmap ---
    heatmap = base.mark_rect().encode(
        x='target_client:N',
        y='other_client:N',
        color=alt.Color('weighted_score:Q', title='Weighted Dissimilarity', scale=alt.Scale(scheme='viridis')),
        tooltip=['target_client', 'other_client', 'epoch', 'label', 'weighted_score:Q']
    ).properties(
        width=200,
        height=200,
        title='Latent Dissimilarity Matrix (weighted)'
    )
    
    # Text labels on heatmap
    text_labels = base.mark_text(
        align='center',
        baseline='middle',
        fontSize=14,
        fontWeight='bold'
    ).encode(
        x='target_client:N',
        y='other_client:N',
        text=alt.Text('weighted_score:Q', format='.2f'),
        color=alt.condition(
            'datum.weighted_score > 0.5',
            alt.value('white'),
            alt.value('black')
        )
    )
    
    # --- Line Chart Over Time ---
    long_df = df.melt(
        id_vars=['epoch', 'label', 'target_client', 'other_client', 'pair'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )
    
    metric_chart = alt.Chart(long_df).add_params(metric_select).transform_filter(
        metric_select
    ).mark_line(point=True).encode(
        x=alt.X('label:N', title='Month'),
        y=alt.Y('value:Q', title='Metric Value'),
        color=alt.Color('pair:N', title='Client Pair'),
        strokeDash='metric:N',
        tooltip=['target_client', 'other_client', 'label', 'metric', 'value']
    ).properties(
        width=300,
        height=200,
        title='Metric Evolution Over Time'
    ).add_params(
        epoch_select,
    ).transform_filter(
        epoch_select
    )
    
    # --- Base Chart with Weighted Score ---
    line_base = alt.Chart(df).add_params(
        *params,
        epoch_select,
    ).transform_filter(
        epoch_select
    ).transform_calculate(
        weighted_score=" + ".join([f"datum.{m} * {m}" for m in metrics])
    )
    
    line_chart = line_base.mark_line(point=True).encode(
        x=alt.X('label:N', title='Month Label'),
        y=alt.Y('weighted_score:Q', title='Weighted Score'),
        color=alt.Color('pair:N', title='Client Pair'),
        tooltip=['target_client', 'other_client', 'label:N', 'epoch:Q', 'weighted_score:Q']
    ).properties(
        width=300,
        height=200,
        title='Weighted Distance Over Time (Client Focused)'
    )
    
    # --- Combine Layout ---
    final_chart = (heatmap + text_labels | line_chart | metric_chart).configure_title(anchor='start')
    final_chart.save(f'{results_dir}/Latent_Similarities.html')