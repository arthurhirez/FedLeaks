import os
import pandas as pd
import altair as alt
from sklearn.preprocessing import MinMaxScaler

alt.data_transformers.enable("vegafusion")

def format_latent_dict(latent_dfs):
    """Adds timestamp columns and renames latent dimensions for PCA/UMAP outputs."""
    for case in latent_dfs.values():
        for epoch in case.values():
            date_cols = epoch['latent_space'].iloc[:, :4].reset_index(drop=True)
            for space in epoch.keys():
                if ('pca' in space) or ('umap' in space):
                    epoch[space] = pd.concat([date_cols, epoch[space]], axis=1)
                    epoch[space].columns = date_cols.columns.tolist() + ['latent_1', 'latent_2']

def load_and_scale_data(id_network, id_experiment, tgt_district = 'District_A'):
    """Loads client data, parses timestamps, and scales features."""
    df = pd.read_csv(f'datasets/leaks/{id_network}/{id_experiment}/{tgt_district.replace("District_", "Client")}_Baseline.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(df.iloc[:, 1:])
    scaled_df = pd.DataFrame(scaled, columns=df.columns[1:])
    scaled_df.insert(0, 'timestamp', df['timestamp'])

    return scaled_df


def combine_latents(latent_dfs):
    """Combines all epoch latent data into a single DataFrame with metadata."""
    df_all = []

    for epoch, df_epoch in latent_dfs['Baseline'].items():
        for method in ['pca_scl', 'umap_scl']:
            df = df_epoch[method].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['epoch'] = epoch
            df['method'] = method
            df_all.append(df)

    df_combined = pd.concat(df_all, ignore_index=True)
    df_combined['month'] = df_combined['timestamp'].dt.month

    df_combined['hour'] = df_combined['timestamp'].dt.hour
    df_combined['hour_filter'] = df_combined['hour'].apply(lambda x: x - 12 if x >= 12 else x)

    return df_combined


def plot_latent_heatmap(df_combined, results_id):
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
        x=alt.X('latent_1:Q', bin=alt.Bin(maxbins=100), title="Latent x"),
        y=alt.Y('latent_2:Q', bin=alt.Bin(maxbins=100), title="Latent y"),
        color=alt.Color('value:N', scale=alt.Scale(scheme='tableau20'), title='Color'),
        tooltip=['label:N', 'month:Q', 'hour:Q', 'epoch:Q']
    ).add_params(
        epoch_select, month_selection, hour_selection, method_selection, color_field_selection
    ).properties(
        width=350,
        height=350,
        title="Interactive Latent Space Heatmap (Color-coded)"
    ).interactive()

    folded.save(f'results/imgs/Heat_{results_id}_proto.html')


def plot_time_series_and_latents(df_combined, scaled_df, results_id, batch_temporal=14):
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
    x_range = [int(df_combined['latent_1'].min()) - 3, int(df_combined['latent_1'].max()) + 3]
    y_range = [int(df_combined['latent_2'].min()) - 3, int(df_combined['latent_2'].max()) + 3]
    
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
        x=alt.X('latent_1:Q', bin=alt.Bin(maxbins=100), title="Latent x"),
        y=alt.Y('latent_2:Q', bin=alt.Bin(maxbins=100), title="Latent y"),
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
    final_plot.save(f'results/imgs/Series_{results_id}.html')
