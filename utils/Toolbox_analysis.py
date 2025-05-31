import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


alt.data_transformers.enable("vegafusion")


def process_latent_df(df_latent, umap_neighbors=50, umap_min_dist=0.95, reduce_raw = False, id_cols = None, return_scaled = False):
    # Get feature columns (assumed to be latent vectors)
    feature_cols = [col for col in df_latent.columns if col.startswith('x_')]

    # Original (unscaled) features
    X_raw = df_latent[feature_cols].values

    # Scaled features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    if return_scaled:
        df_latent[feature_cols] = X_scaled

    X_pca_scaled, X_umap_scaled = reduce_dims(
        X=X_scaled,
        method=None,
        n_components=2,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist
    )

    if id_cols is None:
        df_pca_scaled = pd.DataFrame(X_pca_scaled, columns=['latent_x', 'latent_y'])
        df_umap_scaled = pd.DataFrame(X_umap_scaled, columns=['latent_x', 'latent_y'])
    else:
        df_pca_scaled = df_latent[id_cols].copy()
        df_pca_scaled[['latent_x', 'latent_y']] = X_pca_scaled

        df_umap_scaled = df_latent[id_cols].copy()
        df_umap_scaled[['latent_x', 'latent_y']] = X_umap_scaled
        
    if reduce_raw:
        # Dimensionality reduction
        X_pca_raw, X_umap_raw = reduce_dims(
            X=X_raw,
            method=None,
            n_components=2,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist
        )
        # Create DataFrames
        df_pca_raw = pd.DataFrame(X_pca_raw, columns=['latent_x', 'latent_y'])
        df_umap_raw = pd.DataFrame(X_umap_raw, columns=['latent_x', 'latent_y'])

        return df_latent, (df_pca_raw, df_umap_raw), (df_pca_scaled, df_umap_scaled)

    return df_latent, df_pca_scaled, df_umap_scaled
    

def reduce_dims(X, method=None, n_components=2, umap_neighbors=15, umap_min_dist=0.1):
    """
    Applies PCA and UMAP to the input data.

    Parameters:
    - X: array-like of shape (n_samples, n_features)
    - n_components: int, number of dimensions for the projection
    - umap_neighbors: int, UMAP neighbors
    - umap_min_dist: float, UMAP min_dist

    Returns:
    - X_pca: PCA-reduced data
    - X_umap: UMAP-reduced data
    """

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)
    if method == 'PCA':
        return X_pca, None

    reducer = umap.UMAP(n_components=n_components, n_neighbors=umap_neighbors, min_dist=umap_min_dist)
    X_umap = reducer.fit_transform(X)
    if method == 'UMAP':
        return None, X_umap

    return X_pca, X_umap


def plot_reduced(X_pca, X_umap, labels=None, figsize=(12, 5)):
    """
    Plots PCA and UMAP reduced data side-by-side.

    Parameters:
    - X_pca: array-like, PCA-reduced data
    - X_umap: array-like, UMAP-reduced data
    - labels: array-like of shape (n_samples,), optional
    - figsize: tuple, figure size
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {label: color for label, color in zip(unique_labels, colors)}

        for label in unique_labels:
            idx = labels == label
            axs[0].scatter(X_pca[idx, 0], X_pca[idx, 1], c=[label_color_map[label]], label=f"{label}", s=10)
            axs[1].scatter(X_umap[idx, 0], X_umap[idx, 1], c=[label_color_map[label]], label=f"{label}", s=10)

        axs[0].set_title("PCA Projection")
        axs[1].set_title("UMAP Projection")
        axs[0].axis("off")
        axs[1].axis("off")

        legend_elements = [Patch(facecolor=label_color_map[label], label=f'{label}') for label in unique_labels]
        axs[1].legend(handles=legend_elements)
    else:
        axs[0].scatter(X_pca[:, 0], X_pca[:, 1], s=10)
        axs[1].scatter(X_umap[:, 0], X_umap[:, 1], s=10)
        axs[0].set_title("PCA Projection")
        axs[1].set_title("UMAP Projection")
        axs[0].axis("off")
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def create_latent_df(X_index, x_lat, label='Teste', is_unix=False):
    """
    Create a DataFrame with timestamps, latent vector components, and anomaly labels.

    Parameters:
    - X_index: array of shape (n_samples,), can be UNIX timestamps or datetime
    - x_lat: array of shape (n_samples, n_latent_dims)
    - anomaly_df: DataFrame with 'start' and 'end' columns for anomaly intervals
    - is_unix: bool, if True converts X_index from UNIX timestamp to datetime

    Returns:
    - DataFrame with 'timestamp', 'label', and x_0 to x_n columns
    """
    # Step 1: Convert timestamps if needed
    if is_unix:
        timestamps = pd.to_datetime(X_index, unit='s')
    else:
        timestamps = pd.to_datetime(X_index)  # ensures consistent datetime dtype

    # Step 2: Create column names for latent dimensions
    latent_dim_names = [f'x_{i}' for i in range(x_lat.shape[1])]

    # Step 3: Create DataFrame
    df = pd.DataFrame(x_lat, columns=latent_dim_names)
    df['timestamp'] = timestamps
    df['label'] = label

    # Step 5: Reorder columns
    df = df[['timestamp', 'label'] + latent_dim_names]

    return df

def plot_reduced_method(X_pca, X_umap, labels=None, method='PCA', ax=None, title=None, figsize=(12, 5)):
    """
    Plots reduced data based on the selected method (PCA or UMAP).
    Accepts external axis (ax) for subplot usage.
    """
    # if method not in ['PCA', 'UMAP']:
    #     raise ValueError("Invalid method! Use 'PCA' or 'UMAP'.")

    if method is None:
        return plot_reduced(X_pca, X_umap, labels=labels, figsize=(12, 5))

    internal_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        internal_fig = True
    else:
        fig = ax.figure

    # Setup
    if labels is not None:
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        label_color_map = {label: color for label, color in zip(unique_labels, colors)}

        for label in unique_labels:
            idx = labels == label
            if method == 'PCA':
                ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=[label_color_map[label]], label=f"{label}", s=10)
            else:
                ax.scatter(X_umap[idx, 0], X_umap[idx, 1], c=[label_color_map[label]], label=f"{label}", s=10)

        legend_elements = [Patch(facecolor=label_color_map[label], label=f'{label}') for label in unique_labels]
        ax.legend(handles=legend_elements)
    else:
        proj = X_pca if method == 'PCA' else X_umap
        ax.scatter(proj[:, 0], proj[:, 1], s=10)

    ax.set_title(title if title else f"{method} Projection")
    ax.axis("off")

    if internal_fig:
        plt.tight_layout()
        plt.show()

def extract_anomalies_df(context, merged_context):
    """
    Extracts and combines anomaly intervals from context and merged_context into one DataFrame.

    Each anomaly is expected to have [start, end, severity].
    The resulting DataFrame includes a 'start', 'end', 'severity', and 'id' column.

    Returns:
    - pd.DataFrame
    """
    def convert(array, source_id):
        if array is None or len(array) == 0:
            return pd.DataFrame(columns=['start', 'end', 'severity', 'id'])
        array = np.array(array)
        df = pd.DataFrame(array, columns=['start', 'end', 'severity'])
        df['start'] = pd.to_datetime(df['start'], unit='s')
        df['end'] = pd.to_datetime(df['end'], unit='s')
        df['id'] = source_id
        return df

    dfs = [
        convert(context.get('anomalies', []), 'context_anomalies'),
        convert(merged_context.get('anomalies', []), 'merged_anomalies'),
        convert(merged_context.get('unified_AD', []), 'unified_AD'),
        convert(merged_context.get('unified_AD_FD', []), 'unified_AD_FD')
    ]

    return pd.concat(dfs, ignore_index=True)

def create_labeled_df(X_index, x_lat, anomaly_df, is_unix=False):
    """
    Create a DataFrame with timestamps, latent vector components, and anomaly labels.

    Parameters:
    - X_index: array of shape (n_samples,), can be UNIX timestamps or datetime
    - x_lat: array of shape (n_samples, n_latent_dims)
    - anomaly_df: DataFrame with 'start' and 'end' columns for anomaly intervals
    - is_unix: bool, if True converts X_index from UNIX timestamp to datetime

    Returns:
    - DataFrame with 'timestamp', 'label', and x_0 to x_n columns
    """
    # Step 1: Convert timestamps if needed
    if is_unix:
        timestamps = pd.to_datetime(X_index, unit='s')
    else:
        timestamps = pd.to_datetime(X_index)  # ensures consistent datetime dtype

    # Step 2: Create column names for latent dimensions
    latent_dim_names = [f'x_{i}' for i in range(x_lat.shape[1])]

    # Step 3: Create DataFrame
    df = pd.DataFrame(x_lat, columns=latent_dim_names)
    df['timestamp'] = timestamps

    # Step 4: Add label column
    df['label'] = 'Normal'
    for _, row in anomaly_df.iterrows():
        start = row['start']
        end = row['end']
        df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end), 'label'] = 'Leakage'

    # Step 5: Reorder columns
    df = df[['timestamp', 'label'] + latent_dim_names]

    return df



# TODO: IMPROVE CLIENTS LABELS
def plot_all_clients_side_by_side(model, priv_train_loaders, metric='acc'):
    metric_names = ['acc', 'prec', 'rec', 'f1']
    metric_idx = metric_names.index(metric)
    print(metric_idx)
    n_clients = len(model.nets_list)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left Plot: Fit History (Loss) ---
    for i in range(n_clients):
        fit_history = model.nets_list[i].fit_history
        axes[0].plot(fit_history, label=f'Client {i}')
    axes[0].set_title('Fit History (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()

    # --- Right Plot: Selected Metric ---
    for i in range(n_clients):
        metrics = priv_train_loaders[i]['metrics']
        metric_values = [m[metric_idx] for m in metrics]
        axes[1].plot(metric_values, label=f'Client {i}')
    axes[1].set_title(f'{metric.upper()} per Client')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel(metric.upper())
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def plot_data_leak(
        df1,
        df2=None,
        labels1=None,
        labels2=None,
        convert_timestamp=False,
        ax=None,
        ylim1=None,
        ylim2=None,
        title='Overlayed Time Series Data'
):
    """
    Plots df1 on primary y-axis and optionally df2 on a secondary y-axis using datetime index or UNIX timestamp.
    Accepts external axis (ax) for subplot usage.
    """
    internal_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        internal_fig = True
    else:
        fig = ax.figure

    # # Convert index to UNIX timestamp if requested
    # time_index_1 = df1.index.astype('int64') // 10**9 if convert_timestamp else df1.index
    # time_index_2 = df2.index.astype('int64') // 10**9 if (df2 is not None and convert_timestamp) else (df2.index if df2 is not None else None)

    # Convert index to UNIX timestamp if requested
    if convert_timestamp:
        time_index_1 = df1.index.astype('int64') // 10 ** 9
        if df2 is not None:
            time_index_2 = df2.index.astype('int64') // 10 ** 9
    else:
        time_index_1 = df1.index
        if df2 is not None:
            time_index_2 = df2.index

    # Plot df1
    for col in df1.columns:
        label = f"{labels1}_{col}" if labels1 else col
        ax.plot(time_index_1, df1[col], lw=4, label=label)

    if ylim1 is not None:
        ax.set_ylim(ylim1)

    ax.set_xlabel('Timestamp (s since epoch)' if convert_timestamp else 'Timestamp')
    ax.set_ylabel('Pressure node')

    # Plot df2 on secondary axis
    if df2 is not None:
        ax2 = ax.twinx()
        for col in df2.columns:
            label = f"{labels2}_{col}" if labels2 else col
            ax2.plot(time_index_2, df2[col], label=label, color='black', linestyle='-', alpha=0.35)
        if ylim2 is not None:
            ax2.set_ylim(ylim2)
        ax2.set_ylabel('Flow pipe')

        # Combine legends
        lines1, labels1_ = ax.get_legend_handles_labels()
        lines2, labels2_ = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1_ + labels2_, loc='upper right')
    else:
        ax.legend(loc='upper right')

    ax.grid(True)
    ax.set_title(title)

    if internal_fig:
        plt.tight_layout()
        plt.show()


def plot_data_leak_altair(
        df1,
        df2=None,
        labels1=None,
        labels2=None,
        convert_timestamp=False,
        ylim1=None,
        ylim2=None
):
    """
    Creates an Altair layered time series plot for df1 and optionally df2.
    df1 on primary y-axis, df2 on simulated secondary y-axis.

    Parameters:
    - df1: First DataFrame (required)
    - df2: Second DataFrame (optional)
    - labels1: Label prefix for df1 columns (optional)
    - labels2: Label prefix for df2 columns (optional)
    - convert_timestamp: If True, converts datetime index to UNIX timestamp
    - ylim1: Tuple to set y-limits for df1
    - ylim2: Tuple to set y-limits for df2
    """
    # Convert index to column
    df1 = df1.copy()
    df1['timestamp'] = df1.index
    if convert_timestamp:
        df1['timestamp'] = df1['timestamp'].astype('int64') // 10 ** 9

    df1_melted = df1.melt(id_vars='timestamp', var_name='Variable', value_name='Value')
    if labels1:
        df1_melted['Variable'] = labels1 + '_' + df1_melted['Variable']

    base = alt.Chart(df1_melted).mark_line().encode(
        x=alt.X('timestamp:T' if not convert_timestamp else 'timestamp:Q', title='Timestamp'),
        y=alt.Y('Value:Q', title='Primary Values', scale=alt.Scale(domain=ylim1) if ylim1 else alt.Undefined),
        color='Variable:N'
    )

    if df2 is not None:
        df2 = df2.copy()
        df2['timestamp'] = df2.index
        if convert_timestamp:
            df2['timestamp'] = df2['timestamp'].astype('int64') // 10 ** 9

        df2_melted = df2.melt(id_vars='timestamp', var_name='Variable', value_name='Value')
        if labels2:
            df2_melted['Variable'] = labels2 + '_' + df2_melted['Variable']

        secondary = alt.Chart(df2_melted).mark_line(strokeDash=[4, 2], color='black', opacity=0.8).encode(
            x=alt.X('timestamp:T' if not convert_timestamp else 'timestamp:Q'),
            y=alt.Y('Value:Q', title='Secondary Values', scale=alt.Scale(domain=ylim2) if ylim2 else alt.Undefined),
            detail='Variable:N'
        )

        chart = alt.layer(base, secondary).resolve_scale(y='independent').properties(
            width=600,
            height=200,
            title='Overlayed Time Series Data'
        )
    else:
        chart = base.properties(
            width=600,
            height=200,
            title='Overlayed Time Series Data'
        )

    return chart