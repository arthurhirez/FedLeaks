import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import pywt

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine, cityblock
from scipy.signal import correlate
from scipy.fft import fft


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import wasserstein_distance, energy_distance
from scipy.linalg import subspace_angles
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import entropy

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from itertools import combinations

def proto_analysis(data_latent, normalize=True):
    features = [col for col in data_latent.columns if col.startswith('x_')]
    results = []

    # 1) Within each epoch-label group: pairwise cosine and manhattan distances
    for (epoch, label), group in data_latent.groupby(['epoch', 'label']):
        clients = group['client_id'].unique()
        feat_matrix = {c: group[group['client_id'] == c][features].values.flatten() for c in clients}

        for c1, c2 in combinations(clients, 2):
            v1 = feat_matrix[c1]
            v2 = feat_matrix[c2]

            res = {
                'epoch': epoch,
                'label': label,
                'client_1': c1,
                'client_2': c2,
                'cosine': cosine_distance_proto(v1, v2),
                'manhattan': manhattan_distance_proto(v1, v2),
            }
            results.append(res)
    df_distances = pd.DataFrame(results)
    
    results_signal = []
    
    # Group by epoch
    for epoch, group in data_latent.groupby('epoch'):
        clients = group['client_id'].unique()
        
        # For each client, get matrix: labels (months) × features, sorted by label (month)
        client_series = {
            c: group[group['client_id'] == c].sort_values('label')[features].values
            for c in clients
        }
        
        for c1, c2 in combinations(clients, 2):
            p1 = client_series[c1]
            p2 = client_series[c2]
            
            # Make sure same length in labels (months)
            min_len = min(p1.shape[0], p2.shape[0])
            p1 = p1[:min_len]
            p2 = p2[:min_len]
            
            res = {
                'epoch': epoch,
                'client_1': c1,
                'client_2': c2,
                'wavelet': wavelet_distance(p1, p2),
                'dft': dft_similarity(p1, p2),
                'autocorr': autocorr_similarity(p1, p2),
            }
            results_signal.append(res)
    df_signal = pd.DataFrame(results_signal)

    df_final = df_distances.merge(
        df_signal,
        on=['epoch', 'client_1', 'client_2'],
        how='left',
        suffixes=('', '_signal')
    )

    # Normalize if asked (only for numeric cols)
    if normalize:
        scaler = MinMaxScaler()
        cols_to_scale = ['cosine', 'manhattan', 'wavelet', 'dft', 'autocorr']
        df_final[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])

    return df_final


def distributions_analysis(data_distribution, target_client="District_E", epoch = 0, normalize=True):
    """
    Perform temporal distribution comparison of latent representations between a target client 
    and all other clients across time (label/month), using a variety of statistical, geometric, 
    and information-theoretic metrics.

    Parameters:
    -----------
    data_distribution : pd.DataFrame
        DataFrame containing latent space representations with at least:
        - 'client_id' column for client identifier.
        - 'label' column for time/month index.
        - columns prefixed with 'x_' as latent features.
    
    target_client : str, default="District_E"
        The client to compare against all other clients across months.
    
    normalize : bool, default=True
        Whether to apply min-max normalization to the resulting metric columns.

    Returns:
    --------
    df_results : pd.DataFrame
        A DataFrame with pairwise metrics between the target client and other clients for each month.
        Columns include:
            - 'month': Time label
            - 'target_client', 'other_client': Identifiers
            - 'MMD': Maximum Mean Discrepancy
            - 'Wasserstein': Wasserstein (Earth Mover's) Distance
            - 'Energy': Energy Distance
            - 'SubspaceAlignment': Subspace alignment score
            - 'DTW': Dynamic Time Warping over latent trajectories
            - 'MutualInfo': Mutual information between latent distributions
            - 'KL': Kullback-Leibler divergence
            - 'JSD': Jensen-Shannon divergence
    """
    df = data_distribution.copy()
    features = [col for col in df.columns if col.startswith("x_")]
    clients = df["client_id"].unique()
    months = sorted(df["label"].unique())

    results = []

    for month in tqdm(months):
        df_month = df[df.label == month]
        X_target = df_month[df_month.client_id == target_client][features].values


        for other_client in clients:
            if other_client == target_client:
                continue

            X_other = df_month[df_month.client_id == other_client][features].values
            if X_target.size == 0 or X_other.size == 0:
                continue

            # 1. Distribution metrics
            mmd_val, w_dist, e_dist = compare_clients_distribution(X_target, X_other)

            # 2. Subspace alignment
            sa = subspace_alignment(X_target, X_other)

            # 3. DTW over historical latent trajectory
            traj_target = dtw_client_trajectory(df[df.label <= month], target_client, features)
            traj_other = dtw_client_trajectory(df[df.label <= month], other_client, features)
            min_len = min(len(traj_target), len(traj_other))
            dtw_val, _ = fastdtw(traj_target[:min_len], traj_other[:min_len], dist=cosine)

            # 4. Mutual Information
            mi = compute_mi(df, target_client, other_client, month, features)

            # 5. KL & JSD
            kl = compute_kl_divergence(X_target, X_other)
            jsd = compute_js_divergence(X_target, X_other)


            results.append({
                "epoch" : epoch,
                "label": month,
                "target_client": target_client,
                "other_client": other_client,
                "MMD": mmd_val,
                "Wasserstein": w_dist,
                "Energy": e_dist,
                "SubspaceAlignment": sa,
                "DTW": dtw_val,
                "MutualInfo": mi,
                "KL": kl,
                "JSD": jsd,
            })

    df_results = pd.DataFrame(results)

    # Optional normalization
    if normalize:
        metric_cols = [
            "MMD", "Wasserstein", "Energy", "SubspaceAlignment",
            "DTW", "MutualInfo", "KL", "JSD",
        ]
        scaler = MinMaxScaler()
        df_results[metric_cols] = scaler.fit_transform(df_results[metric_cols])

    return df_results



def cosine_distance_proto(v1, v2):
    return cosine(v1, v2)

def manhattan_distance_proto(v1, v2):
    return cityblock(v1, v2)


def wavelet_distance(protos_a, protos_b, wavelet='db1'):
    dist = 0
    for i in range(protos_a.shape[1]):
        coeffs_a = pywt.dwt(protos_a[:, i], wavelet)
        coeffs_b = pywt.dwt(protos_b[:, i], wavelet)
        dist += np.linalg.norm(np.array(coeffs_a[0]) - np.array(coeffs_b[0]))  # Compare approx coefficients
    return dist

def dft_similarity(protos_a, protos_b):
    fft_a = np.fft.fft(protos_a, axis=0)
    fft_b = np.fft.fft(protos_b, axis=0)
    return np.linalg.norm(np.abs(fft_a - fft_b))

import numpy as np

def autocorr_similarity(protos_a, protos_b, lag=1):
    def autocorr(x, lag):
        n = len(x)
        if n <= lag + 1:
            # Not enough data to compute correlation for this lag
            return np.nan
        x1 = x[:-lag]
        x2 = x[lag:]
        # Check if variance is zero, which causes correlation to be NaN
        if np.std(x1) == 0 or np.std(x2) == 0:
            return np.nan
        return np.corrcoef(x1, x2)[0, 1]
    
    acc = 0
    valid_counts = 0
    for i in range(protos_a.shape[1]):
        a_corr = autocorr(protos_a[:, i], lag)
        b_corr = autocorr(protos_b[:, i], lag)
        # Only accumulate if both correlations are valid numbers
        if not np.isnan(a_corr) and not np.isnan(b_corr):
            acc += abs(a_corr - b_corr)
            valid_counts += 1
    
    # If no valid correlations were computed, return NaN or 0 as fallback
    if valid_counts == 0:
        return np.nan
    else:
        return acc / valid_counts  # average difference across features



def compute_kl_divergence(Xa, Xb, bins=30):
    kl_scores = []
    for i in range(Xa.shape[1]):
        hist_a, _ = np.histogram(Xa[:, i], bins=bins, density=True)
        hist_b, _ = np.histogram(Xb[:, i], bins=bins, density=True)
        hist_a += 1e-8  # smooth to avoid log(0)
        hist_b += 1e-8
        kl = entropy(hist_a, hist_b)
        kl_scores.append(kl)
    return np.mean(kl_scores)

def compute_js_divergence(Xa, Xb, bins=30):
    js_scores = []
    for i in range(Xa.shape[1]):
        hist_a, _ = np.histogram(Xa[:, i], bins=bins, density=True)
        hist_b, _ = np.histogram(Xb[:, i], bins=bins, density=True)
        hist_a += 1e-8
        hist_b += 1e-8
        m = 0.5 * (hist_a + hist_b)
        js = 0.5 * entropy(hist_a, m) + 0.5 * entropy(hist_b, m)
        js_scores.append(js)
    return np.mean(js_scores)


# --- Drift Metrics Functions

def compute_mmd(X, Y, gamma=1.0):
    K = rbf_kernel(X, X, gamma=gamma)
    L = rbf_kernel(Y, Y, gamma=gamma)
    KL = rbf_kernel(X, Y, gamma=gamma)
    return K.mean() + L.mean() - 2 * KL.mean()

def subspace_alignment(X1, X2, n_components=10):
    pca1 = PCA(n_components=n_components).fit(X1)
    pca2 = PCA(n_components=n_components).fit(X2)
    angles = subspace_angles(pca1.components_.T, pca2.components_.T)
    return np.sum(np.cos(angles))

def dtw_client_trajectory(df, client, features):
    grouped = df[df.client_id == client].sort_values(by="label")
    monthly = grouped.groupby("label")[features].mean().values
    return monthly

def run_dbscan(df_sub, features):
    db = DBSCAN(eps=0.5, min_samples=5).fit(df_sub[features])
    return db.labels_

def spectral_cluster_latent(df_sub, n_components=2, features = []):
    sim = cosine_similarity(df_sub[features])
    embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
    X_trans = embedding.fit_transform(sim)
    return X_trans

def compute_mi(df, client_a, client_b, month, features):
    Xa = df[(df.label == month) & (df.client_id == client_a)][features].values
    Xb = df[(df.label == month) & (df.client_id == client_b)][features].values
    if len(Xa) == 0 or len(Xb) == 0:
        return np.nan
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    Xa_d = est.fit_transform(Xa)
    Xb_d = est.fit_transform(Xb)
    mi_scores = [mutual_info_classif(Xa_d, Xb_d[:, i], discrete_features=True).mean() for i in range(Xb_d.shape[1])]
    return np.mean(mi_scores)

def compare_clients_distribution(Xa, Xb):
    mmd_val = compute_mmd(Xa, Xb, gamma=0.5)
    w_dist = np.mean([wasserstein_distance(Xa[:, i], Xb[:, i]) for i in range(Xa.shape[1])])
    e_dist = energy_distance(Xa.flatten(), Xb.flatten())
    return mmd_val, w_dist, e_dist


