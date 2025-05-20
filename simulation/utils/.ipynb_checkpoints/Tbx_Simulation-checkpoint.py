import copy
import random
from collections import defaultdict

import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.configs import DISTRIBUTION_PATTERNS

alt.data_transformers.enable("vegafusion")


def convert_flow_rate(value, from_unit):
    """
    Convert a flow rate to all other units: GPM, L/s, m³/day, m³/month, m³/hour (CMH).

    Parameters:
        value (float): The flow rate value.
        from_unit (str): The unit of the input value. One of: 'gpm', 'l/s', 'm3/day', 'm3/month', 'm3/h'.

    Returns:
        dict: Converted values in all units.
    """
    from_unit = from_unit.lower()

    # Conversion factors to liters per second (base unit)
    to_lps = {
        'gpm': 0.0630902,  # 1 GPM = 0.0630902 L/s
        'l/s': 1.0,
        'm3/day': 1000 / 86400,  # 1 m³/day = 1000 / 86400 L/s
        'm3/month': 1000 / (30 * 86400),  # 1 m³/month = 1000 / (30 * 86400) L/s
        'm3/h': 1000 / 3600,  # 1 m³/h = 1000 / 3600 L/s
    }

    if from_unit not in to_lps:
        raise ValueError(f"Unsupported unit: {from_unit}")

    # Convert input to L/s
    lps = value * to_lps[from_unit]

    # Now convert from L/s to all other units
    conversions = {
        'gpm': lps / 0.0630902,
        'l/s': lps,
        'm3/h': lps * 3600 / 1000,
        'm3/day': lps * 86400 / 1000,
        'm3/month': lps * 86400 * 30 / 1000,

    }

    return conversions


def weighted_blend(previous_pattern, current_pattern, weight_prev=2, weight_curr=1):
    """
    Blend two patterns using the provided weights.
    
    previous_pattern: numpy array from previous epoch (e.g., aux_consumption[0][-2])
    current_pattern: numpy array from current epoch (e.g., aux_consumption[1][0])
    weight_prev: weight for the previous pattern
    weight_curr: weight for the current pattern
    """
    total_weight = weight_prev + weight_curr
    blended = (previous_pattern * weight_prev + current_pattern * weight_curr) / total_weight
    return blended


def compute_total_consumption(multipliers, base_demand):
    """
    Computes the total consumption over the period based on the multipliers and base demand.

    Parameters:
    - multipliers: An array or list of consumption multipliers for each time interval (e.g., 48 for 30-minute intervals).
    - base_demand: The base demand per interval (e.g., consumption in m³ for each time period).
    - n_intervals: The number of time intervals (default is 48, for 30-minute intervals in a day).

    Returns:
    - total_consumption: The total consumption over the period.
    """
    # Ensure that multipliers is a numpy array for easier element-wise operations
    multipliers = np.array(multipliers)

    # Calculate the consumption for each interval
    consumption_per_interval = multipliers * base_demand

    # Total consumption is the sum of all intervals
    total_consumption = np.sum(consumption_per_interval)

    return total_consumption


def generate_consumption_patterns(data_consumption,
                                  original_state,
                                  consumption_patterns,
                                  id_time=0,
                                  epochs=None,
                                  days=None,
                                  n_intervals=24,
                                  unit='l/s',
                                  morning_peak=1.0,
                                  afternoon_peak=0.75,
                                  evening_peak=1.25,
                                  night_consumption=0.25,
                                  variation_strength=0.02):
    consumption_patterns[id_time] = {}

    for node, base_consumption, density in zip(data_consumption['node_id'], data_consumption['consumption'],
                                               data_consumption['density']):
        consumption_patterns[id_time][node] = {}
        consumption_record = []

        for epoch in range(epochs):
            while True:
                var_demand = np.random.normal(loc=0, scale=0.10)
                if var_demand < -0.50:
                    continue
                if abs(var_demand) < 0.075:
                    continue
                break
            consumption_observed = base_consumption * (1 + var_demand)
            consumption_record.append(consumption_observed)

        aux_consumption = []

        for consumption in consumption_record:
            flow = convert_flow_rate(consumption, 'm3/month')
            base_demand = flow[unit]

            config = DISTRIBUTION_PATTERNS.get(density)
            base_peaks, variation_strength = config[:-1], config[-1]

            if density == 'low':
                noisy_peaks = shuffle_and_perturb_array(base_values=base_peaks,
                                                        scale=0.125,
                                                        min_val=0.2)
            elif density == 'medium':
                if random.random() < 0.80:
                    noisy_peaks = add_noise_with_lower_bound(base_values=base_peaks,
                                                             scale=0.075,
                                                             min_val=0.5)
                else:
                    noisy_peaks = shuffle_and_perturb_array(base_values=base_peaks,
                                                            scale=0.05,
                                                            min_val=0.5)
            else:
                noisy_peaks = base_peaks

            morning_peak_var, afternoon_peak_var, evening_peak_var, night_consumption_var = noisy_peaks

            weekly_patterns = generate_weekly_consumption_patterns(base_demand=base_demand,
                                                                   n_intervals=n_intervals,
                                                                   n_days=days,
                                                                   morning_peak=morning_peak_var,
                                                                   afternoon_peak=afternoon_peak_var,
                                                                   evening_peak=evening_peak_var,
                                                                   night_consumption=night_consumption_var,
                                                                   variation_strength=variation_strength)

            weekly_patterns = np.array(weekly_patterns)
            original_consumption = original_state[original_state['node_id'] == node]['consumption'].tolist()[0]
            weekly_patterns = weekly_patterns * (consumption / original_consumption)
            # weekly_patterns = weekly_patterns.tolist()  # Convert back to list of lists

            aux_consumption.append(weekly_patterns)

        wght_prev = [i for i in range(days, 0, -1)]
        wght_curr = [i for i in range(1, days + 1)]

        smooth_epoch_transitions(data_consumption=aux_consumption,
                                 transition_days=days,
                                 weights_prev=wght_prev,
                                 weights_curr=wght_curr,
                                 plot=False
                                 )

        consumption_patterns[id_time][node]['patterns'] = aux_consumption

        flow = convert_flow_rate(original_consumption, 'm3/month')
        consumption_patterns[id_time][node]['base_demand'] = flow[unit]  # base_demand
    # return consumption_patterns


def generate_weekly_consumption_patterns(base_demand, n_intervals=48, n_days=7,
                                         morning_peak=1.5, afternoon_peak=1.4, evening_peak=1.3,
                                         night_consumption=0.5, variation_strength=0.05):
    """
    Generates weekly consumption patterns (n days) using the daily consumption function, based on monthly consumption and base demand.

    Parameters:
    - base_demand: The base daily demand in cubic meters (e.g., base consumption at a given time of day).
    - n_intervals: Number of intervals in the day (default 48, 30-minute intervals).
    - n_days: Number of days to simulate (default is 7 days).
    - morning_peak: Multiplier for the morning peak consumption.
    - evening_peak: Multiplier for the evening peak consumption.
    - night_consumption: Multiplier for the night-time consumption.
    - variation_strength: Strength of random variation between days.

    Returns:
    - A list of numpy arrays, where each array contains the consumption multipliers for each 30-minute interval of a day.
    """

    # The daily consumption should be at least the base demand, adjust if necessary
    # daily_consumption = max(daily_consumption, base_demand)

    weekly_patterns = []

    for _ in range(n_days):
        daily_pattern = generate_daily_consumption_pattern(daily_consumption=base_demand,
                                                           n_intervals=n_intervals, morning_peak=morning_peak,
                                                           afternoon_peak=afternoon_peak, evening_peak=evening_peak,
                                                           night_consumption=night_consumption,
                                                           variation_strength=variation_strength)
        weekly_patterns.append(daily_pattern)

    return weekly_patterns


def generate_daily_consumption_pattern(daily_consumption=1.0, n_intervals=48,
                                       morning_peak=1.5, evening_peak=1.3, afternoon_peak=1.4,
                                       night_consumption=0.5, variation_strength=0.05):
    """
    Generates a daily consumption pattern with a bimodal distribution (higher consumption in the morning and evening),
    and scales the pattern so that the total consumption matches the daily_consumption.
    The afternoon consumption is set slightly higher than the night-time consumption.

    Parameters:
    - base_multiplier: The base multiplier for consumption (default is 1.0).
    - daily_consumption: The target total daily consumption (in cubic meters).
    - n_intervals: Number of intervals in the day (default 48, 30-minute intervals).
    - morning_peak: Multiplier for the morning peak consumption.
    - evening_peak: Multiplier for the evening peak consumption.
    - afternoon_peak: Multiplier for the afternoon peak consumption (increased compared to night time).
    - night_consumption: Multiplier for the night-time consumption.
    - variation_strength: Strength of random variation between days.

    Returns:
    - A numpy array with consumption multipliers for each 30-minute interval.
    """
    # Time intervals for a day (0 to 24 hours in 30-minute intervals)
    time = np.linspace(0, 24, n_intervals)

    # Create a bimodal pattern (morning, afternoon, and evening peaks)
    morning = np.exp(-0.5 * ((time - 7) / 2) ** 2)  # Gaussian peak in the morning (around 7 AM)
    afternoon = np.exp(-0.5 * ((time - 14) / 2) ** 2)  # Gaussian peak in the afternoon (around 2 PM)
    evening = np.exp(-0.5 * ((time - 19) / 2) ** 2)  # Gaussian peak in the evening (around 7 PM)

    # Combine the peaks with night-time consumption
    pattern = (morning_peak * morning +
               afternoon_peak * afternoon +
               evening_peak * evening +
               night_consumption * (1 - morning - afternoon - evening))

    # Scale the pattern by the base multiplier and add random variation
    random_variation = np.random.normal(1, variation_strength, n_intervals)  # Small random variation for each day
    # daily_pattern = base_multiplier * pattern * random_variation + daily_consumption
    daily_pattern = pattern * random_variation + daily_consumption

    # Normalize the pattern to ensure the total daily consumption matches the target daily consumption
    daily_pattern_total = np.sum(daily_pattern)  # Calculate the sum of the pattern

    # daily_pattern_scaled = daily_pattern / daily_pattern_total  # Scale the pattern

    # Pre-checks for NaN, Inf, and zero total
    if np.isnan(daily_pattern_total).any():
        print("Warning: NaN detected in daily_pattern_total.")
    elif np.isinf(daily_pattern_total).any():
        print("Warning: Inf detected in daily_pattern_total.")
    elif np.any(daily_pattern_total == 0):
        print("Warning: Zero detected in daily_pattern_total.")
        print(f"daily_pattern_total: {daily_pattern_total}")
        print(f"daily_pattern: {daily_pattern}")
    else:
        daily_pattern_scaled = daily_pattern / daily_pattern_total

    return daily_pattern_scaled


def add_noise_with_lower_bound(base_values, scale=0.125, min_val=0.01):
    while True:
        noise = np.random.normal(loc=0, scale=scale, size=base_values.shape)
        noisy_values = base_values + noise
        if np.all(noisy_values > min_val):  # Only ensure values don't go too low
            return noisy_values


def shuffle_and_perturb_array(base_values, scale=0.125, min_val=0.01):
    """
    Shuffle a numpy array and add noise with a lower bound.

    Parameters:
    -----------
    values : np.ndarray
        1D array of values to shuffle and perturb.
    noise_scale : float
        Std deviation of Gaussian noise.
    min_val : float
        Minimum allowed value after noise.

    Returns:
    --------
    np.ndarray
        Shuffled and noise-perturbed array.
    """
    values = np.array(base_values).copy()
    np.random.shuffle(values)
    noisy_values = add_noise_with_lower_bound(base_values=values, scale=scale, min_val=min_val)
    return noisy_values


def simulate_urban_growth(G, seed_node, n_steps=5, growth_chance=0.5, max_neighbors_per_step=3):
    """
    Simulates urban densification starting from seed_node over n_steps.
    Uses a probabilistic coin toss to determine if neighbors are upgraded,
    with a cap on the maximum number of neighbors that can be upgraded.
    Randomly selects one of the last upgraded nodes to expand in each step.

    Parameters:
    - G: networkx undirected graph
    - seed_node: node to start growth from
    - n_steps: number of diffusion steps
    - growth_chance: probability [0.0–1.0] that each neighbor is upgraded
    - max_neighbors_per_step: maximum number of neighbors that can be upgraded in each step

    Returns:
    - history: list of sets, showing upgraded nodes at each step
    """
    upgraded_nodes = set([seed_node])
    history = [set(upgraded_nodes)]

    last_upgraded = set([seed_node])  # frontier for next step

    for _ in range(n_steps):
        new_upgrades = set()

        # Randomly select one of the last upgraded nodes
        node = random.choice(list(last_upgraded))

        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)  # randomize neighbors to simulate variation
        count = 0

        for neighbor in neighbors:
            if neighbor not in upgraded_nodes and random.random() < growth_chance:
                new_upgrades.add(neighbor)
                count += 1
                if count >= max_neighbors_per_step:
                    break  # stop if the maximum number of upgrades has been reached

        upgraded_nodes.update(new_upgrades)
        history.append(set(upgraded_nodes))
        last_upgraded = new_upgrades  # update frontier

        # If no new upgrades were made, break early
        if not new_upgrades:
            break

    return history


def plot_growth_over_time(G, history, pos=None, step=None):
    """
    Plots the graph showing upgraded nodes at a specific step.
    If no step is given, it shows the final state.
    """
    if step is None or step >= len(history):
        step = len(history) - 1

    upgraded = history[step]

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    node_colors = ['red' if node in upgraded else 'lightgray' for node in G.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500)
    plt.title(f"Urban Densification at Step {step}")
    plt.show()


def concatenate_consumption_patterns(consumption_patterns):
    """
    Concatenates patterns across all scenarios for each node and computes the mean base demand.

    Returns:
        dict: {
            node_id: {
                'full_series': 1D np.array of concatenated time series,
                'mean_base_demand': float
            }
        }
    """
    pattern_data = defaultdict(list)
    base_demand_data = defaultdict(list)

    # Iterate in order of scenarios to preserve sequence
    for scenario in sorted(consumption_patterns.keys()):
        nodes = consumption_patterns[scenario]
        for node_id, node_data in nodes.items():
            if 'patterns' in node_data:
                pattern_segments = node_data['patterns']  # list of lists
                flat_pattern = np.concatenate(np.concatenate(pattern_segments))
                pattern_data[node_id].append(flat_pattern)

                if 'base_demand' in node_data:
                    base_demand_data[node_id].append(node_data['base_demand'])

    final = {}
    for node_id in pattern_data:
        full_series = np.concatenate(pattern_data[node_id])
        mean_base_demand = np.mean(base_demand_data[node_id]) if base_demand_data[node_id] else None
        final[node_id] = {
            'full_series': full_series,
            'mean_base_demand': mean_base_demand
        }

    return final


def plot_node_time_series(time_series_dict, node_ids, title_prefix=''):
    """
    Plots the full time series for given node IDs.

    Parameters:
    - time_series_dict (dict): Output from `concatenate_consumption_patterns`.
    - node_ids (list): List of node IDs to plot.
    - title_prefix (str): Optional title prefix.
    """
    for node_id in node_ids:
        if node_id not in time_series_dict:
            print(f"Node {node_id} not found.")
            continue

        series = time_series_dict[node_id]['full_series']
        plt.figure(figsize=(12, 3))
        plt.plot(series, label=f'Node {node_id}')
        plt.title(f'{title_prefix} Node {node_id} - Full Pattern')
        plt.xlabel('Time Step')
        plt.ylabel('Consumption')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_smoothing_comparison(before, after, transition_days, node_id=None):
    """
    Plots a comparison of the transition days before and after smoothing.
    """
    for epoch in range(1, len(before)):
        fig, axs = plt.subplots(transition_days, 1, figsize=(10, 3 * transition_days), sharex=True)

        if transition_days == 1:
            axs = [axs]

        for i in range(transition_days):
            idx_prev = -transition_days + i
            idx_curr = i

            axs[i].plot(before[epoch - 1][idx_prev], '--', label='Before (prev)', color='red')
            axs[i].plot(before[epoch][idx_curr], '--', label='Before (curr)', color='orange')
            axs[i].plot(after[epoch - 1][idx_prev], '-', label='After (prev)', color='blue')
            axs[i].plot(after[epoch][idx_curr], '-', label='After (curr)', color='green')

            axs[i].set_title(f'Epoch {epoch - 1}→{epoch}, Day {i + 1}/{transition_days}')
            axs[i].legend()
            axs[i].grid(True)

        suptitle = f"Smoothing Transitions for Node {node_id}" if node_id else "Epoch Transition Smoothing"
        plt.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def smooth_drift_profile(ts, smooth_span=1500, method='log', figsize=None):
    """
    Smooths a sudden jump in a time series using a logarithmic growth curve
    while preserving the local profile (shape) of the data.

    Parameters:
    ----------
    ts : array-like
        The original time series data.
    smooth_span : int, optional
        Number of points after the detected jump to smooth. Default is 10.
    method : str, optional
        Growth method for smoothing ('log' supported; future: 'exp', 'sigmoid').

    Returns:
    -------
    np.ndarray
        Smoothed version of the time series.
    """
    ts_original = np.array(ts).copy()
    ts = np.array(ts).copy()

    # Step 1: Detect jump
    diff = np.diff(ts)
    change_point = np.argmax(diff)

    # Step 2: Define smoothing window
    start = change_point
    end = min(change_point + smooth_span, len(ts) - 1)

    original_slice = ts[start:end + 1]
    original_mean = np.mean(original_slice)

    # Step 3: Generate growth curve
    x = np.linspace(1, (end - start + 1), end - start + 1)

    if method == 'log':
        curve = np.log(x)
    else:
        raise ValueError(f"Unsupported method: {method}")

    curve_normalized = curve / np.mean(curve)
    scaled_curve = curve_normalized * original_mean

    # Step 4: Preserve local shape
    profile = original_slice / original_mean
    smoothed_slice = scaled_curve * profile

    # Step 5: Replace in original series
    ts[start:end + 1] = smoothed_slice

    if figsize is not None:
        plt.figure(figsize=figsize)
        plt.plot(ts_original, label="Original")
        plt.plot(ts, label="Smoothed", linestyle='--')
        plt.axvline(np.argmax(np.diff(ts_original)), color='red', linestyle=':')
        plt.title("Smoothed Time Series")
        plt.legend()
        plt.grid(True)
        plt.show()

    return ts


def smooth_epoch_transitions(data_consumption, transition_days=2,
                             weights_prev=None, weights_curr=None,
                             plot=False, node_id=None):
    """
    Smooths transitions between epochs in consumption patterns.

    Parameters:
    - data_consumption (list): List of epochs, each containing a list of daily patterns.
    - transition_days (int): Number of days to blend at the end/start of each epoch.
    - weights_prev (list): Weights for previous epoch's last `transition_days` days.
    - weights_curr (list): Weights for current epoch's first `transition_days` days.
    - plot (bool): If True, plots the before/after of transitions.
    - node_id (str or None): Optional label for plotting.
    """

    n_epochs = len(data_consumption)

    # Set default weights if not provided
    if weights_prev is None:
        weights_prev = list(range(transition_days, 0, -1))
    if weights_curr is None:
        weights_curr = list(range(1, transition_days + 1))

    assert len(weights_prev) == transition_days, "weights_prev length mismatch"
    assert len(weights_curr) == transition_days, "weights_curr length mismatch"

    if plot:
        original = copy.deepcopy(data_consumption)

    for epoch in range(1, n_epochs):
        for i in range(transition_days):
            idx_prev = -transition_days + i
            idx_curr = i

            blended = weighted_blend(
                data_consumption[epoch - 1][idx_prev],
                data_consumption[epoch][idx_curr],
                weight_prev=weights_prev[i],
                weight_curr=weights_curr[i]
            )

            data_consumption[epoch - 1][idx_prev] = blended
            data_consumption[epoch][idx_curr] = blended

    if plot:
        plot_smoothing_comparison(original, data_consumption, transition_days, node_id)

    return data_consumption
