import glob
import os
import shutil
import subprocess

import pickle
from multiprocessing.managers import Namespace

import pandas as pd
import numpy as np
import wntr
from atmn import ScenarioCollection

from utils.Districts import load_network_and_districts, combine_districts
from utils.Tbx_Leaks import generate_random_leaks_for_all_pipes, get_target_sensors, create_config_xml
from utils.Tbx_Simulation import generate_consumption_patterns, concatenate_consumption_patterns
from utils.Tbx_Simulation import simulate_urban_growth, smooth_drift_profile, shuffle_and_perturb_array

from utils.configs import MAPPING_SENSORS


# -------- LOAD AND ASSIGN DOMAIN -------- #

def load_assign_network(args: Namespace, directory='networks/original/',
                        save_assignments=True, verbose=True):
    """
    Loads a specified network, assigns income and density levels to districts,
    simulates urban growth, modifies building assignments, and generates consumption patterns.

    Parameters:
        id_network (str): Identifier for the network to be loaded.
        args (Namespace): Argument namespace containing simulation parameters.
        directory (str): Path to the directory containing the original networks.
        save_assignments (bool): Whether to save the assignment data to a file.
        verbose (bool): If True, prints detailed information during execution.

    Returns:
        wn: The loaded water network object or network representation used for further simulation or analysis.
        consumption_patterns (dict): Dictionary containing simulated consumption patterns over time for each step.
        data_consumption (pd.DataFrame): Combined building assignment data from all districts, including updated attributes.

    """

    id_network = args.id_network
    id_exp = args.id_exp
    tgt_district = args.tgt_district

    # Load the network graph and corresponding district data
    wn, district_nodes = load_network_and_districts(id_network=id_network,
                                                    directory=directory)

    # Assign income and density attributes to each district using predefined mapping
    for i, district in enumerate(district_nodes.values()):
        district['income'] = args.income_density_mapping[i][0]
        district['density'] = args.income_density_mapping[i][1]

    # Create a visual chart of combined districts (optional visualization)
    chart = combine_districts(
        district_nodes=district_nodes,
        plot_size=(600, 400)
    )

    # Optionally save the initial state of district assignments
    if save_assignments:
        with open(f"networks/assignments/{id_network}_{id_exp}_{tgt_district}_initial.pkl", "wb") as f:
            pickle.dump(district_nodes, f)

    # Retrieve the target district graph for simulation
    G_tgt = district_nodes[tgt_district]['network'].to_graph().to_undirected()

    # Simulate urban growth until the target number of segments is reached
    while True:
        growth_tgt = simulate_urban_growth(G=G_tgt,
                                           seed_node=args.seed_node,
                                           n_steps=args.n_segments - 1,
                                           growth_chance=0.95,
                                           max_neighbors_per_step=4)
        if len(growth_tgt) == args.n_segments:
            break

    # TODO: MELHORAR ESSA CEHCAGEM!
    skip_set = set(MAPPING_SENSORS[args.id_network]['Skip'])
    for s in growth_tgt:
        s.difference_update(skip_set)

    # Initialize tracking variables for used and previous nodes
    used_nodes = set()
    previous_nodes = set()
    growth_tgt.insert(0, set())  # Insert dummy step 0 for consistent indexing
    district_nodes[tgt_district]['assignments'].growth_nodes = growth_tgt
    consumption_patterns = {}
    tgt_nodes = list(growth_tgt[-1])

    # Iterate through each simulation step
    for i, current_nodes in enumerate(growth_tgt):
        new_nodes = current_nodes - previous_nodes
        nodes_to_use = new_nodes - used_nodes

        used_nodes.update(nodes_to_use)
        previous_nodes = current_nodes

        # Print node changes if verbose
        if verbose:
            print(f"Step {i}: Changing nodes {sorted(nodes_to_use)}")
            old_state = district_nodes[tgt_district]['assignments'].data_buildings
            old_state = old_state[old_state['node_id'].isin(nodes_to_use)]

        # Edit node assignments: set income and density to 'high'
        for node in nodes_to_use:
            # TODO: include income/density change in args
            district_nodes[tgt_district]['assignments'].edit_node(
                node_id=node,
                income_level='medium',
                density_level='medium',
                seed=None
            )

        # Update district label in the assignments
        district_nodes[tgt_district]['assignments'].data_buildings['district'] = tgt_district

        # Print changes in node attributes, if verbose
        if verbose:
            new_state = district_nodes[tgt_district]['assignments'].data_buildings
            new_state = new_state[new_state['node_id'].isin(nodes_to_use)]

            for node_id in nodes_to_use:
                old_row = old_state[old_state['node_id'] == node_id].squeeze()
                new_row = new_state[new_state['node_id'] == node_id].squeeze()

                print(f"\nChanges for node_id = {node_id}")
                for col in old_state.columns:
                    old_val = old_row[col]
                    new_val = new_row[col]
                    if old_val != new_val:
                        print(f" - {col}: {old_val} ➝ {new_val}")

        # Merge data from all districts to prepare for consumption modeling
        nodes_districts = []
        aux_origin = []
        for name, district in district_nodes.items():
            aux_data = district['assignments'].data_buildings
            aux_data['district'] = name
            nodes_districts.append(aux_data)
            aux_origin.append(district['assignments'].original_data_buildings)

        data_consumption = pd.concat(nodes_districts)
        origin_data = pd.concat(aux_origin)

        base_peaks = np.array([args.morning_peak, args.afternoon_peak, args.evening_peak, args.night_consumption])
        noisy_peaks = shuffle_and_perturb_array(base_values=base_peaks,
                                                scale=0.125,
                                                min_val=0.25)

        morning_peak_var, afternoon_peak_var, evening_peak_var, night_consumption_var = noisy_peaks

        # Generate consumption pattern for current simulation step
        generate_consumption_patterns(
            data_consumption=data_consumption,
            original_state=origin_data,
            consumption_patterns=consumption_patterns,
            id_time=i,
            epochs=args.epochs_lenght,
            days=args.days_lenght,
            n_intervals=args.n_intervals,
            unit=args.unit,
            morning_peak=morning_peak_var,
            afternoon_peak=afternoon_peak_var,
            evening_peak=evening_peak_var,
            night_consumption=night_consumption_var,
            variation_strength=0.02
        )

    # TODO MELHORAR KKKKK
    district_nodes[tgt_district]['assignments'].data_buildings['drift_nodes'] = \
        district_nodes[tgt_district]['assignments'].data_buildings['node_id'].isin(tgt_nodes).astype(int)
    data_consumption['drift_nodes'] = data_consumption['node_id'].isin(tgt_nodes).astype(int)

    # Optionally save the final state of district assignments
    if save_assignments:
        with open(f"networks/assignments/{id_network}_{id_exp}_{tgt_district}_final.pkl", "wb") as f:
            pickle.dump(district_nodes, f)

    return wn, consumption_patterns, data_consumption


# -------- CREATE CONUSMPTION PATTERNS -------- #


def run_scenarios(water_network: object, consumption_patterns: dict, data_consumption: pd.DataFrame, args: Namespace):
    """
    Executes the full simulation pipeline for a water distribution network under various experimental conditions.

    This function prepares and processes a network simulation by:
    - Loading and assigning network-specific data
    - Generating and applying dynamic consumption patterns
    - Simulating leak scenarios based on specified parameters
    - Defining optimal sensor placements
    - Producing configuration files for downstream analysis

    Parameters:
        water_network: The water network model object used for simulation and analysis.
        consumption_patterns (dict): Time-series consumption data per simulation step, organized by node or zone.
        data_consumption (pd.DataFrame): Building-level consumption assignments, aggregated from all districts.
        args (Namespace): A configuration object.

    Returns:
        dict: A dictionary of auto-generated leak scenarios for use in analysis and validation.
    """

    # Extract key identifiers
    id_network = args.id_network
    id_exp = args.id_exp

    # Merge all individual node consumption patterns into a full time series
    full_time_series = concatenate_consumption_patterns(consumption_patterns)

    drift_nodes = data_consumption[data_consumption['drift_nodes'] == 1]['node_id'].tolist()

    for node in drift_nodes:
        full_time_series[node]['full_series'] = smooth_drift_profile(ts=full_time_series[node]['full_series'],
                                                                     smooth_span=2000,
                                                                     method='log',
                                                                     figsize=None)

    # Assign each node in the network a demand pattern based on the generated time series
    for node in data_consumption['node_id']:
        if 'Pump' in node:
            continue  # Skip pump nodes

        junction = water_network.get_node(node)

        # Ensure the node is a junction where demands are applied
        if isinstance(junction, wntr.network.elements.Junction):
            new_id = f'P{node}_{id_exp}'

            # Set the base demand and assign the corresponding time series pattern
            junction.demand_timeseries_list[0].base_value = full_time_series[node]['mean_base_demand']
            water_network.add_pattern(new_id, full_time_series[node]['full_series'])
            junction.demand_timeseries_list[0].pattern_name = new_id

    # Define output file paths
    path_exp = f"{id_network}_{id_exp}"
    new_path = f'networks/{path_exp}.inp'
    output_config = f"my_collection/{path_exp}_random_config.xml"
    output_directory = f"my_collection/{path_exp}_Random_Multiple"

    # If directory exists, delete it
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    # Create the main directory
    os.makedirs(output_directory)

    # Create subdirectories
    subdirs = ['measurements', 'sensors', 'sensorfaults', 'leaks']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_directory, subdir))

    # Save the modified network as an EPANET .inp file
    wntr.network.write_inpfile(water_network, new_path, version=2.2)

    # Calculate total number of simulation iterations
    total_iterations = args.n_intervals * args.n_segments * args.epochs_lenght * args.days_lenght

    # Automatically generate leak scenarios: 1 scenario with 5 leaks
    auto_leaks = generate_random_leaks_for_all_pipes(
        water_network=water_network,
        iterations=total_iterations,
        num_scenarios=1,
        leaks_per_scenario=5,
        severity=args.value_exp,
        min_duration=24,
        max_duration=300,
        save_dict_path=f'my_collection/{path_exp}.pkl'
    )

    pressure_list = [item for sublist in MAPPING_SENSORS[args.id_network]['Pressure'].values() for item in sublist]
    flow_list = [item for sublist in MAPPING_SENSORS[args.id_network]['Flow'].values() for item in sublist]
    fixed_cases_network = MAPPING_SENSORS[args.id_network]['Fixed']
    skip_cases_network = MAPPING_SENSORS[args.id_network]['Skip']

    # Determine sensor placement targets based on the leak scenarios
    tgt_nodes, tgt_links, scenarios_df = get_target_sensors(
        water_network=water_network,
        auto_leak_config=auto_leaks,
        fixed_cases=fixed_cases_network,
        skip_cases=skip_cases_network
    )

    # Merge target sensors with any pre-defined sensors
    combined_pressure = list(set(pressure_list).union(set(tgt_nodes)))
    combined_flow = list(set(flow_list).union(set(tgt_links)))

    # Create the simulation configuration XML
    create_config_xml(
        id_scenario=f'{path_exp}_Random_Multiple',
        output_file=output_config,
        network_path=f"../{new_path}",
        iterations=total_iterations,
        timestep=int((24 / args.n_intervals) * (60 ** 2)),  # timestep in seconds
        leak_scenarios=auto_leaks,
        pressure_sensors=combined_pressure,
        flow_sensors=combined_flow,
        demand_sensors=combined_pressure,
    )

    # Clean previous simulation results if the directory exists
    dir_path = f"my_collection/{path_exp}_Random_Multiple"
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        print(f"Directory exists: {dir_path} — deleting...")
        shutil.rmtree(dir_path)
        print("Directory deleted.")
    else:
        print(f"Directory does not exist: {dir_path}")

    # Trigger scenario generation using external tool
    try:
        subprocess.run(["atmn-generate", output_config, "-f"], check=True)
        print("Simulation executed successfully.")
    except subprocess.CalledProcessError as e:
        raise MemoryError(f'Simulation failed with return code {e.returncode}')

    return auto_leaks


# -------- COMPILE RESULTS -------- #

def compile_results(leaks_scenarios: dict, args: Namespace):
    """
    Compiles simulation results after running scenarios on a water distribution network.

    This function:
    - Runs simulation scenarios including random leaks using `run_scenarios()`
    - Loads and processes the generated scenario data from the ScenarioCollection
    - Converts leak scenario metadata into CSV format for analysis
    - Extracts and reshapes pressure, flow, and demand time-series data
    - Saves formatted data for each feature and scenario into output directories

    Parameters:
        leaks_scenarios (dict): A dictionary of auto-generated leak scenarios for use in analysis and validation.
        args (Namespace): A namespace object containing experiment and network settings.

    Returns:
        None. Writes processed leak scenario data and sensor time series to disk.
    """

    # Load generated scenarios using the ScenarioCollection class
    # aux_directory = 'simulation/'if args.callable else ''
    my_collection = ScenarioCollection(f'my_collection')
    my_scenario = my_collection.get_scenario(f"{args.id_network}_{args.id_exp}_Random_Multiple")

    scenarios_aux = []

    # Process each scenario from the auto_leak dictionary
    for scenario, cases in leaks_scenarios.items():
        df_aux = pd.DataFrame(cases)
        df_aux = df_aux.fillna(-1)
        df_aux['scenario'] = scenario

        # Convert time values from hours to seconds and ensure integer format
        for time in ['start', 'end', 'peak']:
            df_aux[time] = df_aux[time] * 3600
            df_aux[time] = df_aux[time].astype('Int64')

        scenarios_aux.append(df_aux)

    # Concatenate and save scenario metadata
    df_scenarios = pd.concat(scenarios_aux)

    save_path = f'../data_leaks/{args.id_network}/{args.id_exp}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_scenarios.to_csv(f'{save_path}/scenarios.csv',
                        index=False)

    # Loop through each relevant feature: Flow, Demand, Pressure
    for feature in ['Flow', 'Demand', 'Pressure']:
        leak_configs = my_scenario.list_configs()['LeakConfigs']

        aux_data = []  # Melted data for plotting
        aux_data_leak = []  # Raw leak time series

        var_id = 'pipeId' if feature.lower() == 'flow' else 'node'

        # Process each leak scenario for this feature
        for leak_scenario in leak_configs:
            data_scenario = my_scenario.get(leak_scenario, 'DefaultSensors', 'GT')
            df_feat = data_scenario[feature.lower()].copy()

            # Ensure clean numeric formatting
            for c in df_feat.columns:
                df_feat[c] = df_feat[c].astype(float).round(6)

            aux_leak = df_feat.copy()
            df_feat['time'] = pd.to_datetime(df_feat.index, unit='s')

            # Reshape for long-format plotting or time-series analysis
            df_feat_melted = df_feat.melt(
                ignore_index=False,
                var_name=var_id,
                value_name=feature
            ).reset_index()

            df_feat_melted = df_feat_melted.rename(columns={'time': 'Time'})
            df_feat_melted['scenario'] = leak_scenario

            aux_data.append(df_feat_melted)
            aux_leak['scenario'] = leak_scenario
            aux_data_leak.append(aux_leak)

        # Save full leak time series and melted DataFrame
        df_leak = pd.concat(aux_data_leak).reset_index()
        df_leak = df_leak.rename(columns={'time': 'timestamp'})

        df_leak.to_csv(f'{save_path}/{feature}.csv',
                       index=False)




#
# import pickle
# from multiprocessing.managers import Namespace
#
# import pandas as pd
# import numpy as np
# import wntr
# from atmn import ScenarioCollection
#
# for case in folders:
#     # Load generated scenarios using the ScenarioCollection class
#     my_collection = ScenarioCollection('my_collection')
#     my_scenario = my_collection.get_scenario(case)
#
#     save_path = f'../data_leaks/{args.id_network}/{case.replace("Graeme_", "").replace("_Random_Multiple", "")}'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     for feature in ['Flow', 'Demand', 'Pressure']:
#         leak_configs = my_scenario.list_configs()['LeakConfigs']
#
#         aux_data = []  # Melted data for plotting
#         aux_data_leak = []  # Raw leak time series
#
#         var_id = 'pipeId' if feature.lower() == 'flow' else 'node'
#
#         # Process each leak scenario for this feature
#         for leak_scenario in leak_configs:
#             data_scenario = my_scenario.get(leak_scenario, 'DefaultSensors', 'GT')
#             df_feat = data_scenario[feature.lower()].copy()
#
#             # Ensure clean numeric formatting
#             for c in df_feat.columns:
#                 df_feat[c] = df_feat[c].astype(float).round(6)
#
#             aux_leak = df_feat.copy()
#             df_feat['time'] = pd.to_datetime(df_feat.index, unit='s')
#
#             # Reshape for long-format plotting or time-series analysis
#             df_feat_melted = df_feat.melt(
#                 ignore_index=False,
#                 var_name=var_id,
#                 value_name=feature
#             ).reset_index()
#
#             df_feat_melted = df_feat_melted.rename(columns={'time': 'Time'})
#             df_feat_melted['scenario'] = leak_scenario
#
#             aux_data.append(df_feat_melted)
#             aux_leak['scenario'] = leak_scenario
#             aux_data_leak.append(aux_leak)
#
#         # Save full leak time series and melted DataFrame
#         df_leak = pd.concat(aux_data_leak).reset_index()
#         df_leak = df_leak.rename(columns={'time': 'timestamp'})
#
#         df_leak.to_csv(f'{save_path}/{feature}.csv',
#                        index=False)