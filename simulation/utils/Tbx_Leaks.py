import pickle
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from xml.dom import minidom

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd

alt.data_transformers.enable("vegafusion")

import wntr

import os
os.environ['PYTHONIOENCODING'] = 'utf-8-sig'  # or try the others below




def create_leak_config(leak_name: str, leaks: List[Dict]) -> ET.Element:
    leak_config = ET.Element("LeakConfig", name=leak_name)
    for leak in leaks:
        leak_attrs = {
            "pipeId": str(leak["pipeId"]),
            "type": leak["type"],
            "diameter": str(leak["diameter"]),
            "start": str(leak["start"]),
            "end": str(leak["end"])
        }
        if leak["type"] == "incipient":
            leak_attrs["peak"] = str(leak["peak"])
        ET.SubElement(leak_config, "Leak", **leak_attrs)
    return leak_config


def create_config_xml(
        id_scenario: str,
        output_file: str,
        network_path: str,
        iterations: int,
        timestep: int,
        leak_scenarios: Dict[str, List[Dict]],
        pressure_sensors: Optional[List[int]] = None,
        flow_sensors: Optional[List[int]] = None,
        demand_sensors: Optional[List[int]] = None
):
    root = ET.Element("ScenarioCollection")

    scenario = ET.SubElement(
        root, "Scenario",
        name=id_scenario,
        network=network_path,
        iterations=str(iterations),
        timeStep=str(timestep)
    )

    # SensorConfigs
    sensor_configs = ET.SubElement(scenario, "SensorConfigs")
    sensor_config = ET.SubElement(sensor_configs, "SensorConfig", name="DefaultSensors")

    if pressure_sensors:
        pressure_elem = ET.SubElement(sensor_config, "PressureSensors")
        for sid in pressure_sensors:
            ET.SubElement(pressure_elem, "Sensor", id=str(sid))
    if flow_sensors:
        flow_elem = ET.SubElement(sensor_config, "FlowSensors")
        for sid in flow_sensors:
            ET.SubElement(flow_elem, "Sensor", id=str(sid))
    if demand_sensors:
        demand_elem = ET.SubElement(sensor_config, "DemandSensors")
        for sid in demand_sensors:
            ET.SubElement(demand_elem, "Sensor", id=str(sid))

    # SensorfaultConfigs
    sensor_fault_configs = ET.SubElement(scenario, "SensorfaultConfigs")
    ET.SubElement(sensor_fault_configs, "SensorfaultConfig", name="GT")

    # LeakConfigs
    leak_configs = ET.SubElement(scenario, "LeakConfigs")
    ET.SubElement(leak_configs, "LeakConfig", name="Baseline")  # no-leak scenario

    for leak_name, leak_list in leak_scenarios.items():
        leak_config_elem = create_leak_config(leak_name, leak_list)
        leak_configs.append(leak_config_elem)

    # Beautify XML output
    xml_str = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(xml_str)
    pretty_xml_as_str = parsed.toprettyxml(indent="    ")

    with open(output_file, 'w') as f:
        f.write(pretty_xml_as_str)

    print(f"✅ Config XML written to: {output_file}")


def generate_one_leak_per_pipe(
        pipe_ids: List[int],
        iterations: int,
        leak_type: str = "abrupt",  # or "incipient"
        diameter: float = 0.1,
        start: int = 73,
        end: int = 144,
        peak_offset: Optional[int] = None  # Only for incipient
) -> Dict[str, List[Dict]]:
    scenarios = {}

    for pipe_id in pipe_ids:
        leak = {
            "pipeId": pipe_id,
            "type": leak_type,
            "diameter": diameter,
            "start": start,
            "end": end
        }

        if leak_type == "incipient":
            if peak_offset is not None:
                leak["peak"] = start + peak_offset
            else:
                leak["peak"] = start + (end - start) // 2  # Default to middle

        scenario_name = f"Pipe{pipe_id}_{leak_type.capitalize()}"
        scenarios[scenario_name] = [leak]

    return scenarios


def generate_random_leaks_for_all_pipes(
        water_network,
        # pipe_ids: List[int],
        iterations: int,
        num_scenarios: int = 5,
        leaks_per_scenario: int = 2,
        # diameters: List[float] = [0.05, 0.1, 0.15],
        types: List[str] = ['abrupt', 'incipient'],
        severity=0.45,
        min_duration: int = 24,
        max_duration: int = 96,
        save_dict_path=None
) -> Dict[str, List[Dict]]:
    scenarios = {}

    diameters_network = set()
    pipe_ids = water_network.pipe_name_list

    # pipe_ids = [pipe for pipe in water_network.pipe_name_list if 'P' in pipe]

    for pipe_id in pipe_ids:
        pipe = water_network.get_link(pipe_id)
        diameters_network.add(pipe.diameter)

    diameters_network = list(diameters_network)

    for i in range(num_scenarios):
        leak_list = []
        used_pipes = random.sample(pipe_ids, k=min(leaks_per_scenario, len(pipe_ids)))

        for pipe_id in used_pipes:
            pipe = water_network.get_link(pipe_id)
            pipe_diameter = pipe.diameter

            diameters_viable = [d for d in diameters_network if d <= pipe_diameter]

            leak_type = 'abrupt' if np.random.uniform(low=0., high=1.) > 0.75 else 'incipient'
            diameter = np.random.uniform(low=0.1, high=severity) * random.choice(diameters_viable)
            start = random.randint(0, iterations - max_duration - 1)
            duration = random.randint(min_duration, max_duration)
            end = start + duration
            leak = {
                "pipeId": pipe_id,
                "type": leak_type,
                "diameter": diameter,
                "start": start,
                "end": end
            }
            if leak_type == "incipient":
                leak["peak"] = start + duration // 2
            leak_list.append(leak)

        name = f"AutoScenario_{i + 1}"
        scenarios[name] = leak_list

    if save_dict_path is not None:
        with open(save_dict_path, 'wb') as file:
            pickle.dump(scenarios, file)

    return scenarios


def get_link_and_adjacent_nodes(graph: nx.Graph, link: tuple) -> list:
    """
    For a given link (edge) in a NetworkX graph, returns a list containing
    the two nodes of the link and all their adjacent nodes.

    Args:
        graph: The NetworkX graph.
        link: A tuple representing the link (node1, node2).

    Returns:
        A list of unique nodes including the link's nodes and their neighbors.
    """
    u, v = link
    adjacent_nodes = set()
    adjacent_nodes.add(u)
    adjacent_nodes.add(v)
    for neighbor in graph.neighbors(u):
        adjacent_nodes.add(neighbor)
    for neighbor in graph.neighbors(v):
        adjacent_nodes.add(neighbor)
    return list(adjacent_nodes)


def get_node_names(pipe_id: str, wn: wntr.network.WaterNetworkModel) -> tuple:
    """
    Retrieves the start and end node names for a given pipe ID from a WNTR WaterNetworkModel.

    Args:
        pipe_id: The ID of the pipe in the WaterNetworkModel.
        wn: The WNTR WaterNetworkModel object.

    Returns:
        A tuple containing the start node name and the end node name.
        Returns (None, None) if the pipe ID is not found.
    """
    try:
        link = wn.get_link(pipe_id)
        start_node_name = link.start_node.name
        end_node_name = link.end_node.name
        return start_node_name, end_node_name
    except KeyError:
        return None, None


def get_target_sensors(water_network, auto_leak_config, skip_cases=None, fixed_cases=None):
    # Flatten the data
    flat_data = []
    for scenario, events in auto_leak_config.items():
        for event in events:
            entry = event.copy()
            entry['scenario'] = scenario
            flat_data.append(entry)

    # Create the DataFrame
    df_leaks = pd.DataFrame(flat_data)

    # Apply the function to each row of the DataFrame
    df_leaks[['node_Id_A', 'node_Id_B']] = df_leaks['pipeId'].apply(lambda x: get_node_names(x, water_network)).tolist()

    # Optional: reorder columns
    columns_order = ['scenario', 'pipeId', 'node_Id_A', 'node_Id_B', 'type', 'diameter', 'start', 'end', 'peak']
    df_leaks = df_leaks.reindex(columns=columns_order)

    wn_G = water_network.to_graph()
    wn_G = wn_G.to_undirected()

    all_links_info = {
        scenario: {
            pipe: [] for pipe in df_leaks[df_leaks['scenario'] == scenario]['pipeId'].unique()
        }
        for scenario in df_leaks['scenario'].unique()
    }

    for index, row in df_leaks.iterrows():
        link = (row['node_Id_A'], row['node_Id_B'])
        adjacent_nodes_list = get_link_and_adjacent_nodes(wn_G, link)
        all_links_info[row['scenario']][row['pipeId']].append(link)
        all_links_info[row['scenario']][row['pipeId']].append(adjacent_nodes_list)

    scenarios_data = []
    for scenario, pipe_dict in all_links_info.items():
        for pipe_id, (endpoints, path) in pipe_dict.items():
            for node in path:
                scenarios_data.append({
                    'scenario': scenario,
                    'pipeId': pipe_id,
                    'node': node
                })

    scenarios_df = pd.DataFrame(scenarios_data)

    # Step 3: Altair visualization
    # Betweenness centrality measures how often a node lies on the shortest path between other nodes.
    betweenness_centrality = nx.betweenness_centrality(wn_G)
    highest_betweenness_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:7]
    if fixed_cases is not None:
        highest_betweenness_nodes += fixed_cases

    tgt_nodes = list(set(scenarios_df['node'].tolist() + highest_betweenness_nodes))

    if skip_cases is not None:
        tgt_nodes = [node for node in tgt_nodes if node not in skip_cases]

    tgt_links = scenarios_df['pipeId'].unique().tolist()

    return tgt_nodes, tgt_links, scenarios_df



def plot_networks_leaks(water_network, data_scenarios, domain_x=None, domain_y=None, plot_size=(400, 300)):
    ###################### Step 0: Create selection ######################
    scenarios = data_scenarios['scenario'].dropna().sort_values().drop_duplicates().tolist()
    scenarios.append('Baseline')
    scenario_dropdown = alt.binding_select(options=scenarios, name="Scenarios")
    scenario_select = alt.selection_point(fields=['scenario'], bind=scenario_dropdown, value=scenarios[0])

    selector_node = alt.selection_point(fields=['node'])
    selector_pipe = alt.selection_point(fields=['pipeId'])

    color = (
        alt.when(selector_node)
        .then(alt.Color("node:O").legend(None))
        .otherwise(alt.value("lightgray"))
    )

    ###################### Step 1: Node positions ######################
    node_data = []
    for name, node in water_network.nodes():
        x, y = node.coordinates
        node_data.append({'node': name, 'x': x, 'y': y})
    nodes_df = pd.DataFrame(node_data)

    nodes_plot = pd.merge(nodes_df, data_scenarios[['scenario', 'node']].drop_duplicates(),
                          on='node', how='left')

    ## Betweenness centrality measures how often a node lies on the shortest path between other nodes.
    wn_G = water_network.to_graph()
    wn_G = wn_G.to_undirected()

    betweenness_centrality = nx.betweenness_centrality(wn_G)
    highest_betweenness_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:7]

    BC_nodes = nodes_plot[nodes_plot['node'].isin(highest_betweenness_nodes)].copy()
    BC_nodes['scenario'] = 'Baseline'

    # Base chart config for nodes
    node_base = alt.Chart(nodes_plot).encode(
        x=alt.X('x', scale=alt.Scale(domain=domain_x)),
        y=alt.Y('y', scale=alt.Scale(domain=domain_y)),
        tooltip=['node']
    ).properties(
        width=plot_size[0],
        height=plot_size[1]
    )

    # Base chart config for nodes with betweness centrality
    node_BC = alt.Chart(BC_nodes).encode(
        x=alt.X('x', scale=alt.Scale(domain=domain_x)),
        y=alt.Y('y', scale=alt.Scale(domain=domain_y)),
        tooltip=['node']
    ).properties(
        width=plot_size[0],
        height=plot_size[1]
    ).mark_circle(size=50, color='green')

    # Background nodes: gray and smaller
    nodes_background = node_base.mark_circle(size=30, color='lightgray')

    # Foreground nodes: filtered by scenario, blue and larger
    nodes_highlight = node_base.mark_circle(size=70, color='blue').transform_filter(
        scenario_select
    )

    # Combine node layers
    nodes_chart = alt.layer(
        nodes_background,
        nodes_highlight,
        node_BC
    ).add_params(
        scenario_select
    ).add_params(
        selector_node
    ).properties(title="Dropdown Filtering - Nodes")

    ###################### Step 2: Edges with coordinates ######################
    edge_data = []
    for l in water_network.links():
        x1, y1 = water_network.get_node(l[1]._start_node.name).coordinates
        x2, y2 = water_network.get_node(l[1]._end_node.name).coordinates
        edge_data.append({'pipeId': l[1]._link_name, 'node_u': l[1]._start_node.name, 'x': x1, 'y': y1,
                          'node_v': l[1]._end_node.name, 'x2': x2, 'y2': y2})
    edges_df = pd.DataFrame(edge_data)

    edges_plot = pd.merge(edges_df, data_scenarios[['scenario', 'pipeId']].drop_duplicates(),
                          on='pipeId', how='left')

    # Base chart shared config
    base = alt.Chart(edges_plot).encode(
        x=alt.X('x', scale=alt.Scale(domain=domain_x)),
        y=alt.Y('y', scale=alt.Scale(domain=domain_y)),
        x2='x2',
        y2='y2',
        tooltip=['pipeId']
    ).properties(
        width=plot_size[0],
        height=plot_size[1]
    )

    # Background layer: all edges in gray and thin
    background = base.mark_line(stroke='lightgray', strokeWidth=3)

    # Foreground layer: only selected scenario, in red and thick
    highlight = base.mark_line(stroke='red', strokeWidth=7).transform_filter(
        scenario_select
    )

    # Combine layers
    edges_chart = alt.layer(
        background,
        highlight
    ).add_params(
        selector_pipe
    ).properties(title="Dropdown Filtering").interactive()

    # edges_nodes = pd.melt(
    #     edges_plot[['pipeId', 'node_u', 'node_v']],
    #     id_vars=['pipeId'],
    #     value_name='node'
    # ).drop(columns='variable')

    # selector_pipe_nodes = alt.selection_point(fields=['pipeId'], name='SelectPipe')
    # pipe_node_filter = alt.Chart(edges_nodes).transform_filter(
    #     selector_pipe_nodes
    # )

    # Combine both layers
    network_chart = edges_chart + nodes_chart
    network_chart.configure_view(stroke=None)

    return network_chart, scenario_select, selector_node, selector_pipe


def plot_feature_network(leaks_scenarios, scenario_selector, node_selector, link_selector, feature='Demand',
                         plot_size=(400, 300)):
    leak_configs = leaks_scenarios.list_configs()['LeakConfigs']

    aux_data = []
    var_id = 'pipeId' if feature.lower() == 'flow' else 'node'

    for leak_scenario in leak_configs:
        data_scenario = leaks_scenarios.get(leak_scenario, 'DefaultSensors', 'GT')
        df_feat = data_scenario[feature.lower()].copy()

        for c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(float).round(6)

        df_feat = df_feat.copy()
        df_feat['time'] = pd.to_datetime(df_feat.index, unit='s')

        df_feat = df_feat.set_index(pd.DatetimeIndex(df_feat['time'])).drop(columns='time')

        # Melt the DataFrame, keeping the index
        df_feat_melted = df_feat.melt(
            ignore_index=False,  # Keep the index
            var_name=var_id,  # Name for the column containing original column names (nodes)
            value_name=feature  # Name for the column containing the demand values
        ).reset_index()  # Promote the index to a regular column

        # Rename the index column to 'Time' (or whatever makes sense for your x-axis)
        df_feat_melted = df_feat_melted.rename(columns={'time': 'Time'})
        df_feat_melted['scenario'] = leak_scenario
        aux_data.append(df_feat_melted)

    df_plot = pd.concat(aux_data)

    # Create the Altair chart
    if feature.lower() == 'flow':
        line = alt.Chart(df_plot).mark_line(interpolate="basis").encode(
            x=alt.X('Time:T', title='Time'),
            y=alt.Y(f"{feature}:Q", title=feature),
            color="pipeId:N"  # Use the 'node' column for color encoding
        ).transform_filter(
            link_selector
        ).transform_filter(
            scenario_selector
        ).properties(
            width=plot_size[0],
            height=plot_size[1]
        )
    else:
        line = alt.Chart(df_plot).mark_line(interpolate="basis").encode(
            x=alt.X('Time:T', title='Time'),
            y=alt.Y(f"{feature}:Q", title=feature),
            color="node:N"  # Use the 'node' column for color encoding
        ).transform_filter(
            node_selector
        ).transform_filter(
            scenario_selector
        ).properties(
            width=plot_size[0],
            height=plot_size[1]
        )

    return line