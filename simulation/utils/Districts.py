import random

import altair as alt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

alt.data_transformers.enable("vegafusion")

from collections import Counter

import glob
import wntr

import os



class BuildingAssigner:
    def __init__(self, node_list, water_network, income_level='medium', density_level='medium', seed=23):
        """
        Initialize the building assigner using high-level urban characteristics.

        Parameters:
        -----------
        node_list : list of str
            List of node (junction) names in the district.

        water_network : wntr WaterNetworkModel
            The EPANET network model for the district.

        income_level : str
            One of 'low', 'medium', 'high' to influence standards.

        density_level : str
            One of 'low', 'medium', 'high' to influence building types.

        seed : int, optional
            Seed for reproducibility
        """
        self.node_list = node_list
        self.water_network = water_network
        self.standard_consumption = {1: 5, 2: 10, 3: 15, 4: 25, 5: 40}  # m³/month

        # Generate building templates + profile from high-level inputs
        houses, buildings_P, buildings_M, buildings_G, profile = generate_building_profiles(
            income_level=income_level,
            density_level=density_level,
            seed=seed
        )

        # Store results
        self.templates = {
            'houses': houses,
            'buildings_P': buildings_P,
            'buildings_M': buildings_M,
            'buildings_G': buildings_G
        }
        self.district_profile = profile
        self.data_buildings = None  # will be populated later

    def assign_buildings(self):
        """
        Assign building types to each node in the node list using the generated profile and templates.

        Returns:
        --------
        pd.DataFrame with columns: ['node_id', 'type', 'standard', 'consumption', 'units']
        """
        total_nodes = len(self.node_list)
        assignments = []
        node_index = 0

        # Shuffle nodes to randomize placement
        random.shuffle(self.node_list)

        for group in self.district_profile:
            template_name = group['template']
            ratio = group['ratio']
            config = self.templates[template_name]
            n_group = round(total_nodes * ratio)
            standards = list(config['standards'].items())
            building_type = config['type']

            for _ in range(n_group):
                if node_index >= total_nodes:
                    break
                node_id = self.node_list[node_index]
                standard = self._weighted_choice(standards)
                consumption = self.standard_consumption[standard]
                units = None

                # if building_type.lower() == 'apartments':
                units = random.randint(*config['units_range'])
                consumption *= units

                assignments.append({
                    'node_id': node_id,
                    'type': building_type,
                    'standard': standard,
                    'consumption': consumption,
                    'units': units
                })

                node_index += 1

        # Fill remaining nodes with default House type (mid-standard)
        while node_index < total_nodes:
            node_id = self.node_list[node_index]
            assignments.append({
                'node_id': node_id,
                'type': 'House',
                'standard': 2,
                'consumption': self.standard_consumption[2],
                'units': random.randint(2, 10)
            })
            node_index += 1

        # Create DataFrame
        results_df = pd.DataFrame(assignments)
        results_df['units'] = results_df['units'].fillna(1).astype(int)
        results_df['consumption_unit'] = results_df['consumption'] / results_df['units']

        self.data_buildings = results_df

    def _weighted_choice(self, weighted_dict):
        options, weights = zip(*weighted_dict)
        return random.choices(options, weights=weights)[0]

    def describe_configuration(self):
        print("District Profile (template ratios):")
        for item in self.district_profile:
            print(f"  - {item['template']}: {item['ratio']:.3f}")

        print("\nStandards per template:")
        for name, template in self.templates.items():
            std = template['standards']
            print(f"  {name}: {std}")

    def edit_node(self, node_id, income_level='medium', density_level='medium', seed=None):
        """
        Edit a specific node's building assignment by reassigning it based on new income/density levels.

        Parameters:
        -----------
        node_id : str
            The node to be updated.

        income_level : str
            One of 'low', 'medium', 'high' to influence building standards.

        density_level : str
            One of 'low', 'medium', 'high' to influence building type.

        seed : int, optional
            Optional seed for deterministic assignment.
        """
        if self.data_buildings is None:
            raise ValueError("Buildings must be assigned first. Call assign_buildings() before editing nodes.")

        if node_id not in self.node_list:
            raise ValueError(f"Node '{node_id}' is not in the list of assigned nodes.")

        # Generate new templates/profile for just this node
        houses, buildings_P, buildings_M, buildings_G, profile = generate_building_profiles(
            income_level=income_level,
            density_level=density_level,
            seed=seed
        )

        templates = {
            'houses': houses,
            'buildings_P': buildings_P,
            'buildings_M': buildings_M,
            'buildings_G': buildings_G
        }

        # Randomly choose a template based on the new profile
        templates_with_weights = [(p['template'], p['ratio']) for p in profile]

        while True:
            selected_template = random.choices(
                [tpl for tpl, _ in templates_with_weights],
                weights=[w for _, w in templates_with_weights]
            )[0]
            if density_level != 'low' and selected_template != 'houses':
                break

        config = templates[selected_template]
        standard = self._weighted_choice(list(config['standards'].items()))
        consumption = self.standard_consumption[standard]
        building_type = config['type']
        # units = 1
        #
        # if building_type.lower() == 'apartments':
        units = random.randint(*config['units_range'])
        consumption *= units

        # Update the DataFrame
        new_row = {
            'node_id': node_id,
            'type': building_type,
            'standard': int(standard),
            'consumption': int(consumption),
            'units': int(units),
            'consumption_unit': consumption / units
        }

        new_row_df = pd.DataFrame([new_row], columns=self.data_buildings.columns)
        self.data_buildings.loc[self.data_buildings['node_id'] == node_id] = new_row_df.values


    def plot_assignments(self, show_edges=True, figsize=(12, 10)):
        """
        Plot the network with assigned buildings.

        Parameters:
        -----------
        show_edges : bool
            Whether to plot pipes between nodes.

        figsize : tuple
            Size of the figure.
        """
        # Map of markers by type
        type_marker = {'House': 'o', 'Apartments': 's'}
        standard_colors = {1: '#d73027', 2: '#fc8d59', 3: '#fee090', 4: '#91bfdb', 5: '#4575b4'}

        fig, ax = plt.subplots(figsize=figsize)

        # Optional: draw edges
        if show_edges:
            for link_name, link in self.water_network.links():
                start = self.water_network.get_node(link.start_node_name)
                end = self.water_network.get_node(link.end_node_name)
                x1, y1 = start.coordinates
                x2, y2 = end.coordinates
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1, zorder=1)

        # Draw nodes with styling
        for _, row in self.data_buildings.iterrows():
            node = self.water_network.get_node(row['node_id'])
            x, y = node.coordinates
            marker = type_marker.get(row['type'], 'x')
            color = standard_colors.get(row['standard'], 'black')

            # Marker size: fixed for houses, scaled for apartments
            if row['type'] == 'Apartments' and pd.notna(row['units']):
                size = 40 + row['units'] * 0.5  # scalable size
            else:
                size = 80  # fixed size for houses

            ax.scatter(x, y, c=color, marker=marker, edgecolor='k', s=size, zorder=2)

        ax.set_title("Building Assignments in Water Network", fontsize=14)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)

        # Legend
        house_patch = mlines.Line2D([], [], color='w', marker='o', label='House', markerfacecolor='gray',
                                    markeredgecolor='k', markersize=8)
        apt_patch = mlines.Line2D([], [], color='w', marker='s', label='Apartments (size ∝ units)',
                                  markerfacecolor='gray', markeredgecolor='k', markersize=8)
        std_patches = [mpatches.Patch(color=col, label=f'Standard {std}') for std, col in standard_colors.items()]
        ax.legend(handles=[house_patch, apt_patch] + std_patches, loc='best', fontsize=9)

        plt.tight_layout()
        plt.show()

    def plot_interactive_assignments(self, plot_size=(600, 400)):
        """
        Interactive Altair plot of the water network:
        - Nodes: colored, shaped, sized by attributes
        - Pipes: light gray lines
        - Filters: by type and standard
        """
        # 1. Node positions
        node_data = []
        for name, node in self.water_network.nodes():
            x, y = node.coordinates
            node_data.append({'node_id': name, 'x': x, 'y': y})
        nodes_df = pd.DataFrame(node_data)

        # 2. Pipe edges
        edge_data = []
        for link_name, link in self.water_network.links():
            n1 = self.water_network.get_node(link.start_node_name)
            n2 = self.water_network.get_node(link.end_node_name)
            x1, y1 = n1.coordinates
            x2, y2 = n2.coordinates
            edge_data.append({'x': x1, 'y': y1, 'x2': x2, 'y2': y2})
        edges_df = pd.DataFrame(edge_data)

        # 3. Merge assignments with positions
        shape_map = {'House': 'circle', 'Apartments': 'square'}
        color_map = {1: '#d73027', 2: '#fc8d59', 3: '#fee090', 4: '#91bfdb', 5: '#4575b4'}

        df = self.data_buildings.merge(nodes_df, on='node_id', how='left')
        df['shape'] = df['type'].map(shape_map)
        df['color'] = df['standard'].map(color_map)
        df['size'] = df.apply(
            lambda row: 40 + row['units'] * 0.5 if row['type'] == 'Apartments' and pd.notna(row['units']) else 25,
            axis=1
        )

        # 4. Create selections
        type_select = alt.selection_point(fields=['type'], bind='legend')
        std_select = alt.selection_point(fields=['standard'], bind='legend')

        # 5. Edge chart (always visible)
        edge_chart = alt.Chart(edges_df).mark_line(
            color='gray', strokeWidth=1, opacity=0.75
        ).encode(
            x='x:Q', y='y:Q', x2='x2:Q', y2='y2:Q'
        )

        # 6. Node chart (filtered)
        node_chart = alt.Chart(df).mark_point(
            filled=True, stroke='black', strokeWidth=0.75, fillOpacity=0.95
        ).encode(
            x='x:Q',
            y='y:Q',
            color=alt.Color('standard:N',
                            scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                            legend=alt.Legend(title='Standard')),
            shape=alt.Shape('type:N',
                            scale=alt.Scale(domain=list(shape_map.keys()), range=list(shape_map.values())),
                            legend=alt.Legend(title='Building Type')),
            size=alt.Size('size:Q', legend=None),
            tooltip=[
                alt.Tooltip('node_id:N', title='Node'),
                alt.Tooltip('type:N', title='Type'),
                alt.Tooltip('standard:N', title='Standard'),
                alt.Tooltip('units:Q', title='Units')
            ]
        ).add_params(
            type_select, std_select
        ).transform_filter(
            type_select & std_select
        )

        # 7. Combine charts
        full_chart = (edge_chart + node_chart).properties(
            width=plot_size[0],
            height=plot_size[1],
            title='Building Assignments in Water Network',
        ).configure_view(
            stroke=None
        ).interactive()

        full_chart = (edge_chart + node_chart).interactive()

        return full_chart


def load_network_and_districts(id_network, directory='networks\\original\\'):
    """
    Loads the full EPANET network and all associated district subnetwork files.

    Parameters:
    -----------
    id_network : str
        Base name of the full network file (without extension).

    directory : str
        Path to the folder containing the full network and district files.

    Returns:
    --------
    tuple:
        - wn_full : wntr WaterNetworkModel of the full network
        - district_nodes : dict {district_name: list of junction names}
    """
    # Pattern to find all district files
    pattern = os.path.join(directory, f'*{id_network}*District*.inp')
    district_files = glob.glob(pattern)

    # Load full network
    full_path = os.path.join(directory, f'{id_network}.inp')
    wn_full = wntr.network.WaterNetworkModel(full_path)
    all_nodes = wn_full.junction_name_list

    # Load districts
    district_nodes = {}
    for district in district_files:
        name = district.replace(os.path.join(directory, f'{id_network}_'), '').replace('.inp', '')
        wn = wntr.network.WaterNetworkModel(district)
        district_nodes[name] = {}
        district_nodes[name]['nodes'] = wn.junction_name_list
        district_nodes[name]['network'] = wn

        # Sanity check of district division
    results = check_district_node_coverage(full_node_list=all_nodes,
                                           district_node_dict=district_nodes,
                                           return_dict=True,
                                           verbose=True)
    if results['is_valid']:
        print("\nAll district assignments are valid.")

        return wn_full, district_nodes

    raise AttributeError('Please check the division of the network. Some nodes are overlapping/missing.')


def check_district_node_coverage(full_node_list, district_node_dict, return_dict=False, verbose=True):
    """
    Checks that district node assignments:
    1. Do not overlap (no node appears in more than one district)
    2. Fully cover the nodes in the full network (no missing or extra nodes)

    Parameters:
    -----------
    full_node_list : list
        List of all junction names in the full network.

    district_node_dict : dict
        Dictionary with district names as keys and lists of junction names as values.

    return_dict : bool
        If True, Return results as dictionary.

    verbose : bool
        If True, prints details about overlaps and coverage issues.

    Returns:
    --------
    dict with keys:
        - 'overlapping_nodes': list of nodes found in multiple districts
        - 'missing_in_districts': nodes in full network but missing from districts
        - 'extra_in_districts': nodes in districts not present in full network
        - 'is_valid': True if all checks pass
    """
    # Flatten all district nodes
    all_district_nodes_flat = [node for nodes in district_node_dict.values() for node in nodes['nodes']]
    node_counts = Counter(all_district_nodes_flat)

    # Nodes in more than one district
    overlapping_nodes = [node for node, count in node_counts.items() if count > 1]

    # Set comparisons
    district_node_set = set(all_district_nodes_flat)
    full_node_set = set(full_node_list)

    missing_in_districts = full_node_set - district_node_set
    extra_in_districts = district_node_set - full_node_set

    # Verbose output
    if verbose:
        print("✅ No overlapping nodes." if not overlapping_nodes else "❌ Overlapping nodes found:")
        if overlapping_nodes:
            print(overlapping_nodes)

        print(
            "✅ All full network nodes are included." if not missing_in_districts else "❌ Missing nodes from districts:")
        if missing_in_districts:
            print(missing_in_districts)

        print(
            "✅ No extra nodes in districts." if not extra_in_districts else "⚠️ Extra nodes in districts not in full network:")
        if extra_in_districts:
            print(extra_in_districts)

    # Return results as dictionary
    return {
        'overlapping_nodes': overlapping_nodes,
        'missing_in_districts': list(missing_in_districts),
        'extra_in_districts': list(extra_in_districts),
        'is_valid': not (overlapping_nodes or missing_in_districts or extra_in_districts)
    } if return_dict else None


def combine_districts(district_nodes, init_assigner = True, plot_size=(600, 300)):
    """
    Assigns and plots interactive building distributions for all districts.

    Parameters:
    -----------
    district_nodes : dict
        A dictionary where keys are district names and values are dicts with:
            - 'nodes': list of node IDs
            - 'network': wntr WaterNetworkModel
    plot_size : tuple
        Width and height of each district plot.

    Returns:
    --------
    alt.Chart
        A combined interactive Altair chart for all districts.
    """
    charts = []


    for district_name, district_data in district_nodes.items():
        if init_assigner:
            assigner = BuildingAssigner(
                node_list=district_data['nodes'],
                water_network=district_data['network'],
                income_level=district_data['income'],
                density_level=district_data['density'],
            )

            assigner.assign_buildings()
            district_data['assignments'] = assigner

        chart = district_data['assignments'].plot_interactive_assignments(
            plot_size=plot_size,
        ).properties(title=f"District: {district_name}")


        charts.append(chart)

    # Overlay all district charts into one
    full_chart = alt.layer(*charts).properties(
        width=plot_size[0],
        height=plot_size[1],
        title='Building Assignments in All Districts',
    ).configure_view(
        stroke=None
    ).interactive()

    return full_chart


def generate_building_profiles(income_level='medium', density_level='medium', seed=23):
    """
    Generate building templates and district profiles with slight randomness, based on income and density levels.

    Parameters:
    -----------
    income_level : str
        One of 'low', 'medium', 'high'
    density_level : str
        One of 'low', 'medium', 'high'
    seed : int, optional
        Seed for reproducibility

    Returns:
    --------
    houses, buildings_P, buildings_M, buildings_G, district_profile
    """
    if seed is not None:
        random.seed(seed)

    # Base standards by income
    base_standards = {
        'low': {1: 0.40, 2: 0.40, 3: 0.175, 4: 0.025},
        'medium': {1: 0.10, 2: 0.30, 3: 0.50, 4: 0.10},
        'high': {3: 0.20, 4: 0.40, 5: 0.40}
    }

    def jitter_and_normalize(d, jitter=0.05):
        jittered = {k: max(0, v + random.uniform(-jitter, jitter)) for k, v in d.items()}
        total = sum(jittered.values())
        normalized = {k: round(v / total, 3) for k, v in jittered.items()}
        return normalized

    standards = jitter_and_normalize(base_standards[income_level])

    # Building templates
    houses = {
        'type': 'House',
        'standards': jitter_and_normalize({k: v for k, v in standards.items() if k <= 3 or income_level == 'high'}),
        'units_range': (3, 15)
    }

    buildings_P = {
        'type': 'Apartments',
        'standards': jitter_and_normalize({k: v for k, v in standards.items() if k >= 2}),
        'units_range': (15, 45)
    }

    buildings_M = {
        'type': 'Apartments',
        'standards': jitter_and_normalize({k: v for k, v in standards.items() if k >= 2}),
        'units_range': (50, 125)
    }

    buildings_G = {
        'type': 'Apartments',
        'standards': jitter_and_normalize({k: v for k, v in standards.items() if k >= 3}),
        'units_range': (125, 300)
    }

    # Base profile by density
    base_profile = {
        'low': {'houses': 0.80, 'buildings_P': 0.175, 'buildings_M': 0.025},
        'medium': {'houses': 0.30, 'buildings_P': 0.45, 'buildings_M': 0.25},
        'high': {'houses': 0.075, 'buildings_P': 0.275, 'buildings_M': 0.425, 'buildings_G': 0.225}
    }

    # Jitter and normalize ratios
    raw_ratios = {k: max(0, v + random.uniform(-0.025, 0.025)) for k, v in base_profile[density_level].items()}
    total = sum(raw_ratios.values())
    profile = [{'template': k, 'ratio': round(v / total, 3)} for k, v in raw_ratios.items()]

    return houses, buildings_P, buildings_M, buildings_G, profile
