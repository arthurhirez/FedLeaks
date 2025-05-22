import numpy as np

# Based on populational density, there will be different consumption patterns
## The morning/afternoon/night consumption peaks will vary
## [morning_peak, afternoon_peak, evening_peak, night_consumption, variation_strength]
DISTRIBUTION_PATTERNS = {
    'low': np.array([4.0, 1.0, 4.0, 0.2, 0.125]),
    'medium': np.array([3.0, 1.7, 2.0, 0.5, 0.075]),
    'high': np.array([0.8, 0.8, 1.2, 0.3, 0.010]),
}

MAPPING_SENSORS = {
    'Graeme': {
        'Pressure': {
            'Client_A': ['64', '75'],
            'Client_B': ['79', '86'],  # '93',
            'Client_C': ['74', '58'],
            'Client_D': ['31', '26'],
            'Client_E': ['41', '12'],
        },
        'Flow': {
            'Client_A': ['102', '100', '85'],
            'Client_B': ['157', '135', '165'],
            'Client_C': ['110', '119', '91'],
            'Client_D': ['38', '96', '69'],
            'Client_E': ['59', '11', '19'],
        },
        'Fixed': ['79', '58', '24', '98', '75', '23', '27'],
        'Skip': ['114']
    },
    'Balerma': {
        'Pressure': {
            'Client_A': ['345', '187'],
            'Client_B': ['372', '106'],
            'Client_C': ['27', '63'],
        },
        'Flow': {
            'Client_A': ['512', '593', '317'],
            'Client_B': ['550', '349', '131'],
            'Client_C': ['575', '149', '221'],
        },
        'Fixed': ['90', '13', '365', '268', '249', '38', '63'],
        'Skip': ['38', '43', '44', '88']
    },
    'CTown': {
        'Pressure': {
            'Client_A': ['J67', 'J134'],
            'Client_B': ['J320', 'J190'],  # '93',
            'Client_C': ['J411', 'J194'],
        },
        'Flow': {
            'Client_A': ['P1036', 'P86', 'P947'],
            'Client_B': ['P840', 'P468', 'P772'],
            'Client_C': ['P19', 'P23', 'P13'],
        },
        'Fixed': ['TEM QUE ESCOLHER'],
        'Skip': ['TEM QUE IMPLEMENTAR >> NODE FOR NODE IF "J" NOT IN NODE']
    }
}


def get_domain_sensor_map(domain_name: str):
    """Return pressure and flow node lists for a given client name."""
    pressure_nodes = MAPPING_SENSORS[domain_name].get('Pressure')
    flow_nodes = MAPPING_SENSORS[domain_name].get('Flow')

    fixed_nodes = MAPPING_SENSORS[domain_name].get('Fixed')
    skip_nodes = MAPPING_SENSORS[domain_name].get('Skip')

    if pressure_nodes is None or flow_nodes is None:
        raise ValueError(f"Unknown client '{domain_name}' in sensor maps.")

    return pressure_nodes, flow_nodes, fixed_nodes, skip_nodes
