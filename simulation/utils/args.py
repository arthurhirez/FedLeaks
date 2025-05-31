from argparse import ArgumentParser


def create_experiment_args() -> ArgumentParser:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    level_map = {
        'L': 'low',
        'M': 'medium',
        'H': 'high'
    }

    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)

    # Identification
    parser.add_argument('--experiment_id', type=str, default='Pipeline_DEBUG_ARGS',
                        help='Experiment identifier')
    parser.add_argument('--extra_coments', type=str, default='DEBUG_ARGS',
                        help='Aditional info')
    parser.add_argument('--id_network', type=str, default='Graeme',
                        help='Identifier for the water network model.')


    # Features drift
    parser.add_argument('--tgt_district', type=str, default='District_D',
                        help='Target district name.')
    parser.add_argument('--drift_income', type=str, default='low',
                        help='Income change over time.')
    parser.add_argument('--drift_density', type=str, default='medium',
                        help='Populational desity change over time.')
    parser.add_argument('--seed_node', type=str, default='2',
                        help='Seed node for simulation.')
    parser.add_argument('--income_density_mapping', type=str, default= 'HL_LL_LL_LL_LH',
                        help='List of income:density category mappings. Format: income:density')


    # Processes
    parser.add_argument('--run_simulation', type=bool, default=True)
    parser.add_argument('--process_latents', type=bool, default=True)
    parser.add_argument('--detect_anomalies', type=bool, default=True)
    parser.add_argument('--generate_viz', type=bool, default=True)

    return parser



def add_simulation_args(parser: ArgumentParser) -> None:
    # Time variables
    parser.add_argument('--n_segments', type=int, default=12,
                        help='Number of urban growth simulation segments.')
    parser.add_argument('--epochs_lenght', type=int, default=6,
                        help='Number of epochs per simulation run.')
    parser.add_argument('--days_lenght', type=int, default=5,
                        help='Number of days in each epoch.')
    parser.add_argument('--n_intervals', type=int, default=24,
                        help='Number of time intervals per day.')

    # Hydraulic variables
    parser.add_argument('--unit', type=str, default='l/s',
                        help='Measurement unit for flow.')
    parser.add_argument('--leak_severity', type=float, default=0.15,
                        help='Leak severity multiplier.')