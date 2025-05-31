from datasets import Priv_NAMES as DATASET_NAMES
from datasets import get_private_dataset
from models import get_all_models, get_model

from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('False1', 'no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    
    # Identification
    parser.add_argument('--experiment_id', type=str, default='Pipeline_Full_debug',
                        help='Experiment identifier')
    parser.add_argument('--extra_coments', type=str, default='proto_month_DEBUGANDO',
                        help='Aditional info')
    parser.add_argument('--id_network', type=str, default='Graeme',
                        help='Identifier for the water network model.')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    # Features drift
    parser.add_argument('--tgt_district', type=str, default='District_D',
                        help='Target district name.')
    parser.add_argument('--drift_income', type=str, default='medium',
                        help='Income change over time.')
    parser.add_argument('--drift_density', type=str, default='medium',
                        help='Populational desity change over time.')
    parser.add_argument('--seed_node', type=str, default='12',
                        help='Seed node for simulation.')
    parser.add_argument('--income_density_mapping', type=str, default= 'ML_LM_LH_LL_LL',
                        help='List of income:density category mappings. Format: income:density')


    # Processes
    parser.add_argument('--run_simulation', type=str2bool, default=False)
    parser.add_argument('--federated_training', type=str2bool, default=True)
    parser.add_argument('--process_latents', type=str2bool, default=True)
    parser.add_argument('--detect_anomalies', type=str2bool, default=False)
    parser.add_argument('--generate_viz', type=str2bool, default=False)

    return parser


def add_federated_args(parser: ArgumentParser) -> None:
    # Communication - epochs
    parser.add_argument('--communication_epoch', type=int, default=15,
                        help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='The Local Epoch for each Participant')

    # Data parameter
    parser.add_argument('--dataset', type=str, default='fl_leaks', choices=DATASET_NAMES,
                        help='Which scenario to perform experiments on.')
    parser.add_argument('--domains', type=dict, default={'Graeme': 5}, # TODO REFACTOR TYPE
                        help='Domains and respective number of participants.')

    ## Time series preprocessing
    parser.add_argument('--interval_agg', type=int, default=2 * 60 ** 2,
                        help='Agregation interval (seconds) of time series')
    parser.add_argument('--window_size', type=int, default=84,
                        help='Rolling window length')

    # Model (AER) parameters
    parser.add_argument('--input_size', type=int, default=5,
                        help='Number of sensors')  # TODO adaptar
    parser.add_argument('--output_size', type=int, default=5,
                        help='Shape output - dense layer')
    parser.add_argument('--lstm_units', type=int, default=30,
                        help='Number of LSTM units (the latent space will have dimension 2 times bigger')

    # Participants info
    parser.add_argument('--parti_num', type=int, default=None,
                        help='The Number for Participants. If "None" will be setted as the sum of values described in --domain')
    parser.add_argument('--online_ratio', type=float, default=1,
                        help='The Ratio for Online Clients')

    # Federated parameters
    parser.add_argument('--model', type=str, default='fpl',
                        help='Federated Model name.', choices=get_all_models()) #fedavg
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--pri_aug', type=str, default='weak',  # weak strong
                        help='Augmentation for Private Data')
    parser.add_argument('--learning_decay', type=bool, default=False,
                        help='The Option for Learning Rate Decay')
    parser.add_argument('--averaging', type=str, default='weight',
                        help='The Option for averaging strategy')

    parser.add_argument('--infoNCET', type=float, default=0.02,
                        help='The InfoNCE temperature')
    parser.add_argument('--T', type=float, default=0.05,
                        help='The Knowledge distillation temperature')
    parser.add_argument('--weight', type=int, default=1,
                        help='The Weigth for the distillation loss')


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