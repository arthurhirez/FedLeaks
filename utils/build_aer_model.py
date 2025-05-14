import json
from copy import deepcopy

from backbone.AER import AER

def build_model_from_json_file(json_path: str, custom_args: dict = None, dense_layer=1) -> AER:
    """
    Builds an AER model from a JSON config file, optionally overriding
    default hyperparameters with `custom_args`.
    """

    def resolve_placeholders(config, hyperparams):
        """Recursively replace {'__RESOLVE__': key} with hyperparams[key]."""
        if isinstance(config, dict):
            if '__RESOLVE__' in config:
                key = config['__RESOLVE__']
                # print(key, hyperparams[key])
                return hyperparams[key]
            return {k: resolve_placeholders(v, hyperparams) for k, v in config.items()}
        elif isinstance(config, list):
            return [resolve_placeholders(item, hyperparams) for item in config]
        else:
            return config

    # Load the JSON config
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Extract fixed hyperparameters from JSON
    fixed = config.get('hyperparameters', {}).get('fixed', {})
    hyperparams = {key: value['default'] for key, value in fixed.items() if 'default' in value}


    # Override with custom args if provided
    if custom_args:
        hyperparams.update(custom_args)

    # Resolve "__RESOLVE__" references in layer definitions
    hyperparams['layers_encoder'] = resolve_placeholders(hyperparams['layers_encoder'], hyperparams)
    hyperparams['layers_decoder'] = resolve_placeholders(hyperparams['layers_decoder'], hyperparams)

    # # Patch dense output layer if needed
    # if dense_layer:
    #     try:
    #         hyperparams['layers_decoder'][2]['parameters']['layer']['parameters']['out_features'] = dense_layer
    #     except KeyError:
    #         raise KeyError('Layer decoder must specify dense layers.')

    # Define expected constructor arguments
    constructor_keys = [
        'optimizer', 'learning_rate', 'epochs', 'batch_size',
        'shuffle', 'verbose', 'callbacks', 'reg_ratio',
        'lstm_units', 'validation_split', 'layers_encoder', 'layers_decoder',
        'window_size', 'input_shape', 'target_shape'
    ]
    constructor_kwargs = {k: hyperparams.pop(k) for k in constructor_keys if k in hyperparams}

    return AER(**constructor_kwargs, **hyperparams)

