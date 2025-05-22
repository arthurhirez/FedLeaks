import warnings
from argparse import ArgumentParser
import pandas as pd
import pickle

import subprocess
import sys
import os

from datasets import Priv_NAMES as DATASET_NAMES
from datasets import get_private_dataset
from models import get_all_models, get_model
from utils.Server import train
from utils.Toolbox_analysis import create_latent_df, process_latent_df
from utils.Toolbox_visualization import format_latent_dict, load_and_scale_data, combine_latents, plot_latent_heatmap, plot_time_series_and_latents
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--run_simulation', type=bool, default=False, help='The Device Id for Experiment')
    parser.add_argument('--detect_anomalies', type=bool, default=False)
    parser.add_argument('--generate_viz', type=bool, default=True, help='Creates and saves interactive visualizations')


    # Communication - epochs
    parser.add_argument('--communication_epoch', type=int, default=3,
                        help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch', type=int, default=3, help='The Local Epoch for each Participant')

    # Participants info
    parser.add_argument('--parti_num', type=int, default=None, help='The Number for Participants. If "None" will be setted as the sum of values described in --domain')
    parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')

    # Data parameter
    parser.add_argument('--dataset', type=str, default='fl_leaks', choices=DATASET_NAMES, help='Which scenario to perform experiments on.')
    parser.add_argument('--experiment_id', type=str, default='Pipeline_Full', help='Experiment identifier')
    parser.add_argument('--extra_coments', type=str, default='proto_month', help='Aditional info')
    parser.add_argument('--domains', type=dict, default={
                                                        'Graeme': 5,
                                                        # 'Balerma': 3,
                                                        },
                        help='Domains and respective number of participants.')

    ## Time series preprocessing
    parser.add_argument('--interval_agg', type=int, default=2 * 60 ** 2,
                        help='Agregation interval (seconds) of time series')
    parser.add_argument('--window_size', type=int, default=84, help='Rolling window length')

    # Model (AER) parameters
    parser.add_argument('--input_size', type=int, default=5, help='Number of sensors')  #TODO adaptar
    parser.add_argument('--output_size', type=int, default=5, help='Shape output - dense layer')
    parser.add_argument('--lstm_units', type=int, default=30,
                        help='Number of LSTM units (the latent space will have dimension 2 times bigger')
    

    # Federated parameters
    parser.add_argument('--model', type=str, default='fpl', help='Federated Model name.', choices=get_all_models()) #fedavg

    parser.add_argument('--structure', type=str, default='homogeneity')

    parser.add_argument('--pri_aug', type=str, default='weak',  # weak strong
                        help='Augmentation for Private Data')
    parser.add_argument('--learning_decay', type=bool, default=False, help='The Option for Learning Rate Decay')
    parser.add_argument('--averaging', type=str, default='weight', help='The Option for averaging strategy')

    parser.add_argument('--infoNCET', type=float, default=0.02, help='The InfoNCE temperature')
    parser.add_argument('--T', type=float, default=0.05, help='The Knowledge distillation temperature')
    parser.add_argument('--weight', type=int, default=1, help='The Weigth for the distillation loss')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.parti_num is None:
        args.parti_num = sum(args.domains.values())

    # If run_simulation is True, execute the Pipeline_Simulation script
    if args.run_simulation:
        subprocess.run([
            sys.executable,
            'Pipeline_Simulation.py',
            '--id_exp', args.experiment_id,
            '--generate_viz', str(args.generate_viz),
            '--callable', str(args.run_simulation)
        ], check=True, cwd='simulation')

    
    results = {}

    for scenario in ['Baseline']:#, 'AutoScenario_1']:
        results[scenario] = {}

        priv_dataset = get_private_dataset(args)

        backbones_list = priv_dataset.get_backbone(
            parti_num=args.parti_num,
            names_list=None,
            n_series=args.input_size
        )

        model = get_model(backbones_list, args, priv_dataset)

        priv_train_loaders, aux_latent = train(
            model=model,
            private_dataset=priv_dataset,
            scenario=scenario,
            args=args
        )

        results[scenario]['lat'] = aux_latent
        results[scenario]['model'] = model

    label_clients = [
        'District_A', 'District_B', 'District_C', 'District_D', 'District_E',
        'District_2A', 'District_2B', 'District_2C'
    ]

    base_index = priv_train_loaders[0]['X_index']
    latent_dfs = {}

    for scenario, case in results.items():
        latent_dfs[scenario] = {}
        for epoch, space in enumerate(case['lat']):
            aux_latents = []
            for i, client in enumerate(space):
                baseline_lat = create_latent_df(
                    X_index=base_index,
                    x_lat=client,
                    label=f"{scenario}__{label_clients[i]}__{epoch}",
                    is_unix=True
                )
                aux_latents.append(baseline_lat)

            df_latent = pd.concat(aux_latents)

            df_latent, df_pca_raw, df_umap_raw, df_pca_scaled, df_umap_scaled = process_latent_df(
                df_latent,
                umap_neighbors=50,
                umap_min_dist=0.95
            )

            latent_dfs[scenario][epoch] = {
                'latent_space': df_latent,
                'pca_raw': df_pca_raw,
                'pca_scl': df_pca_scaled,
                'umap_raw': df_umap_raw,
                'umap_scl': df_umap_scaled
            }

    agg_int = int(args.interval_agg / 3600)
    results_id = f'{args.experiment_id}_{args.communication_epoch}_{args.local_epoch}_{agg_int}_{args.window_size}_{args.extra_coments}'
    
    results_path = f"results/results_{results_id}.pkl"
    latent_path = f"results/latent_{results_id}.pkl"

    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    with open(latent_path, 'wb') as f:
        pickle.dump(latent_dfs, f)
    print("Train process finisehd.\nLog files saved in results directory.")

    if args.generate_viz:
        print("Generating visualization...")
        # Add timestamps to latent outputs
        format_latent_dict(latent_dfs)
        
        # Load and normalize time-series
        scaled_df = load_and_scale_data(id_network = 'Graeme', id_experiment = args.experiment_id)
        
        # Combine all latent outputs
        df_combined = combine_latents(latent_dfs)
        
        # Plot heatmap and save
        plot_latent_heatmap(df_combined, results_id)
        
        # Plot time-series and save
        batch_temporal = (agg_int * args.window_size) / 24
        plot_time_series_and_latents(df_combined, scaled_df, results_id, batch_temporal=batch_temporal)
        print("Done. See results in 'results/imgs'")

if __name__ == "__main__":
    main()
