import os
import pickle
import warnings

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datasets import get_private_dataset
from utils.Server import local_evaluate
from utils.Toolbox_analysis import create_latent_df, process_latent_df
from utils.Toolbox_postprocessing import proto_analysis, distributions_analysis
from utils.Toolbox_visualization import load_and_scale_data, combine_latents, plot_latent_heatmap, plot_time_series_and_latents
from utils.Toolbox_visualization import plot_proto_similar, plot_distribution_similar

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.args import create_experiment_args, add_federated_args
from argparse import Namespace

def get_parser() -> Namespace:
    parser = create_experiment_args()
    add_federated_args(parser)
    return parser.parse_args()


def main():
    args = get_parser()

    # Process income_density_mapping
    parts = args.income_density_mapping.split('_')
    level_map = {'H': 'high', 'M': 'medium', 'L': 'low'}  # make sure level_map exists
    args.income_density_mapping = [(level_map[p[0]], level_map[p[1]]) for p in parts]

    # Compute parti_num if needed
    if args.parti_num is None:
        args.parti_num = sum(args.domains.values())

    
    agg_int = int(args.interval_agg / 3600)
    results_id = f'{args.experiment_id}_{args.communication_epoch}_{args.local_epoch}_{agg_int}_{args.window_size}_{args.extra_coments}'
    results_dir = f"results/{results_id}"
    
    results_path = f"{results_dir}/results.pkl"
    with open(results_path, 'rb') as f:
        results = pickle.load(f)


    args = results['Baseline']['args']
    label_clients = [
        'District_A', 'District_B', 'District_C', 'District_D', 'District_E',
        'District_2A', 'District_2B', 'District_2C'
    ]
    
    priv_dataset = get_private_dataset(args)
    backbones_list = priv_dataset.get_backbone(
        parti_num=args.parti_num,
        names_list=None,
        n_series=args.input_size
    )
    
    train_DL = priv_dataset.get_data_loaders()
    base_index = train_DL[0]['X_index']
    latent_dfs = {}
    scenarios = ['Baseline']
            

    all_latents = defaultdict(list)
    all_protos = defaultdict(list)
    all_dists = defaultdict(list)
    all_pca = defaultdict(list)
    all_umap = defaultdict(list)
    
    for scenario in scenarios:
        global_model_history = results[scenario]['model']['global_weights_history']
        
        iterator = tqdm(range(args.communication_epoch))
        for epoch in iterator:
            aux_latents = []
            state_dict = global_model_history[epoch]
            for net in backbones_list:
                net.load_state_dict(state_dict)
    
            latent_spaces = local_evaluate(model=backbones_list,
                                           train_dl=train_DL,
                                           private_dataset=priv_dataset,
                                           group_detections=False,
                                           detect_anomalies=False)
    
            for i, client in enumerate(latent_spaces):
                client_lat = create_latent_df(
                    X_index=base_index,
                    x_lat=client,
                    label=f"{label_clients[i]}__{epoch}",
                    is_unix=True
                )
                aux_latents.append(client_lat)
    
            data_latent = pd.concat(aux_latents)
            data_latent[['client_id', 'epoch']] = data_latent['label'].str.split('__', expand=True)
            data_latent['label'] = data_latent['timestamp'].dt.month
    
            id_cols = ['client_id', 'label', 'epoch']
            feat_cols = [col for col in data_latent.columns if 'x_' in col]
            aux_agg = data_latent[id_cols + feat_cols]
            aux_agg = aux_agg.groupby(id_cols).mean().reset_index()
            aux_agg = aux_agg.sort_values(by=['epoch', 'label', 'client_id']).reset_index(drop=True)
            aux_agg['client_id'] += '__proto'
    
            tag = f"{scenario}_{epoch}"
            df_exp_proto = proto_analysis(data_latent=aux_agg, normalize=False)
            aux_df = df_exp_proto.copy()
            aux_df.columns = ['epoch', 'label', 'client_2', 'client_1', 'cosine', 'manhattan', 'wavelet', 'dft', 'autocorr']
            
            df_exp_proto = pd.concat([df_exp_proto, aux_df[df_exp_proto.columns.tolist()]])
            df_exp_proto = df_exp_proto.sort_values(by=['epoch', 'label', 'client_1', 'client_2']).reset_index(drop=True)
            # df_exp_proto['id'] = tag
            
            # df_exp_latent = distributions_analysis(data_distribution=data_latent[id_cols + feat_cols],
            #                                        target_client=args.tgt_district, epoch = epoch, normalize=True)
            # df_exp_latent['id'] = tag
    
            merged_latent = pd.concat([data_latent[['timestamp'] + id_cols + feat_cols], aux_agg])
            merged_latent = merged_latent.reset_index(drop=True)
    
            if args.generate_viz:
                df_latent, df_pca_scaled, df_umap_scaled = process_latent_df(
                    df_latent=merged_latent,
                    umap_neighbors=35,
                    umap_min_dist=0.85,
                    reduce_raw=False,
                    id_cols=['timestamp'] + id_cols,
                    return_scaled=False
                )
                # df_latent['id'] = tag
                df_pca_scaled['method'] = 'pca'
                df_umap_scaled['method'] = 'umap'
    
                all_latents[scenario].append(df_latent)
                all_pca[scenario].append(df_pca_scaled)
                all_umap[scenario].append(df_umap_scaled)
            else:
                # merged_latent['id'] = tag
                all_latents[scenario].append(merged_latent)
    
            all_protos[scenario].append(df_exp_proto)
            # all_dists[scenario].append(df_exp_latent)

    
    for scenario in scenarios:
        pd.concat(all_latents[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_latent_space.parquet", index=False)
        pd.concat(all_protos[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_proto.parquet", index=False)
        # pd.concat(all_dists[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_distribution.parquet", index=False)
    
        if args.generate_viz:
            pd.concat(all_pca[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_pca.parquet", index=False)
            pd.concat(all_umap[scenario], ignore_index=True).to_parquet(f"{results_dir}/{scenario}_umap.parquet", index=False)


    print("Latent processing process finisehd.\nLog files saved in results directory.")
    
    if args.generate_viz:
        print("Generating visualization...")       
        # Add timestamps to latent outputs
        # format_latent_dict(latent_dfs)
        
        # Load and normalize time-series
        scaled_df = load_and_scale_data(id_network = 'Graeme', id_experiment = args.experiment_id, tgt_district = args.tgt_district)
        
        # Combine all latent outputs
        df_combined = combine_latents(results_dir)
        
        # Plot heatmap and save
        plot_latent_heatmap(df_combined[~df_combined['label'].str.contains('proto')], results_dir)
        
        # Plot time-series and save
        batch_temporal = (agg_int * args.window_size) / 24
        plot_time_series_and_latents(df_combined[~df_combined['label'].str.contains('proto')],
                                     scaled_df,  results_dir,
                                     batch_temporal=batch_temporal)

        
        df_exp_proto = pd.read_parquet(f'{results_dir}/Baseline_proto.parquet')
        int_cols = ['epoch', 'label']
        for c in int_cols:
            df_exp_proto[c] = df_exp_proto[c].astype(int)
        plot_proto_similar(df_exp_proto, results_dir)

        df_exp_latent = pd.read_parquet(f'{results_dir}/Baseline_distribution.parquet')
        int_cols = ['epoch', 'label']
        for c in int_cols:
            df_exp_latent[c] = df_exp_latent[c].astype(int)
        plot_distribution_similar(df_exp_latent, results_dir)
        
        print(f"Done. See results in '{results_dir}'")

if __name__ == "__main__":
    main()
