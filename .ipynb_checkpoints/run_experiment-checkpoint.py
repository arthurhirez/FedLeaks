import pickle
import subprocess
import sys
import warnings
from argparse import Namespace
import os
from datasets import get_private_dataset
from models import get_model
from utils.Server import train

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.args import create_experiment_args, add_federated_args
from utils.conf import set_random_seed

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

    set_random_seed(args.seed)
    
    # If run_simulation is True, execute the Pipeline_Simulation script
    if args.run_simulation:
        subprocess.run([
            sys.executable,
            'Pipeline_Simulation.py',
            '--experiment_id', args.experiment_id,
            '--generate_viz', str(args.generate_viz),
            '--tgt_district', args.tgt_district,
            '--seed_node', args.seed_node,
            '--drift_income', args.drift_income,
            '--drift_density', args.drift_density
        ], check=True, cwd='simulation')

    if args.federated_training:
        results = {}
    
        for scenario in ['Baseline']:  # , 'AutoScenario_1']:
            results[scenario] = {}
    
            priv_dataset = get_private_dataset(args)
    
            backbones_list = priv_dataset.get_backbone(
                parti_num=args.parti_num,
                names_list=None,
                n_series=args.input_size
            )
    
            model = get_model(backbones_list, args, priv_dataset)
    
            train(model=model,
                  private_dataset=priv_dataset,
                  scenario=scenario,
                  args=args
                  )
    
            results[scenario]['model'] = model.compile_final_results()
            results[scenario]['args'] = args
    
        agg_int = int(args.interval_agg / 3600)
        results_id = f'{args.experiment_id}_{args.communication_epoch}_{args.local_epoch}_{agg_int}_{args.window_size}_{args.extra_coments}'
        
        results_dir = f"results/{results_id}"
        os.makedirs(results_dir, exist_ok=True)
    
        results_path = f"{results_dir}/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
    
        print("Train process finished.\nLog files saved in results directory.")

    if args.process_latents:
        cmd = [
            sys.executable,
            'Process_Results.py',
            '--experiment_id', str(args.experiment_id),
            '--extra_coments', str(args.extra_coments),
            '--interval_agg', str(args.interval_agg),
            '--communication_epoch', str(args.communication_epoch),
            '--local_epoch', str(args.local_epoch),
            '--window_size', str(args.window_size),
            '--parti_num', str(args.parti_num),
            '--input_size', str(args.input_size),
            '--tgt_district', str(args.tgt_district),
            '--generate_viz', str(args.generate_viz),
        ]
        print("Running command:", cmd)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
