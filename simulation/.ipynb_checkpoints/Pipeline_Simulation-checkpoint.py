import os
import sys
from argparse import Namespace

from utils.Tbx_Pipeline import load_assign_network, run_scenarios, compile_results
from utils.Tbx_Simulation import district_visualization

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.args import create_experiment_args, add_simulation_args


def get_parser() -> Namespace:
    parser = create_experiment_args()
    add_simulation_args(parser)

    return parser.parse_args()

def main():
    args = get_parser()

    # Process income_density_mapping
    parts = args.income_density_mapping.split('_')
    level_map = {'H': 'high', 'M': 'medium', 'L': 'low'}  # make sure level_map exists
    args.income_density_mapping = [(level_map[p[0]], level_map[p[1]]) for p in parts]


    # Set up environment
    scripts_path = r"C:\Users\arthu\anaconda3\envs\TsLeaks\Scripts"
    os.environ["PATH"] = scripts_path + os.pathsep + os.environ["PATH"]

    # Load network and assign buildings
    wn, consumption_patterns, data_consumption = load_assign_network(
        directory='networks\\original\\',
        save_assignments=True,
        verbose=False,
        args=args
    )

    # Run scenarios
    auto_leaks = run_scenarios(
        water_network=wn,
        consumption_patterns=consumption_patterns,
        data_consumption=data_consumption,
        args=args
    )

    # Compile and save results
    compile_results(
        leaks_scenarios=auto_leaks,
        args=args
    )

    if args.generate_viz:
        print("Generating visualization...")
        district_visualization(id_network = args.id_network,
                               id_exp = args.experiment_id,
                               tgt_district = args.tgt_district,
                               save_path='../results/imgs')
        print("Done. See results in 'results/imgs'")

if __name__ == "__main__":
    main()
