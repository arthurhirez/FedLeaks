import os
import argparse

from utils.Tbx_Pipeline import load_assign_network, run_scenarios, compile_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run water network simulation with specified parameters.")

    # Identification
    parser.add_argument('--id_network', type=str, default='Balerma', help='Identifier for the water network model.')
    parser.add_argument('--id_exp', type=str, default='PIPELINE_PYTHON', help='Experiment identifier.')

    # Features drift
    parser.add_argument('--tgt_district', type=str, default='District_A', help='Target district name.')
    parser.add_argument('--seed_node', type=str, default='415', help='Seed node for simulation.')  # 415
    parser.add_argument('--income_density_mapping', type=dict,
                        default=
                        # [('low', 'low'),
                        #           ('low', 'low'),
                        #           ('low', 'high'),
                        #           ('medium', 'medium'),
                        #           ('high', 'low')],
                        [('low', 'medium'),
                         ('low', 'low'),
                         ('low', 'low')
                         ],
                        help='List of income:density category mappings. Format: income:density')

    # Time variables
    parser.add_argument('--n_segments', type=int, default=7, help='Number of urban growth simulation segments.')
    parser.add_argument('--epochs_lenght', type=int, default=6, help='Number of epochs per simulation run.')
    parser.add_argument('--days_lenght', type=int, default=5, help='Number of days in each epoch.')
    parser.add_argument('--n_intervals', type=int, default=24, help='Number of time intervals per day.')

    # Experiment value
    parser.add_argument('--value_exp', type=float, default=0.15, help='Leak severity multiplier.')

    # Hydraulic variables
    parser.add_argument('--unit', type=str, default='l/s', help='Measurement unit for flow.')
    parser.add_argument('--morning_peak', type=float, default=1.0, help='Multiplier for morning consumption peak.')
    parser.add_argument('--afternoon_peak', type=float, default=0.75, help='Multiplier for afternoon consumption peak.')
    parser.add_argument('--evening_peak', type=float, default=1.25, help='Multiplier for evening consumption peak.')
    parser.add_argument('--night_consumption', type=float, default=0.25, help='Multiplier for night-time consumption.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up environment
    scripts_path = r"C:\Users\arthu\anaconda3\envs\TsLeaks\Scripts"
    os.environ["PATH"] = scripts_path + os.pathsep + os.environ["PATH"]

    # Load network and assign buildings
    wn, consumption_patterns, data_consumption = load_assign_network(
        directory='networks\\original\\',
        save_assignments=True,
        verbose=True,
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


if __name__ == "__main__":
    main()
