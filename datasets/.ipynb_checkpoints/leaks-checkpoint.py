import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from argparse import Namespace
from typing import Dict

from backbone.AER import AER
from utils.build_aer_model import build_model_from_json_file

from datasets.utils.federated_dataset import FederatedDataset
from simulation.utils.configs import get_domain_sensor_map

from utils.timeseries_preprocessing import time_segments_aggregate, rolling_window_sequences, slice_array_by_dims


# TODO AQUI É O MELHOR LUGAR MESMO?
def initizalize_backbone(args: Namespace,
                         config_json='backbone/AER.json',
                         custom_parameters=None,
                         ):
    if custom_parameters is None:
        custom_parameters = {
            'epochs': args.local_epoch,
            'verbose': True,
            'window_size': args.window_size,
            'input_shape': [args.window_size, args.input_size],
            'target_shape': [args.window_size, args.input_size],
            'lstm_units': args.lstm_units,
            'repeat_vector_n': args.window_size,
            'n_outputs': args.input_size,
            'agg_interval': int(args.interval_agg / (60**2)),
            # 'initialize_model' : True,
        }

    model = build_model_from_json_file(json_path=config_json,
                                       custom_args=custom_parameters,
                                       dense_layer=args.output_size)

    return model


class WaterLeakClientDatasetBuilder:
    def __init__(self, id_network: str, id_exp: str, root_dir: str = "data_leaks"):
        self.id_network = id_network
        self.id_exp = id_exp
        self.root_dir = root_dir
        self.data_clients: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.leaks_scenarios = {}

        self.pressure_map, self.flow_map, _, _ = get_domain_sensor_map(self.id_network)

        self.data_Pressure = None
        self.data_Flow = None
        self.data_scenarios = None

    def load_data(self):
        self.data_Pressure = pd.read_csv(f"{self.root_dir}/{self.id_network}/{self.id_exp}/Pressure.csv")
        self.data_Flow = pd.read_csv(f"{self.root_dir}/{self.id_network}/{self.id_exp}/Flow.csv")
        self.data_scenarios = pd.read_csv(f"{self.root_dir}/{self.id_network}/{self.id_exp}/scenarios.csv")

    def get_labels(self):
        return self.leaks_scenarios

    def build_and_save(self, save_dir="datasets/leaks"):
        if self.data_Pressure is None or self.data_Flow is None or self.data_scenarios is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.data_clients = {client: {} for client in self.pressure_map}

        for scenario in self.data_Pressure['scenario'].unique():
            aux_scenario = self.data_scenarios[self.data_scenarios['scenario'] == scenario]

            # Create labels of the anomalies
            timestamps = self.data_Pressure[self.data_Pressure['scenario'] == scenario]['timestamp'].values
            labels = np.zeros_like(timestamps, dtype=int)
            for _, row in aux_scenario.iterrows():
                start = row['start']
                end = row['end']
                labels[(timestamps >= start) & (timestamps <= end)] = 1

            df_labels = pd.DataFrame({
                'timestamp': timestamps,
                'label': labels
            })

            # Store it for the scenario
            self.leaks_scenarios[scenario] = df_labels

            for client, press_nodes in self.pressure_map.items():
                nodes_press = ['timestamp'] + press_nodes
                aux_press = self.data_Pressure[self.data_Pressure['scenario'] == scenario][nodes_press]

                nodes_flow = ['timestamp'] + self.flow_map[client]
                aux_flow = self.data_Flow[self.data_Flow['scenario'] == scenario][nodes_flow]

                merged_df = pd.merge(aux_press.reset_index(drop=True),
                                     aux_flow.reset_index(drop=True),
                                     how='left', on='timestamp')

                # merged_df['Leakages'] = labels
                self.data_clients[client][scenario] = merged_df

                # Ensure output directory exists
                client_save_dir = os.path.join(save_dir, self.id_network, self.id_exp)
                os.makedirs(client_save_dir, exist_ok=True)

                # Save to CSV
                file_name = f"{client.replace('_', '')}_{scenario}.csv"
                merged_df.to_csv(os.path.join(client_save_dir, file_name), index=False)

        self.scenarios_list = self.data_Pressure['scenario'].unique().tolist()

    def get_data(self):
        return self.data_clients


class FedLeaksGeo(FederatedDataset):
    NAME = 'fl_leaks'
    SETTING = 'domain_skew'

    def __init__(self, args):
        super().__init__(args)
        self.scenario = None
        self.leaks_data = {}
        self.leaks_labels = {}
        self.scaler = None

        self.CLIENTS_DICT = args.domains
        self.DOMAINS_LIST = list(self.CLIENTS_DICT.keys())
        self.EXP_ID = [args.experiment_id for _ in self.DOMAINS_LIST] if isinstance(args.experiment_id, str) else args.experiment_id
        self.MAP_CLIENTS = [name for name, count in self.CLIENTS_DICT.items() for _ in range(count)]
        self.EXPERIMENT_IDS = {D: E for D, E in zip(self.DOMAINS_LIST, self.EXP_ID)}



    def get_data_loaders(self, selected_domain_list=[], scenario='Baseline'):

        using_list = self.DOMAINS_LIST if len(selected_domain_list) == 0 else selected_domain_list
        self.scenario = scenario

        # TODO não acho que lista com dominios seja o melhor para os dloaders
        # Load the raw data and save individual files for each domain/client/scenario
        for _, domain in enumerate(using_list):
            builder = WaterLeakClientDatasetBuilder(id_network=domain, id_exp=self.EXPERIMENT_IDS[domain])
            builder.load_data()
            builder.build_and_save()
            self.leaks_data[domain] = builder.get_data()  # for each domain holds a dict with clients/scenarios
            self.leaks_labels[domain] = builder.get_labels()

        traindls = []
        for domain in self.leaks_data.values():
            train_df = [
                self.preprocess_series(series=client_dict[self.scenario])
                for client_dict in domain.values()
                # if scenario in client_dict
            ]

            traindls.append(train_df)

        return [client for domain in traindls for client in domain]  # , testdls, federation#, train_dataset_list, test_dataset_list

    # @staticmethod
    def preprocess_series(self, series):
        """
        Full preprocessing pipeline: aggregation, normalization, rolling window.

        Args:
            series (pd.DataFrame): Raw time series with a 'timestamp' column.

        Returns:
            dict: A dictionary containing preprocessed data with keys:
                  'X', 'y', 'X_index', 'y_index', etc.
        """
        context = {}

        target_columns = [i for i in range(series.shape[1] - 1)]

        # Step 0: Aggregate time series
        aggregation_transform = self.get_aggregation_transform(interval=self.args.interval_agg)
        X_agg, idx_agg = aggregation_transform(series)
        context['X_raw'] = X_agg
        # context['index_raw'] = idx_agg

        # Step 1: Normalize
        normalize_transform = self.get_normalization_transform()
        X_norm = normalize_transform(X_agg)
        context['X'] = X_norm
        context['X_norm'] = X_norm
        context['index'] = idx_agg

        # Step 2: Rolling window
        rolling_transform = self.get_rolling_window_transform(window_size=self.args.window_size)
        X_seq, y_seq, X_idx, y_idx = rolling_transform(X_norm, idx_agg, target_columns)

        # Step 3: Slice targets from sequence
        y_seq = slice_array_by_dims(X_seq, target_index=target_columns, axis=2)

        # Store all results in context
        context['X'] = X_seq
        context['y'] = y_seq
        context['X_index'] = X_idx
        context['y_index'] = y_idx

        return context

    def get_labels(self):
        return self.leaks_labels

    # @staticmethod
    # TODO CORRIGIR PARAMETROS! ADICIRONAR CUSTOM DICT AQUI!
    def get_backbone(self, parti_num, names_list, n_series):
        nets_dict = {'aer': AER}
        nets_list = []
        if names_list is None:
            for j in range(parti_num):
                # TODO VERIFICAR CASOS EM QUE n_series É DIFERENTE!!
                net = initizalize_backbone(args=self.args)
                nets_list.append(net)
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name]())
        return nets_list

    @staticmethod
    def get_aggregation_transform(interval=3 * 60 ** 2, method='mean', time_column='timestamp'):
        """
        Returns a transform function that aggregates time series data.
        """

        def transform(series):
            X, index = time_segments_aggregate(
                X=series,
                interval=interval,
                time_column=time_column,
                method=method
            )
            return X, index

        return transform

    # @staticmethod
    def get_normalization_transform(self):
        """
        Fits and applies MinMaxScaler, returns the scaled data.
        """

        def normalize(X):
            self.scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            return self.scaler.fit_transform(X)

        return normalize

    def get_denormalization_transform(self):
        """
        Returns a function that reverses MinMaxScaler transform using the instance's scaler.
        """

        def denormalize(X):
            if not hasattr(self, 'scaler'):
                raise ValueError("Scaler not found. You must call the normalization function first.")
            return self.scaler.inverse_transform(X)

        return denormalize

    @staticmethod
    def get_rolling_window_transform(window_size=100, target_size=1, step_size=1, offset=0):
        """
        Returns a transform function that applies rolling windows.
        """

        def transform(X, index, target_columns):
            X_seq, y_seq, X_idx, y_idx = rolling_window_sequences(
                X=X,
                index=index,
                window_size=window_size,
                target_size=target_size,
                step_size=step_size,
                target_column=target_columns,
                offset=offset,
                drop=None,
                drop_windows=False
            )

            y_seq = slice_array_by_dims(X=X_seq, target_index=target_columns, axis=2)
            return X_seq, y_seq, X_idx, y_idx

        return transform