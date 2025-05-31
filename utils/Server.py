import pandas as pd
import numpy as np
from argparse import Namespace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets.utils import FederatedDataset
from models.utils.federated_model import FederatedModel
from utils.timeseries_detection import find_anomalies

def train(model: FederatedModel, private_dataset: FederatedDataset, scenario: str,
          args: Namespace) -> None:
    # if args.csv_log:
    #     csv_writer = CsvWriter(args, private_dataset)

    priv_train_loaders = private_dataset.get_data_loaders(scenario=scenario)

    private_train_loaders = []
    private_train_labels = []
    for loader in priv_train_loaders:
        private_train_loaders.append(loader['X'])
        private_train_labels.append(loader['X_index'])

    model.trainloaders = private_train_loaders  # TODO REVER ISSO!!!
    model.trainlabels = loader['X_index']
    
    if hasattr(model, 'ini'):
        model.ini()

    latent_history = []


    iterator = tqdm(range(args.communication_epoch))
    for epoch_index in iterator:
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(priloader_list = private_train_loaders,
                                                   prilabel_list = private_train_labels)

        # print(10 * '**--')
        aux_latent = local_evaluate(model=model,
                                    train_dl=priv_train_loaders,
                                    private_dataset=private_dataset,
                                    group_detections=False,
                                   detect_anomalies = args.detect_anomalies)

        latent_history.append(aux_latent)

        # print(10*'**--')
        # print('COM AGREGAÇÃO')
        # local_evaluate(model = model, train_dl = priv_train_loaders[1], df_results = df_results, group_detections = True)
    # return priv_train_loaders, latent_history



def local_evaluate(model,
                   train_dl: list[dict],
                   private_dataset: FederatedDataset,
                   group_detections: bool = False,
                  detect_anomalies : bool = True) -> list:

    labels_map = private_dataset.get_labels()
    map_clients = private_dataset.MAP_CLIENTS
    aux_latent = []

    # Can be both a list of models or a Federated model
    nets_list = model.nets_list if isinstance(model, FederatedModel) else model

    for client, (net, dl) in enumerate(zip(nets_list, train_dl)):
        dl['ry_hat'], dl['y_hat'], dl['fy_hat'], dl['x_lat'] = net.predict(dl['X'])
        if not detect_anomalies:
            aux_latent.append(dl['x_lat'])
            continue
            
        dl['errors'] = net.compute_errors(dl['X'], dl['ry_hat'], dl['y_hat'], dl['fy_hat'])

        aux_anomaly = []
        for error in dl['errors']:
            detections = find_anomalies(
                errors=error,
                index=dl['index'],
                window_size_portion=0.35,
                window_step_size_portion=0.10,
                fixed_threshold=True,
                inverse=True,
                anomaly_padding=50,
                lower_threshold=True
            )
            if len(detections):
                aux_anomaly.append(detections)

        if len(aux_anomaly) == 0:
            aux_latent.append(dl['x_lat'])
            continue

        df_anomalies = pd.DataFrame(np.vstack(aux_anomaly), columns=['start', 'end', 'severity'])
        if group_detections:
            df_anomalies = group_anomalies(df_anomalies)

        labels_df = labels_map[map_clients[client]][private_dataset.scenario]
        process_anomalies(dl, df_anomalies, labels_df)

        aux_latent.append(dl['x_lat'])

    return aux_latent


# def global_evaluate(model: FederatedModel, test_dl, setting: str, name: str) -> Tuple[list, list]:
#     accs = []
#     net = model.global_net
#     status = net.training
#     net.eval()
#     for j, dl in enumerate(test_dl):
#         correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
#         for batch_idx, (images, labels) in enumerate(dl):
#             with torch.no_grad():
#                 images, labels = images.to(model.device), labels.to(model.device)
#                 outputs = net(images)
#                 _, max5 = torch.topk(outputs, 5, dim=-1)
#                 labels = labels.view(-1, 1)
#                 top1 += (labels == max5[:, 0:1]).sum().item()
#                 top5 += (labels == max5).sum().item()
#                 total += labels.size(0)
#         top1acc = round(100 * top1 / total, 2)
#         top5acc = round(100 * top5 / total, 2)
#         # if name in ['fl_digits','fl_officecaltech']:
#         accs.append(top1acc)
#         # elif name in ['fl_office31','fl_officehome']:
#         #     accs.append(top5acc)
#     net.train(status)
#     return accs


def evaluate_predictions(df, label_col='label', pred_col='y_pred'):
    y_true = df[label_col]
    y_pred = df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Acc: {acc:.4f}\tPrec: {prec:.4f}\tRec: {rec:.4f}\tF1: {f1:.4f}")

    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.show()

    return acc, prec, rec, f1


def process_anomalies(dl, df_anomalies, labels_df):
    labels_df = labels_df.copy()
    labels_df['timestamp'] = labels_df['timestamp'].astype(float)
    y_pred = np.zeros_like(labels_df['timestamp'], dtype=int)

    for _, row in df_anomalies.iterrows():
        mask = (labels_df['timestamp'] >= row['start']) & (labels_df['timestamp'] <= row['end'])
        y_pred[mask] = 1

    labels_df['y_pred'] = y_pred
    dl['results'] = labels_df
    dl['anomalies'] = df_anomalies

    if dl.get('metrics') is None:
        dl['metrics'] = []

    dl['metrics'].append(evaluate_predictions(labels_df))


def group_anomalies(df_anomalies):
    # Ensure start and end are float (or numeric)
    df = df_anomalies.copy()
    df[['start', 'end']] = df[['start', 'end']].astype(float)

    # Step 1: Sort by start
    df = df.sort_values(by='start').reset_index(drop=True)

    # Step 2: Merge overlapping intervals
    merged = []
    current_group = []
    current_start, current_end = df.loc[0, 'start'], df.loc[0, 'end']

    for idx, row in df.iterrows():
        s, e = row['start'], row['end']
        if s <= current_end:
            current_group.append(row)
            current_end = max(current_end, e)
        else:
            # Only keep group if it has 2+ overlaps
            if len(current_group) >= 2:
                # Convert to DataFrame
                group_df = pd.DataFrame(current_group)
                # Choose the smallest duration interval
                group_df['duration'] = group_df['end'] - group_df['start']
                smallest = group_df.loc[group_df['duration'].idxmin()]
                merged.append(smallest[['start', 'end', 'severity']])
            # Start new group
            current_group = [row]
            current_start, current_end = s, e

    # Check last group
    if len(current_group) >= 2:
        group_df = pd.DataFrame(current_group)
        group_df['duration'] = group_df['end'] - group_df['start']
        smallest = group_df.loc[group_df['duration'].idxmin()]
        merged.append(smallest[['start', 'end', 'severity']])

    # Step 3: Final DataFrame
    df_cleaned = pd.DataFrame(merged).reset_index(drop=True)

    return df_cleaned