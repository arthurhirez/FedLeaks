import pandas as pd
import numpy as np
from numpy import ndarray

import torch
import torch.nn as nn
import torch.optim as optim

import importlib
from sklearn.preprocessing import MinMaxScaler
from utils.timeseries_errors import reconstruction_errors, regression_errors


# -------- CUSTOM SUPPORT LAYERS --------
class RepeatVector(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        if x.ndim == 3:  # Check if the input is 3D
            return x.repeat(1, self.n, 1)  # Repeat along the time axis
        else:
            return x.unsqueeze(1).repeat(1, self.n, 1)  # Fallback for 2D


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


# -------- AutoEncoder with bidirectional LSTM --------
class AER(nn.Module):
    """Autoencoder with bi-directional regression for time series anomaly detection.

    Args:
        layers_encoder (list):
            List containing layers of encoder.
        layers_generator (list):
            List containing layers of generator.
        optimizer (str):
            String denoting the keras optimizer.
        input_shape (tuple):
            Optional. Tuple denoting the shape of an input sample.
        learning_rate (float):
            Optional. Float denoting the learning rate of the optimizer. Default 0.005.
        epochs (int):
            Optional. Integer denoting the number of epochs. Default 2000.
        batch_size (int):
            Integer denoting the batch size. Default 64.
        hyperparameters (dictionary):
            Optional. Dictionary containing any additional inputs.
    """
    def __init__(self, layers_encoder: list, layers_decoder: list,
                 optimizer: str, learning_rate: float = 0.001,
                 epochs: int = 35, batch_size: int = 64, shuffle: bool = True,
                 verbose: bool = True, callbacks: tuple = tuple(), reg_ratio: float = 0.5,
                 input_shape: tuple = tuple(), lstm_units: int = 30, validation_split: float = 0.0,
                 id_exp: str = '001', id_comm_epoch: str = 'VERIFY',  agg_interval: int = 2,
                 load_trained_model: bool = False, load_path: str = None,
                 **hyperparameters):
        super().__init__()

        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.input_shape = input_shape  # (seq_len, n_features)
        self.seq_len, self.input_dim = input_shape
        self.agg_interval = agg_interval

        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.reg_ratio = reg_ratio
        self.validation_split = validation_split
        self.lstm_units = lstm_units
        self.hyperparameters = hyperparameters

        # Initialize encoder and decoder
        self.encoder, enc_output_dim = self._build_stack(self.layers_encoder, self.input_dim)
        self.decoder, _ = self._build_stack(self.layers_decoder, enc_output_dim)

        # Initialize optimizer
        self.optimizer_class = get_class_from_str(optimizer)
        self.learning_rate = learning_rate
        # self.optimizer = None

        # Callbacks (store only the structure, not the real tf objects)
        self.callbacks = callbacks

        # ID and logging
        self.id_exp = id_exp
        self.id_comm_epoch = id_comm_epoch
        self.total_local_epoch = 0
        self.path = f"saved_model/aer_{self.id_exp}/{self.id_comm_epoch}"
        self.loaded_model = False
        self._fitted = False
        self.fit_history = []

        if load_trained_model and load_path:
            self._load_model_internal(load_path)

    def _build_stack(self, layers_config, input_dim):
        # Use your build_stack function here
        return build_stack(layers_config, self.hyperparameters, input_dim)

    def prepare_data(self, X: np.ndarray, X_index: np.ndarray, y: np.ndarray = None):
        """
        Prepares PyTorch DataLoader from input data, including hour-based label.
        """
        if y is None:
            y = X.copy()

        # Extract 12-hour format label from timestamps
        timestamps = pd.to_datetime(X_index, unit='s')
        # hour = timestamps.hour
        # label = hour.where(hour < 12, hour - 12).to_numpy()
        # label = label / self.agg_interval
        month = timestamps.month - 1
        label = month.to_numpy()
        label_tensor = torch.tensor(label, dtype=torch.int64)  # or float32 if needed

        # Prepare input and targets
        X = X[:, 1:-1, :]
        ry, y, fy = y[:, 0], y[:, 1:-1], y[:, -1]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        ry_tensor = torch.tensor(ry, dtype=torch.float32).unsqueeze(-1)
        fy_tensor = torch.tensor(fy, dtype=torch.float32).unsqueeze(-1)

        # Add label tensor to dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, ry_tensor, y_tensor, fy_tensor, label_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return dataloader

    def initialize_optimizer(self):
        if not hasattr(self, "optimizer"):
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

    def _augment_hyperparameters(self, X, y, kwargs):
        X = torch.as_tensor(X)
        self.input_shape = X.shape[1:]  # (seq_len, n_features)
        self.latent_shape = (self.lstm_units * 2,)

        kwargs['repeat_vector_n'] = self.input_shape[0] + 2
        kwargs['lstm_units'] = int(self.lstm_units)
        return kwargs

    def _build_aer(self, **kwargs):
        hyperparameters = self.hyperparameters.copy()
        hyperparameters.update(kwargs)

        # Build encoder and decoder
        self.encoder, enc_output_dim = self._build_stack(self.layers_encoder, self.input_shape[-1])
        self.decoder, _ = self._build_stack(self.layers_decoder, enc_output_dim)


    def forward(self, x):
        # Encode input sequence
        _, (h_n, _) = self.encoder(x)  # h_n shape: (num_directions * num_layers, batch, hidden_size)

        # Flatten hidden states (handle bidirectional)
        batch_size = h_n.shape[1]
        hidden = h_n.transpose(0, 1).reshape(batch_size, -1)  # shape: (batch, hidden_dim * num_directions)

        # Decode
        repeated = self.decoder[0](hidden)                   # RepeatVector
        seq_out, (x_lat, _) = self.decoder[1](repeated)      # LSTM
        decoded = self.decoder[2](seq_out)                   # TimeDistributed(Dense)

        # Slice output
        ry = decoded[:, 0]                                   # First timestep (backward prediction)
        y = decoded[:, 1:-1]                                 # Middle (reconstruction)
        fy = decoded[:, -1]                                  # Last timestep (forward prediction)

        # Ensure reconstruction shape matches input
        y = y.reshape(batch_size, x.shape[1], -1)
        latent = x_lat.detach().numpy()
        latent = latent.reshape(x_lat.shape[1], -1)
        latent = x_lat.reshape(x_lat.shape[1], -1)

        return ry, y, fy, latent

    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        Fit the model using PyTorch.

        Args:
            X (ndarray): Input sequences.
            y (ndarray): Optional target sequences for reconstruction.
        """
        device = "cpu"

        if y is None:
            y = X.copy()

        X = X[:, 1:-1, :]
        ry, y, fy = y[:, 0], y[:, 1:-1], y[:, -1]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        ry_tensor = torch.tensor(ry, dtype=torch.float32).unsqueeze(-1).to(device)
        fy_tensor = torch.tensor(fy, dtype=torch.float32).unsqueeze(-1).to(device)

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._build_aer(**kwargs)
            self.to(device)
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
            # optimizer = self.optimizer

        dataset = torch.utils.data.TensorDataset(X_tensor, ry_tensor, y_tensor, fy_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 shuffle=self.shuffle)

        criterion = nn.MSELoss()

        self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for xb, ryb, yb, fyb in dataloader:
                self.optimizer.zero_grad()
                out_ry, out_y, out_fy = self.forward(xb)
                # print(f"ryb shape: {ryb.shape}, yb shape: {yb.shape}, fyb shape: {fyb.shape}")
                loss_ry = criterion(out_ry, torch.squeeze(ryb))
                loss_y = criterion(out_y, yb)
                loss_fy = criterion(out_fy, torch.squeeze(fyb))

                loss = (self.reg_ratio / 2) * loss_ry + (1 - self.reg_ratio) * loss_y + (self.reg_ratio / 2) * loss_fy
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

            self.fit_history.append(epoch_loss)

        self.total_local_epoch += self.epochs
        self._fitted = True
        print(f'epochs: {self.epochs} / total {self.total_local_epoch}')

    # @torch.no_grad()
    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict values using the initialized PyTorch model.

        Args:
            X (ndarray): Input sequences.

        Returns:
            Tuple: (ry, y, fy, latent_representation)
        """
        self.eval()
        device = "cpu"

        with torch.no_grad():

            X = X[:, 1:-1, :]
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            # Encode input sequence
            _, (h_n, _) = self.encoder(X_tensor)

            # Flatten hidden states (handle bidirectional)
            batch_size = h_n.shape[1]
            hidden = h_n.transpose(0, 1).reshape(batch_size, -1)  # shape: (batch, hidden_dim * num_directions)

            # Decode
            repeated = self.decoder[0](hidden)  # RepeatVector
            seq_out, (x_lat, _) = self.decoder[1](repeated)  # LSTM
            decoded = self.decoder[2](seq_out)  # TimeDistributed(Dense)

            # Slices
            ry = decoded[:,  0]
            y = decoded[:,  1:-1].reshape(X_tensor.shape[0], X_tensor.shape[1], -1)
            fy = decoded[:,  -1]
            latent = x_lat.cpu().numpy()
            latent = latent.reshape(x_lat.shape[1], -1)

        return ry, y, fy, latent


    def compute_errors(self, X, ry_hat, y_hat, fy_hat):


        # Convert to numpy if they're PyTorch tensors
        if hasattr(X, 'detach'):
            X = X.detach().cpu().numpy()
        if hasattr(ry_hat, 'detach'):
            ry_hat = ry_hat.detach().cpu().numpy()
        if hasattr(y_hat, 'detach'):
            y_hat = y_hat.detach().cpu().numpy()
        if hasattr(fy_hat, 'detach'):
            fy_hat = fy_hat.detach().cpu().numpy()

        input_window = self.input_shape[0]
        aux_errors = []

        for channel in range(X.shape[2]):
            errors = score_anomalies(
                y = X[:, :, channel].reshape(-1, input_window, 1),
                ry_hat = ry_hat[:, channel].reshape(-1, 1),
                y_hat = y_hat[:, :, channel].reshape(-1, input_window - 2, 1),
                fy_hat = fy_hat[:, channel].reshape(-1, 1),
                smoothing_window = 0.01,
                smooth = True,
                mask = True,
                comb = 'mult',
                lambda_rec = 0.5,
                rec_error_type = "dtw"
            )
            aux_errors.append(errors)

        return aux_errors

    def create_dataloader(self, X: np.ndarray, y: np.ndarray = None, **kwargs):

        device = "cpu"

        if y is None:
            y = X.copy()

        X = X[:, 1:-1, :]
        ry, y, fy = y[:, 0], y[:, 1:-1], y[:, -1]

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        ry_tensor = torch.tensor(ry, dtype=torch.float32).unsqueeze(-1).to(device)
        fy_tensor = torch.tensor(fy, dtype=torch.float32).unsqueeze(-1).to(device)

        if not self._fitted:
            self._augment_hyperparameters(X, y, kwargs)
            self._build_aer(**kwargs)
            self.to(device)
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X_tensor, ry_tensor, y_tensor, fy_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 shuffle=self.shuffle)

        return dataloader

# -------- AUXILIARY FUNCTIONS -------- #
 # -------- CREATE MODEL --------
def get_layer_class(class_path: str):
    mapping = {
        # Short names
        "LSTM": nn.LSTM,
        "Bidirectional": "Bidirectional",
        "Dense": nn.Linear,
        "Linear": nn.Linear,
        "RepeatVector": "RepeatVector",
        "TimeDistributed": "TimeDistributed",
    }

    if class_path not in mapping:
        raise ValueError(f"Unsupported layer class: {class_path}")
    return mapping[class_path]



def build_layer(layer_dict, hyperparameters, encoder, current_input_dim=None):
    layer_type = get_layer_class(layer_dict['class'])
    params = layer_dict.get('parameters', {}).copy()

    # Resolve hyperparameter references
    for key, value in params.items():
        if isinstance(value, str):
            params[key] = hyperparameters.get(value, value)

    if layer_type == nn.LSTM:
        if current_input_dim is not None:
            params['input_size'] = current_input_dim
        return nn.LSTM(**params, batch_first=True)

    elif layer_type == nn.Linear:
        if 'units' in params:
            params['out_features'] = params.pop('units')
        if current_input_dim is not None:
            params['in_features'] = current_input_dim
        return nn.Linear(**params)

    elif layer_type == "Bidirectional":
        inner_layer_dict = params['layer']
        units = inner_layer_dict['parameters']['hidden_size']
        lstm_layer = nn.LSTM(input_size=current_input_dim,
                             hidden_size=units,
                             batch_first=True,
                             bidirectional=True)
        return lstm_layer

    elif layer_type == "RepeatVector":
        return RepeatVector(params['n'])

    elif layer_type == "TimeDistributed":
        # inner_layer = build_layer(params['layer'], hyperparameters, current_input_dim)
        inner_layer_dict = params['layer']
        out_features = inner_layer_dict['parameters']['out_features']
        inner_layer = nn.Linear(in_features=current_input_dim, out_features=out_features)
        return TimeDistributed(inner_layer)

    raise NotImplementedError(f"Unknown layer type or unhandled layer class: {layer_type}")


def get_class_from_str(full_class_string):
    """Import a class from a full class path string (e.g. 'torch.optim.Adam')."""
    module_name, class_name = full_class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def build_stack(layers, hyperparameters, input_dim):
    modules = []
    current_dim = input_dim
    encoder = True
    for layer_dict in layers:
        layer = build_layer(layer_dict, hyperparameters, encoder, current_input_dim=current_dim)

        # Update current_dim (very basic inference; refine as needed)
        if isinstance(layer, nn.LSTM):
            hidden = layer.hidden_size
            current_dim = hidden * 2 if layer.bidirectional else hidden
            encoder = False
        elif isinstance(layer, nn.Linear):
            current_dim = layer.out_features
        elif isinstance(layer, RepeatVector):
            # current_dim = hyperparameters['window_size']
            pass
        elif isinstance(layer, TimeDistributed):
            current_dim = layer.module.out_features

        modules.append(layer)

    return nn.Sequential(*modules), current_dim


 # -------- ANOMALY DETECTION --------
def bi_regression_errors(y: ndarray, ry_hat: ndarray, fy_hat: ndarray,
                         smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True):
    """Compute an array of absolute errors comparing the forward and reverse predictions with
    the expected output.

    Anomaly scores are created in the forward and reverse directions. Scores in overlapping indices
    are averaged while scores in non-overlapping indices are taken directly from either forward or
    reverse anomaly scores.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.

    Returns:
        ndarray:
            Array of errors.
    """
    time_steps = len(y[0]) - 1
    mask_steps = int(smoothing_window * len(y)) if mask else 0
    ry, fy = y[:, 0], y[:, -1]

    f_scores = regression_errors(fy, fy_hat, smoothing_window=smoothing_window, smooth=smooth)
    f_scores[:mask_steps] = 0
    f_scores = np.concatenate([np.zeros(time_steps), f_scores])

    r_scores = regression_errors(ry, ry_hat, smoothing_window=smoothing_window, smooth=smooth)
    r_scores[:mask_steps] = min(r_scores)
    r_scores = np.concatenate([r_scores, np.zeros(time_steps)])

    scores = f_scores + r_scores
    scores[time_steps + mask_steps:-time_steps] /= 2
    return scores


def score_anomalies(y: ndarray, ry_hat: ndarray, y_hat: ndarray, fy_hat: ndarray,
                    smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True,
                    comb: str = 'mult', lambda_rec: float = 0.5, rec_error_type: str = "dtw"):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'dtw' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of errors.
    """

    reg_scores = bi_regression_errors(y, ry_hat, fy_hat,
                                      smoothing_window=smoothing_window,
                                      smooth=smooth,
                                      mask=mask
                                      )
    rec_scores, _ = reconstruction_errors(y[:, 1:-1], y_hat,
                                          smoothing_window=smoothing_window,
                                          smooth=smooth,
                                          rec_error_type=rec_error_type)
    mask_steps = int(smoothing_window * len(y)) if mask else 0
    rec_scores[:mask_steps] = min(rec_scores)
    rec_scores = np.concatenate([np.zeros(1), rec_scores, np.zeros(1)])

    scores = None
    if comb == "mult":
        feature_range = (1, 2)
        reg_scores = MinMaxScaler(feature_range).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler(feature_range).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = np.multiply(reg_scores, rec_scores)

    elif comb == "sum":
        feature_range = (0, 1)
        reg_scores = MinMaxScaler(feature_range).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler(feature_range).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = (1 - lambda_rec) * reg_scores + lambda_rec * rec_scores

    elif comb == "rec":
        scores = rec_scores

    elif comb == "reg":
        scores = reg_scores

    return scores