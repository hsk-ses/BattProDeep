import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.utils import resample
import tensorflow as tf
import tensorflow_probability as tfp

# Importing necessary modules from source and scripts
from src.common.utils import *

# Setup for directories and paths
current_directory = os.getcwd()
sys.path.extend([current_directory, str(Path(current_directory).parent.absolute()),
                 os.path.join(Path(current_directory).parent.absolute(), "scripts")])

tfd = tfp.distributions
tfpl = tfp.layers


def negative_log_likelihood(y_true, y_pred):
    """Calculates the negative log likelihood between true and predicted values."""
    return -y_pred.log_prob(y_true)


class CycleAging:
    """Implements cycle aging simulation based on the Mechanism protocol."""

    def __init__(self, load_trained_models: bool = False, complete_data: bool = False, **kwargs):
        """
        Initializes the cycleAging with a configuration dictionary.

        Args:
            load_trained_models (bool): Whether to load trained models or not.
            complete_data (bool, optional): Whether data is completed or not. The public version of calendar aging
                dataset does not include all cells' data. Defaults to True.
            **kwargs: Additional keyword arguments for model paths.
        """
        # super().__init__(load_trained_models)
        self.degradation_type = "cycle"
        self.complete_data = complete_data
        self._initialize_paths(kwargs)

        if load_trained_models:
            self._load_models()

    def _initialize_paths(self, kwargs: dict):
        """
        Initializes directory paths for models and results.

        Args:
            kwargs (dict): Keyword arguments for directory paths.
        """
        parent_directory = str(Path(__file__).absolute().parent.parent.parent)
        self.models_folder = os.path.join(parent_directory, kwargs.get('models_relative_folder', "trained_models"))
        self.model_path_cycle = os.path.join(self.models_folder,
                                                kwargs.get('cycle_trained_folder', self.degradation_type))
        self.fig_save_path = os.path.join(parent_directory, kwargs.get('results_relative_folder', "results"))
        cycle_results_folder = kwargs.get('cycle_results_folder', self.degradation_type)
        self.fig_save_path_cycle = os.path.join(parent_directory, kwargs.get('cycle_figures_folder',
                                                                                os.path.join(cycle_results_folder,
                                                                                             "figures")))
        self.normalizer_path_cycle = os.path.join(self.model_path_cycle,
                                                     kwargs.get('cycle_normalizers_folder', "normalizer"))
        self.model_folder_virtual = os.path.join(self.model_path_cycle,
                                                 kwargs.get('model_relative_path_virtual', "virtual_fec"))
        self.model_folder_det = os.path.join(self.model_path_cycle,
                                             kwargs.get('model_relative_path_det', "deterministic"))
        aleatoric_model_folder = (
            kwargs.get("aleatoric_model_folder")
            or kwargs.get("cycle_aleatoric_model_folder")
            or kwargs.get("model_folder_prob")
        )
        if aleatoric_model_folder is not None:
            aleatoric_model_folder = Path(aleatoric_model_folder)
            if aleatoric_model_folder.is_absolute():
                self.model_folder = str(aleatoric_model_folder)
            else:
                repo_relative = Path(parent_directory) / aleatoric_model_folder
                self.model_folder = str(
                    repo_relative
                    if repo_relative.exists()
                    else Path(self.model_path_cycle) / aleatoric_model_folder
                )
        else:
            self.model_folder = os.path.join(
                self.model_path_cycle,
                kwargs.get(
                    'model_relative_path',
                    os.path.join('probabilistic', '01-aleatoric'),
                ),
            )
        epistemic_model_folder = (
            kwargs.get("epistemic_model_folder")
            or kwargs.get("cycle_epistemic_model_folder")
            or kwargs.get("model_folder_ep")
        )
        if epistemic_model_folder is not None:
            epistemic_model_folder = Path(epistemic_model_folder)
            if epistemic_model_folder.is_absolute():
                self.model_folder_ep = str(epistemic_model_folder)
            else:
                repo_relative = Path(parent_directory) / epistemic_model_folder
                self.model_folder_ep = str(
                    repo_relative
                    if repo_relative.exists()
                    else Path(self.model_path_cycle) / epistemic_model_folder
                )
        else:
            self.model_folder_ep = os.path.join(
                self.model_path_cycle,
                kwargs.get(
                    'model_relative_path_ep',
                    kwargs.get(
                        'epistemic_model_relative_path',
                        kwargs.get(
                            'model_relative_path',
                            os.path.join('probabilistic', '02-epistemic'),
                        ),
                    ),
                ),
            )

    def _load_models(self):
        """Loads trained models if specified during initialization."""
        self.model_virtual = self.get_trained_model(self.model_folder_virtual)
        self.model_det = self.get_trained_model(self.model_folder_det)
        self.model_prob = self.get_trained_model(self.model_folder,
                                                 custom_objects={'IndependentNormal': tfpl.IndependentNormal,
                                                                 'nll': negative_log_likelihood})

        self.bootstrap_models = self.get_trained_model(self.model_folder_ep,
                                                 custom_objects={'IndependentNormal': tfpl.IndependentNormal,
                                                                  'negative_log_likelihood': negative_log_likelihood},
                                                                  load_epistemic=True)
        self.n_boot = len(self.bootstrap_models)

    def _probabilistic_mean_std(self, x):
        out = self.model_prob(x, training=False)
        if hasattr(out, "stddev"):
            return (
                out.mean().numpy().reshape(-1),
                out.stddev().numpy().reshape(-1),
            )

        raw_model = getattr(self, "_model_prob_raw_output", None)
        if raw_model is None:
            raw_model = tf.keras.Model(
                inputs=self.model_prob.input,
                outputs=self.model_prob.layers[-2].output,
            )
            self._model_prob_raw_output = raw_model

        raw_params = raw_model(x, training=False)
        dist = tfpl.IndependentNormal(event_shape=1)(raw_params)

        return (
            dist.mean().numpy().reshape(-1),
            dist.stddev().numpy().reshape(-1),
        )

    def train(self, df_comb: pd.DataFrame, df_calendar: pd.DataFrame, bootstrap: bool = False, is_virtual: bool = False,
              save_model: bool = True, **kwargs):
        """
        Trains the mechanism's model on provided data.

        Args:
            df_comb (pd.DataFrame): Dataframe containing the cycle and calendar data for training.
            df_calendar (pd.DataFrame): Dataframe containing the calendar data for training.
            bootstrap (bool, optional): Bootstrap the model. Defaults to False.
            is_virtual (bool): Whether the model is virtual or not.
            save_model (bool): Whether to save the trained model or not.
            **kwargs: Additional keyword arguments for training configuration.
        """
        epochs = kwargs.get('epochs', 100)
        learning_rate = kwargs.get('learning_rate', 0.0001)
        loss = kwargs.get('loss', 'mae')
        batch_size = kwargs.get('batch_size', 1)
        verbose = kwargs.get('verbose', 1)
        n_bootstrap = kwargs.get('n_bootstrap', 100)
        file_path = kwargs.get('file_path', None)
        interpolate_rows = kwargs.get('interpolate_rows', True)

        data = {
            'combined': df_comb,
            'calendar': df_calendar
        }

        self.preprocess_data(data, interpolate_rows=interpolate_rows)
        print("Training model with provided data.")
        self.model = self._get_network(is_virtual=is_virtual)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._loss_function = loss if is_virtual else negative_log_likelihood
        self.model.compile(optimizer=optimizer, loss=self._loss_function)

        self.fit_args = {'batch_size': batch_size, 'epochs': epochs, 'verbose': verbose}
        if self.dev_conditions:
            self.fit_args['validation_data'] = (self.X_val, self.Y_val)

        if bootstrap:
            self.bootstrap_models = self.bootstrap_training(n_bootstrap=n_bootstrap)
        else:
            self.model.fit(self.X_train, self.Y_train, **self.fit_args)

        if save_model:
            self._save_model(file_path, is_virtual)

    def bootstrap_training(self, n_bootstrap: int, **kwargs) -> List[Model]:
        """Train multiple models using bootstrap samples of the training data.

        Args:
            n_bootstrap (int): Number of bootstrap samples (models) to train.
            n_epochs (int): Number of epochs to train each model.

        Returns:
            List[Model]: A list containing trained models, each model trained on a bootstrap sample of the training data.
        """
        learning_rate = kwargs.get('learning_rate', 0.0001)
        bootstrap_models: List[Model] = []
        n_points = self.X_train.shape[0]
        n_features = self.X_train.shape[2]

        for _ in range(n_bootstrap):
            # Sample with replacement from the original dataset
            X_boot, y_boot = resample(self.X_train.reshape(n_points, n_features), self.Y_train.reshape(n_points))
            X_boot = X_boot.reshape(n_points, 1, n_features)
            y_boot = y_boot.reshape(n_points, 1)

            self.model = self._get_network(is_virtual=False)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss=self._loss_function)
            self.model.fit(X_boot, y_boot, **self.fit_args)

            # Store the model
            bootstrap_models.append(self.model)

        return bootstrap_models

    def _save_model(self, file_path: str = None, is_virtual: bool = False):
        """
        Saves the trained model to the specified file path.

        Args:
            file_path (str, optional): The path where the model will be saved. Defaults to None.
            is_virtual (bool, optional): Whether the model is virtual or not. Defaults to False.
        """
        if file_path is None:
            now = datetime.now()
            datetime_str = now.strftime('%Y%m%d_%H%M%S')
            model_name = f'model_{"pb_al_" if not is_virtual else ""}{datetime_str}'
            model_folder = self.model_folder_virtual if is_virtual else self.model_folder
            file_path = os.path.join(model_folder, model_name)

        self.save_trained_model(file_path)

    def save_trained_model(self, file_path: str):
        """
        Saves a trained model to a file.

        Args:
            file_path (str): The path where the model will be saved.
        """
        print(f"Saving trained model to {file_path}.")
        self.model.save(file_path)
        zip_and_delete_directory(file_path, file_path)

    def weighted_average_temperature(self, profile: pd.DataFrame,  soc_indices: list, temperature_column: str = "Max. Cell Temperature in Ãƒâ€šÃ‚Â°C") -> float:
        """
        Calculates the weighted average temperature for a given time window in the profile dataframe.
        The weight is determined by the temperature value using an exponential scaling.
        Higher temperatures get more weight.

        Args:
            profile (pd.DataFrame): The dataframe containing the temperature column.
            window_size (int): Size of the window (number of rows or time steps).
            temperature_column (str): The name of the column containing temperature values.

        Returns:
            float: The weighted average temperature for the time window.
        """
        # Initialize the weighted sum and total weight
        weighted_sum = 0
        total_weight = 0

        for idx in soc_indices:  # Only use the indices corresponding to the same time snippet as SOC
            # Extract the temperature for the current timestep
            temperature = profile.loc[idx, temperature_column]

            # Define the weight as an exponential function of the temperature (higher temperature gets more weight)
            weight = np.exp(temperature / 25)  # arrhenius based scaling factor: 25Ãƒâ€šÃ‚Â°C gives a weight of e**(25/25)=2.718.
            # A temperature of 50Ãƒâ€šÃ‚Â°C gives a weight of e**(50/25)=7.389.

            # Add the weighted temperature to the sum
            weighted_sum += temperature * weight
            total_weight += weight

        # Compute the weighted average temperature
        weighted_avg_temp = weighted_sum / total_weight if total_weight != 0 else 0
        return weighted_avg_temp

    def lfp_cycle_BattProDeep(
        self,
        profile: pd.DataFrame,
        years: int = 1,
        fec_window: int = 3,
        temperature: float = 25,
        temperature_type: str = "constant",
        temperature_column: str = "Temperature",
        cumulative_loss0: float = 0.0,
        last_std0: float = 0.0,
        log_file: str = "cycle_inputs.pkl",
        eps: float | None = None,
        fec_scale: float = 1.0,
    ):
        """
        Cycle aging model (BattProDeep) - looped-safe version:
        - Handles continuous FEC growth across years.
        - Scales effective FEC axis by 'fec_scale' (years multiplier).
        """

        num_steps = len(profile)
        loss_array = np.zeros(num_steps)
        loss_array_std = np.zeros(num_steps)
        delta_std_array = np.zeros(num_steps)

        cumulative_loss = float(cumulative_loss0)
        stddev = float(last_std0)
        soc_indices = []

        fec_t0 = profile["FEC"].iloc[0] * fec_scale
        logs = []

        with tqdm(total=num_steps, desc="Calculating Cyclic Aging (BattProDeep)") as pbar:
            for i, idx in enumerate(profile.index):
                fec_t1 = profile.loc[idx, "FEC"] * fec_scale
                delta_fec = fec_t1 - fec_t0

                # trigger model prediction at FEC window boundary
                if (delta_fec >= fec_window) or (idx == profile.index[-1]):
                    soc_t1 = profile.loc[soc_indices, "soc"].mean() if soc_indices else profile["soc"].iloc[0]

                    # temperature selection
                    if temperature_type == "constant":
                        temperature_t1 = temperature
                    elif temperature_type == "profile":
                        temperature_t1 = self.weighted_average_temperature(profile, soc_indices, temperature_column)
                    else:
                        raise ValueError(f"Invalid temperature_type: {temperature_type}")

                    doc_series = profile.loc[soc_indices, "Half_Cycle_Depth"].abs()

                    doc_t1 = ( doc_series.max()
                        if not doc_series.empty and not doc_series.isna().all()
                        else 0.0 )
                    charge_rate = profile.loc[soc_indices, "Charge_Rate"].replace(0, np.nan).mean()
                    discharge_rate = profile.loc[soc_indices, "Discharge_Rate"].abs().replace(0, np.nan).mean()

                    # --- log
                    logs.append({
                        "timestep": idx,
                        "soc": soc_t1,
                        "temp": temperature_t1,
                        "doc": doc_t1,
                        "charge_rate": charge_rate,
                        "discharge_rate": discharge_rate,
                        "fec_t0": fec_t0,
                        "fec_t1": fec_t1
                    })

                    # --- model inputs
                    new_data_t0 = np.array([fec_t0, charge_rate, discharge_rate, temperature_t1, soc_t1 * 100, doc_t1 * 100])
                    new_data_t1 = np.array([fec_t1, charge_rate, discharge_rate, temperature_t1, soc_t1 * 100, doc_t1 * 100])

                    new_data_t0 = scale_data(None, new_data_t0.reshape(-1, 1, new_data_t0.shape[0]),
                                            path=self.normalizer_path_cycle, val_test_flag=True)
                    new_data_t1 = scale_data(None, new_data_t1.reshape(-1, 1, new_data_t1.shape[0]),
                                            path=self.normalizer_path_cycle, val_test_flag=True)

                    raw_output_model = tf.keras.Model(inputs=self.model_prob.input,
                                                    outputs=self.model_prob.layers[-2].output)
                    raw_params_t0 = raw_output_model(new_data_t0)
                    raw_params_t1 = raw_output_model(new_data_t1)
                    dist_t0 = tfpl.IndependentNormal(event_shape=1)(raw_params_t0)
                    dist_t1 = tfpl.IndependentNormal(event_shape=1)(raw_params_t1)

                    # loss = (dist_t1.mean().numpy().reshape(-1)[0] -
                    #         dist_t0.mean().numpy().reshape(-1)[0]) / 100
                    stddev = dist_t1.stddev().numpy().reshape(-1)[0] / 100




                    ####
                    stddev_t0_frac = dist_t0.stddev().numpy().reshape(-1)[0] / 100
                    if eps is None:
                        loss = (dist_t1.mean().numpy().reshape(-1)[0] -
                            dist_t0.mean().numpy().reshape(-1)[0]) / 100
                        delta_stddev = np.sqrt(np.maximum(stddev**2 - stddev_t0_frac**2, 0.0))
                        delta_std_array[i] = delta_stddev
                    else:
                        loss = (dist_t1.mean().numpy().reshape(-1)[0] -
                            dist_t0.mean().numpy().reshape(-1)[0]) / 100
                        delta_stddev = np.sqrt(np.maximum(stddev**2 - stddev_t0_frac**2, 0.0))
                        loss = loss + eps * delta_stddev

                    #####

                    loss_array[i] = loss
                    loss_array_std[i] = stddev
                    cumulative_loss += loss
                    fec_t0 = fec_t1
                    soc_indices = []
                else:
                    loss_array[i] = 0
                    loss_array_std[i] = stddev
                    soc_indices.append(idx)
                    delta_std_array[i] = 0

                pbar.update(1)

        # append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            print(f"Appended {len(logs)} rows -> {os.path.basename(log_file)}")

        return loss_array, loss_array_std, cumulative_loss, stddev, fec_t0, delta_std_array

    def _bootstrap_subset_indices(self, n_boot: int = 32, subset_seed: int | None = 42) -> np.ndarray:
        total = len(self.bootstrap_models)
        n_boot = min(int(n_boot), total)
        if n_boot < 1:
            raise ValueError("No bootstrap models are loaded for epistemic prediction.")

        if not hasattr(self, "_bootstrap_subset_cache"):
            self._bootstrap_subset_cache = {}

        key = (n_boot, None if subset_seed is None else int(subset_seed))
        if key not in self._bootstrap_subset_cache:
            if n_boot == total:
                indices = np.arange(total, dtype=np.int32)
            else:
                rng = np.random.default_rng(subset_seed)
                indices = rng.choice(total, size=n_boot, replace=False).astype(np.int32)
            self._bootstrap_subset_cache[key] = indices

        return self._bootstrap_subset_cache[key]

    def epistemic_stats(self, x, n_boot: int = 32, subset_seed: int | None = 42):
        """
        x: (M, 1, F)
        returns:
            mean: (M,)
            std : (M,)
        """

        preds = []
        subset_indices = self._bootstrap_subset_indices(n_boot=n_boot, subset_seed=subset_seed)
        for idx in subset_indices:
            m = self.bootstrap_models[int(idx)]
            out = m(x, training=False)

            # TFP safety: extract tensor
            if hasattr(out, "mean"):
                out = out.mean()

            out = tf.convert_to_tensor(out).numpy().reshape(-1)
            preds.append(out)

        preds = np.stack(preds, axis=0)  # (B, M)

        return preds.mean(axis=0), preds.std(axis=0)


    def mu_epistemic(self,x, zeta):
        zeta = np.asarray(zeta).astype(int).reshape(-1)

        mu = np.empty(len(zeta), dtype=float)

        for k in np.unique(zeta):
            idx = (zeta == k)
            out = self.bootstrap_models[k](x[idx], training=False)
            if hasattr(out, "mean"):
                out = out.mean()
            mu[idx] = tf.convert_to_tensor(out).numpy().reshape(-1) / 100.0

        return mu

    def bootstrap_mean_matrix(self, x, n_boot: int = 32, subset_seed: int | None = 42):
        subset_indices = self._bootstrap_subset_indices(n_boot=n_boot, subset_seed=subset_seed)
        preds = []

        for idx in subset_indices:
            m = self.bootstrap_models[int(idx)]
            out = m(x, training=False)
            if hasattr(out, "mean"):
                out = out.mean()
            preds.append(tf.convert_to_tensor(out).numpy().reshape(-1))

        return np.stack(preds, axis=0) / 100.0
    def reference_epistemic_factors(
        self,
        x0_ref,
        x1_ref,
        n_boot: int = 32,
        subset_seed: int | None = 42,
        eps_num: float = 1e-9,
    ) -> np.ndarray:
        pred_pair = self.bootstrap_mean_matrix(
            np.concatenate([x0_ref, x1_ref], axis=0),
            n_boot=n_boot,
            subset_seed=subset_seed,
        )
        pred_t0 = pred_pair[:, 0]
        pred_t1 = pred_pair[:, 1]
        delta_boot = np.maximum(pred_t1 - pred_t0, eps_num)
        delta_mean = float(delta_boot.mean())
        if not np.isfinite(delta_mean) or delta_mean <= eps_num:
            return np.ones_like(delta_boot, dtype=np.float32)
        return (delta_boot / delta_mean).astype(np.float32, copy=False)

    def mu_epistemic_fast(self, x, zeta, n_boot: int = 32, subset_seed: int | None = 42):
        pred_matrix = self.bootstrap_mean_matrix(x, n_boot=n_boot, subset_seed=subset_seed)
        n_boot = pred_matrix.shape[0]
        zeta = np.asarray(zeta).astype(int).reshape(-1) % n_boot
        return pred_matrix[zeta, np.arange(len(zeta))]

    def centered_bootstrap_increment(
        self,
        x0,
        x1,
        delta_center: np.ndarray,
        zeta,
        n_boot: int = 32,
        subset_seed: int | None = 42,
        centering: str = "absolute",
        epistemic_scale: float = 1,#1.0,
    ) -> np.ndarray:
        pred_pair = self.bootstrap_mean_matrix(
            np.concatenate([x0, x1], axis=0),
            n_boot=n_boot,
            subset_seed=subset_seed,
        )
        n_particles = x0.shape[0]
        delta_boot = pred_pair[:, n_particles:] - pred_pair[:, :n_particles]
        delta_boot_mean = delta_boot.mean(axis=0)

        zeta_idx = np.asarray(zeta).astype(int).reshape(-1) % delta_boot.shape[0]
        selected = delta_boot[zeta_idx, np.arange(n_particles)]
        delta_center = np.asarray(delta_center, dtype=np.float32).reshape(n_particles)
        anomaly = selected - delta_boot_mean
        scale = float(epistemic_scale)
        if centering == "relative":
            denom = np.maximum(np.abs(delta_boot_mean), 1e-12)
            delta_mu = delta_center * (1.0 + scale * (anomaly / denom))
        elif centering == "absolute":
            delta_mu = delta_center + scale * anomaly
        elif centering == "minchange":
            denom = np.maximum(np.abs(delta_boot_mean), 1e-12)
            delta_abs = delta_center + scale * anomaly
            delta_rel = delta_center * (1.0 + scale * (anomaly / denom))
            use_rel = np.abs(delta_rel - delta_center) <= np.abs(delta_abs - delta_center)
            delta_mu = np.where(use_rel, delta_rel, delta_abs)
        else:
            raise ValueError(f"Unknown bootstrap centering mode: {centering}")
        delta_mu = np.maximum(delta_mu, 0.0).astype(np.float32, copy=False)

        bias = float(delta_mu.mean() - delta_center.mean())
        if np.isfinite(bias) and bias > 0.0:
            delta_mu = np.maximum(delta_mu - bias, 0.0).astype(np.float32, copy=False)

        if n_particles > 0:
            delta_mu[0] = delta_center[0]
        return delta_mu.astype(np.float32, copy=False)

    def reference_centered_bootstrap_increment(
        self,
        x0_ref,
        x1_ref,
        delta_center: np.ndarray,
        zeta,
        n_boot: int = 32,
        subset_seed: int | None = 42,
        epistemic_scale: float = 0.5,#1.0,
    ) -> np.ndarray:
        pred_pair = self.bootstrap_mean_matrix(
            np.concatenate([x0_ref[:1], x1_ref[:1]], axis=0),
            n_boot=n_boot,
            subset_seed=subset_seed,
        )
        delta_boot = pred_pair[:, 1] - pred_pair[:, 0]
        delta_boot_mean = float(delta_boot.mean())

        delta_center = np.asarray(delta_center, dtype=np.float32).reshape(-1)
        zeta_idx = np.asarray(zeta).astype(int).reshape(-1) % delta_boot.shape[0]
        anomaly = delta_boot[zeta_idx] - delta_boot_mean
        scale = float(epistemic_scale)

        denom = max(abs(delta_boot_mean), 1e-12)
        delta_abs = delta_center + scale * anomaly
        delta_rel = delta_center * (1.0 + scale * (anomaly / denom))
        use_rel = np.abs(delta_rel - delta_center) <= np.abs(delta_abs - delta_center)
        delta_mu = np.where(use_rel, delta_rel, delta_abs)
        delta_mu = np.maximum(delta_mu, 0.0).astype(np.float32, copy=False)

        bias = float(delta_mu.mean() - delta_center.mean())
        if np.isfinite(bias) and bias > 0.0:
            delta_mu = np.maximum(delta_mu - bias, 0.0).astype(np.float32, copy=False)

        if delta_mu.size > 0:
            delta_mu[0] = delta_center[0]
        return delta_mu.astype(np.float32, copy=False)

    def lfp_cycle_BattProDeep_vectorized(
        self,
        M,
        profile,
        soc_particles,
        delta_FEC_particles,FEC_particles,DoD_particles,timestamps_particles, charge_rate_particles,discharge_rate_particles,
        years: int = 1,
        fec_window: int = 3,
        temperature: float = 25,
        temperature_type: str = "constant",
        temperature_column: str = "Temperature",
        cumulative_loss0:np.ndarray = 0.0, #should be multiple particles??
        max_spread0:float=0,

        log_file: str = "cycle_inputs.pkl",
        eps: float | None = None,
        zeta: np.ndarray | None = None,
        n_boot: int = 32,
        bootstrap_seed: int | None = 42,
        epistemic_mode: str = "anchored", # or "full"
        epistemic_support_scale: float = 1.0,
        fec_scale: float = 1.0,
        return_det: bool = False,
    ):

        #num_steps = len(soc_particles, axis=1) #lenght of "timesteps" in the soc_particles
        num_steps = soc_particles.shape[1]
        loss_array_particles = np.zeros((M, num_steps), dtype=np.float32)
        det_loss_array_particles = np.zeros((M, num_steps), dtype=np.float32) if return_det else None
        fec_array_particles = np.zeros((M, num_steps), dtype=np.float32)
        #loss_array_std = np.zeros(num_steps)
        #delta_std_array = np.zeros(num_steps)
        #cumulative_loss_trace = np.zeros((M, num_steps))        # store cumulative loss per timestep, I dont need this cumulative_loss_particles[trigger_idx] += loss_particles
        #fec_t0_trace = np.zeros((M, num_steps))  # store FEC per timestep
        cumulative_loss_particles = np.asarray(cumulative_loss0, dtype=np.float32)
        if cumulative_loss_particles.ndim == 0:
            cumulative_loss_particles = np.full(M, float(cumulative_loss_particles), dtype=np.float32)
        else:
            cumulative_loss_particles = cumulative_loss_particles.reshape(M).astype(np.float32, copy=False)
        #stddev = float(last_std0)
        ##
        ##soc_indices = [[] for _ in range(M)]
        ##
        window_start = np.zeros(M, dtype=np.int32)
        soc_sum = np.zeros(M, dtype=np.float32)
        soc_count = np.zeros(M, dtype=np.int32)

        dod_max = np.zeros(M, dtype=np.float32)

        charge_sum = np.zeros(M, dtype=np.float32)
        charge_count = np.zeros(M, dtype=np.int32)

        discharge_sum = np.zeros(M, dtype=np.float32)
        discharge_count = np.zeros(M, dtype=np.int32)
        max_spread = np.float32(max_spread0)
        logs = []
        FEC_particles_arr = np.asarray(FEC_particles, dtype=np.float32) * np.float32(fec_scale)
        DoD_particles_arr = np.asarray(DoD_particles, dtype=np.float32)
        charge_rate_particles_arr = np.asarray(charge_rate_particles, dtype=np.float32)
        discharge_rate_particles_arr = np.asarray(discharge_rate_particles, dtype=np.float32)
        timestamps_shared = np.asarray(timestamps_particles)
        fec_t0_particles = FEC_particles_arr[:, 0].astype(np.float32, copy=True)
        if eps is None:
            print("vectorized version needs epsilon")
            eps = np.zeros(M, dtype=np.float32)
        else:
            eps = np.asarray(eps, dtype=np.float32).reshape(M,)
        if zeta is None:
            zeta = np.zeros(M, dtype=np.float32)
        else:
            zeta = np.asarray(zeta, dtype=np.float32).reshape(M,)

        with tqdm(total=num_steps, desc="Calculating Cyclic Aging (BattProDeep)") as pbar:
            for i, idx in enumerate(timestamps_shared):
                # Calculate per-particle delta FEC
                # Current FEC for all particles
                fec_t1_particles = FEC_particles_arr[:, i]
                delta_fec_particles = fec_t1_particles - fec_t0_particles

                # Trigger model evaluation where delta FEC >= window or last step
                trigger_particles = (delta_fec_particles >= fec_window) | (i == num_steps - 1)
                trigger_idx = np.where(trigger_particles)[0]  # indices of triggered particles
                not_trigger_idx = np.where(~trigger_particles)[0]  # indices of accumulating particles

                # Vectorized: compute features only for triggered particles
                if len(trigger_idx) > 0:
                    # Prepare arrays to collect features
                    soc_t1_vec = np.zeros(len(trigger_idx), dtype=np.float32)
                    doc_t1_vec = np.zeros(len(trigger_idx), dtype=np.float32)
                    charge_rate_t1_vec = np.zeros(len(trigger_idx), dtype=np.float32)
                    discharge_rate_t1_vec = np.zeros(len(trigger_idx), dtype=np.float32)
                    temperature_t1_vec = np.zeros(len(trigger_idx), dtype=np.float32)

                    soc_t1_vec = np.where(
                    soc_count[trigger_idx] > 0,
                    soc_sum[trigger_idx] / soc_count[trigger_idx],
                    soc_particles[trigger_idx, i]
                )

                    doc_t1_vec = dod_max[trigger_idx]

                    charge_rate_t1_vec = np.where(
                        charge_count[trigger_idx] > 0,
                        charge_sum[trigger_idx] / charge_count[trigger_idx],
                        charge_rate_particles_arr[trigger_idx, i] #fallback use c rate for current timestep
                    )

                    discharge_rate_t1_vec = np.where(
                        discharge_count[trigger_idx] > 0,
                        discharge_sum[trigger_idx] / discharge_count[trigger_idx],
                        discharge_rate_particles_arr[trigger_idx, i] #fallback use c rate for current timestep
                    )

                    if temperature_type == "constant":
                        temperature_t1_vec[:] = temperature
                    else:
                        # temperature_t1_vec = np.array([self.weighted_average_temperature(profile=profile, soc_indices=timestamps_arr[m][soc_indices[m]],  # timestamps for accumulated SOC steps
                        #                                                                   temperature_column=temperature_column) for m in trigger_idx])
                        temperature_t1_vec = np.array([
                        self.weighted_average_temperature(
                            profile=profile,
                            soc_indices=timestamps_shared[window_start[m]:i+1],
                            temperature_column=temperature_column
                        )
                        for m in trigger_idx
                    ], dtype=np.float32)


                    # --- Build NN input arrays ONLY for triggered particles
                    x0 = np.stack(
                        [
                            fec_t0_particles[trigger_idx],
                            charge_rate_t1_vec,
                            discharge_rate_t1_vec,
                            temperature_t1_vec,
                            soc_t1_vec * 100.0,
                            doc_t1_vec * 100.0,
                        ],
                        axis=1,
                    ).astype(np.float32, copy=False)

                    x1 = np.stack(
                        [
                            fec_t1_particles[trigger_idx],
                            charge_rate_t1_vec,
                            discharge_rate_t1_vec,
                            temperature_t1_vec,
                            soc_t1_vec * 100.0,
                            doc_t1_vec * 100.0,
                        ],
                        axis=1,
                    ).astype(np.float32, copy=False)

                    # Reshape for model: (batch, timesteps=1, features)
                    x0 = scale_data(None, x0.reshape(len(trigger_idx),1,-1), path=self.normalizer_path_cycle, val_test_flag=True)
                    x1 = scale_data(None, x1.reshape(len(trigger_idx),1,-1), path=self.normalizer_path_cycle, val_test_flag=True)

                    zeta_triggered = zeta[trigger_idx]
                    prob_mean_pair, sigma_pair = self._probabilistic_mean_std(
                        np.concatenate([x0, x1], axis=0)
                    )
                    prob_mean_pair = prob_mean_pair / 100.0
                    sigma_pair = sigma_pair / 100.0
                    n_trigger = len(trigger_idx)
                    sigma_a_t0 = sigma_pair[:n_trigger]
                    sigma_a_t1 = sigma_pair[n_trigger:]
                    x0_ref_for_sigma = None
                    x1_ref_for_sigma = None

                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        mu_t0 = prob_mean_pair[:n_trigger]
                        mu_t1 = prob_mean_pair[n_trigger:]

                    # ============================================================
                        # EPistemic model (YOUR ACCEPTED FORMULATION)
                        # ============================================================
                    # def mu_epistemic(x, zeta_local):
                    #     mu = self.model_det.predict(x, verbose=False).reshape(-1) / 100.0
                    #     sigma_e = self.epistemic_std(x) / 100.0

                    #     # deterministic re-parameterization (NOT noise)
                    #     return mu * (1.0 + zeta_local * sigma_e)
                     # ============================================================
                    # EPistemic-aware predictions (KEY CHANGE)
                    # ============================================================

                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        delta_mu = mu_t1 - mu_t0
                        if epistemic_mode == "sigma_e_reference_refsigma":
                            soc_ref = (
                                soc_sum[0] / soc_count[0]
                                if soc_count[0] > 0
                                else soc_particles[0, i]
                            )
                            charge_ref = (
                                charge_sum[0] / charge_count[0]
                                if charge_count[0] > 0
                                else charge_rate_particles_arr[0, i]
                            )
                            discharge_ref = (
                                discharge_sum[0] / discharge_count[0]
                                if discharge_count[0] > 0
                                else discharge_rate_particles_arr[0, i]
                            )
                            if temperature_type == "constant":
                                temperature_ref = temperature
                            else:
                                temperature_ref = self.weighted_average_temperature(
                                    profile=profile,
                                    soc_indices=timestamps_shared[window_start[0] : i + 1],
                                    temperature_column=temperature_column,
                                )
                            x0_ref_raw = np.array(
                                [[
                                    fec_t0_particles[0],
                                    charge_ref,
                                    discharge_ref,
                                    temperature_ref,
                                    soc_ref * 100.0,
                                    dod_max[0] * 100.0,
                                ]],
                                dtype=np.float32,
                            )
                            x1_ref_raw = np.array(
                                [[
                                    FEC_particles_arr[0, i],
                                    charge_ref,
                                    discharge_ref,
                                    temperature_ref,
                                    soc_ref * 100.0,
                                    dod_max[0] * 100.0,
                                ]],
                                dtype=np.float32,
                            )
                            x0_ref_for_sigma = scale_data(
                                None,
                                x0_ref_raw.reshape(1, 1, -1),
                                path=self.normalizer_path_cycle,
                                val_test_flag=True,
                            )
                            x1_ref_for_sigma = scale_data(
                                None,
                                x1_ref_raw.reshape(1, 1, -1),
                                path=self.normalizer_path_cycle,
                                val_test_flag=True,
                            )
                    elif epistemic_mode == "full":
                        f_t0 = self.mu_epistemic_fast(x0, zeta_triggered, n_boot=n_boot, subset_seed=bootstrap_seed)
                        f_t1 = self.mu_epistemic_fast(x1, zeta_triggered, n_boot=n_boot, subset_seed=bootstrap_seed)
                        delta_mu = f_t1 - f_t0
                    elif epistemic_mode in (
                        "full_centered",
                        "full_centered_rel",
                        "full_centered_minchange",
                        "full_centered_minchange_reference",
                        "full_centered_minchange_reference_refsigma",
                        "full_centered_support",
                    ):
                        delta_center = (prob_mean_pair[n_trigger:] - prob_mean_pair[:n_trigger]).astype(
                            np.float32,
                            copy=False,
                        )
                        if epistemic_mode in (
                            "full_centered_minchange_reference",
                            "full_centered_minchange_reference_refsigma",
                        ):
                            soc_ref = (
                                soc_sum[0] / soc_count[0]
                                if soc_count[0] > 0
                                else soc_particles[0, i]
                            )
                            charge_ref = (
                                charge_sum[0] / charge_count[0]
                                if charge_count[0] > 0
                                else charge_rate_particles_arr[0, i]
                            )
                            discharge_ref = (
                                discharge_sum[0] / discharge_count[0]
                                if discharge_count[0] > 0
                                else discharge_rate_particles_arr[0, i]
                            )
                            if temperature_type == "constant":
                                temperature_ref = temperature
                            else:
                                temperature_ref = self.weighted_average_temperature(
                                    profile=profile,
                                    soc_indices=timestamps_shared[window_start[0] : i + 1],
                                    temperature_column=temperature_column,
                                )
                            x0_ref_raw = np.array(
                                [[
                                    fec_t0_particles[0],
                                    charge_ref,
                                    discharge_ref,
                                    temperature_ref,
                                    soc_ref * 100.0,
                                    dod_max[0] * 100.0,
                                ]],
                                dtype=np.float32,
                            )
                            x1_ref_raw = np.array(
                                [[
                                    FEC_particles_arr[0, i],
                                    charge_ref,
                                    discharge_ref,
                                    temperature_ref,
                                    soc_ref * 100.0,
                                    dod_max[0] * 100.0,
                                ]],
                                dtype=np.float32,
                            )
                            x0_ref = scale_data(
                                None,
                                x0_ref_raw.reshape(1, 1, -1),
                                path=self.normalizer_path_cycle,
                                val_test_flag=True,
                            )
                            x1_ref = scale_data(
                                None,
                                x1_ref_raw.reshape(1, 1, -1),
                                path=self.normalizer_path_cycle,
                                val_test_flag=True,
                            )
                            x0_ref_for_sigma = x0_ref
                            x1_ref_for_sigma = x1_ref
                            delta_mu = self.reference_centered_bootstrap_increment(
                                x0_ref,
                                x1_ref,
                                delta_center,
                                zeta_triggered,
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                        else:
                            delta_mu = self.centered_bootstrap_increment(
                                x0,
                                x1,
                                delta_center,
                                zeta_triggered,
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                                centering=(
                                    "relative" if epistemic_mode == "full_centered_rel"
                                    else "minchange" if epistemic_mode == "full_centered_minchange"
                                    else "absolute"
                                ),
                                epistemic_scale=(
                                    epistemic_support_scale
                                    if epistemic_mode == "full_centered_support"
                                    else 1.0
                                ),
                            )
                    elif epistemic_mode == "anchored":
                        det_pair = self.model_det.predict(
                            np.concatenate([x0, x1], axis=0),
                            verbose=False,
                        ).reshape(-1) / 100.0
                        n_trigger = len(trigger_idx)
                        det_t0 = det_pair[:n_trigger]
                        det_t1 = det_pair[n_trigger:]
                        delta_det = np.maximum(det_t1 - det_t0, 0.0).astype(np.float32, copy=False)

                        x0_ref_raw = np.array(
                            [[
                                fec_t0_particles[trigger_idx].mean(),
                                charge_rate_t1_vec.mean(),
                                discharge_rate_t1_vec.mean(),
                                temperature_t1_vec.mean(),
                                soc_t1_vec.mean() * 100.0,
                                doc_t1_vec.mean() * 100.0,
                            ]],
                            dtype=np.float32,
                        )
                        x1_ref_raw = np.array(
                            [[
                                fec_t1_particles[trigger_idx].mean(),
                                charge_rate_t1_vec.mean(),
                                discharge_rate_t1_vec.mean(),
                                temperature_t1_vec.mean(),
                                soc_t1_vec.mean() * 100.0,
                                doc_t1_vec.mean() * 100.0,
                            ]],
                            dtype=np.float32,
                        )
                        x0_ref = scale_data(
                            None,
                            x0_ref_raw.reshape(1, 1, -1),
                            path=self.normalizer_path_cycle,
                            val_test_flag=True,
                        )
                        x1_ref = scale_data(
                            None,
                            x1_ref_raw.reshape(1, 1, -1),
                            path=self.normalizer_path_cycle,
                            val_test_flag=True,
                        )
                        epi_factors = self.reference_epistemic_factors(
                            x0_ref,
                            x1_ref,
                            n_boot=n_boot,
                            subset_seed=bootstrap_seed,
                        )
                        zeta_mod = zeta_triggered.astype(np.int32) % epi_factors.shape[0]
                        delta_mu = delta_det * epi_factors[zeta_mod]
                    else:
                        raise ValueError(f"Unknown epistemic_mode: {epistemic_mode}")

                    if epistemic_mode in (
                        "full_centered_minchange_reference_refsigma",
                        "sigma_e_reference_refsigma",
                    ):
                        _, sigma_pair_ref = self._probabilistic_mean_std(
                            np.concatenate([x0_ref_for_sigma, x1_ref_for_sigma], axis=0)
                        )
                        sigma_pair_ref = sigma_pair_ref / 100.0
                        sigma_a_t0 = np.full(n_trigger, sigma_pair_ref[0], dtype=np.float32)
                        sigma_a_t1 = np.full(n_trigger, sigma_pair_ref[1], dtype=np.float32)
                    # ----------------------------
                    # aleatoric uncertainty
                    # ----------------------------
                    # # ----------------------------
                    # # epistemic uncertainty (bootstrap)
                    # # ----------------------------
                    # #alternatively
                    # # sigma_e_t0 = self.epistemic_std(x0)
                    # # sigma_e_t1 = self.epistemic_std(x1)
                    # #full stats
                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        if epistemic_mode == "sigma_e_reference_refsigma":
                            _, sigma_e_pair_ref = self.epistemic_stats(
                                np.concatenate([x0_ref_for_sigma, x1_ref_for_sigma], axis=0),
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                            sigma_e_pair_ref = sigma_e_pair_ref / 100.0
                            sigma_e_t0 = np.full(n_trigger, sigma_e_pair_ref[0], dtype=np.float32)
                            sigma_e_t1 = np.full(n_trigger, sigma_e_pair_ref[1], dtype=np.float32)
                        else:
                            _, sigma_e_pair = self.epistemic_stats(
                                np.concatenate([x0, x1], axis=0),
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                            sigma_e_pair = sigma_e_pair / 100.0
                            sigma_e_t0 = sigma_e_pair[:n_trigger]
                            sigma_e_t1 = sigma_e_pair[n_trigger:]
                    # ## only mean accross M
                    # # x1_mean = np.mean(x1, axis=0, keepdims=True)
                    # # _, sigma_e_t1 = self.epistemic_stats(x1_mean)
                    # # sigma_e_t1 = np.full(len(trigger_idx), sigma_e_t1 / 100.0)
                    # # x0_mean = np.mean(x0, axis=0, keepdims=True)
                    # # _, sigma_e_t0 = self.epistemic_stats(x0_mean)
                    # # sigma_e_t0 = np.full(len(trigger_idx), sigma_e_t0 / 100.0)
                    # =========================================================
                    # 3) OPTION 3: UNCERTAINTY GROWTH MODEL
                    # =========================================================
                    sigma_growth_a = np.maximum(sigma_a_t1 - sigma_a_t0, 0.0)
                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        sigma_growth_e = np.maximum(sigma_e_t1 - sigma_e_t0, 0.0)

                    # sigma_delta = np.sqrt(
                    #     (eps[trigger_idx]**2) *sigma_growth_a**2 + sigma_growth_e**2
                    # )
                    if epistemic_mode == "sigma_total":
                        sigma_delta = np.sqrt(sigma_growth_a**2 + sigma_growth_e**2)
                    else:
                        sigma_delta = sigma_growth_a

                    # =========================================================
                    # 5) stochastic increment (single injection)
                    # =========================================================
                    #delta = delta_mu +   sigma_delta
                    #eps_e = np.sign(eps)
                    delta = delta_mu + eps[trigger_idx] * sigma_delta
                    if epistemic_mode in ("sigma_e", "sigma_e_reference_refsigma"):
                        delta = delta + float(epistemic_support_scale) * zeta_triggered * sigma_growth_e
                    if epistemic_mode in (
                        "full_centered",
                        "full_centered_rel",
                        "full_centered_minchange",
                        "full_centered_minchange_reference",
                        "full_centered_minchange_reference_refsigma",
                        "full_centered_support",
                    ):
                        ref_local = np.where(trigger_idx == 0)[0]
                        if len(ref_local) > 0:
                            delta[ref_local[0]] = delta_mu[ref_local[0]]

                    # =========================================================
                    # 6) enforce physical constraints
                    # =========================================================
                    delta = np.maximum(delta, 0.0)

                    # optional: keep ensemble mass stable (recommended if needed)
                    # delta = delta / (np.mean(delta) + 1e-12) * np.mean(delta)

                    # =========================================================
                    # 7) output
                    # =========================================================
                    loss_triggered = delta
                    det_loss_triggered = delta_mu
                    # --- particle-space container ---
                    loss_particles = np.zeros(M, dtype=np.float32)
                    loss_particles[trigger_idx] = loss_triggered
                    if return_det:
                        det_loss_particles = np.zeros(M, dtype=np.float32)
                        det_loss_particles[trigger_idx] = det_loss_triggered
                    #####

                    # Assign losses and update cumulative values
                    # Assign back to full arrays
                    loss_array_particles[:, i] = loss_particles
                    if return_det:
                        det_loss_array_particles[:, i] = det_loss_particles
                    fec_array_particles[:, i] = fec_t1_particles
                    # Update cumulative loss safely
                    cumulative_loss_particles += loss_particles  # use full array, not trigger_idx

                    # Update FEC only for triggered particles
                    fec_t0_particles[trigger_idx] = fec_t1_particles[trigger_idx]

                    window_start[trigger_idx] = i + 1
                    soc_sum[trigger_idx] = 0
                    soc_count[trigger_idx] = 0

                    dod_max[trigger_idx] = 0

                    charge_sum[trigger_idx] = 0
                    charge_count[trigger_idx] = 0

                    discharge_sum[trigger_idx] = 0
                    discharge_count[trigger_idx] = 0

                     # --- log
                    logs.append({
                        "timestep": idx,
                        "soc": soc_t1_vec.mean(axis=0),
                        "temp": temperature_t1_vec.mean(),
                        "doc": doc_t1_vec.mean(),
                        "charge_rate": charge_rate_t1_vec.mean() ,
                        "discharge_rate": discharge_rate_t1_vec.mean() ,
                        "fec_t0": fec_t0_particles.mean(axis=0),
                        #"last_std1":sigma_a_t1.mean(axis=0), #fractional
                        #"last_std_e1":sigma_e_t1.mean(axis=0), #fractional
                        "loss_inc":loss_particles.mean(axis=0),
                        #"L1":mu_t1.mean(axis=0),
                        "cum_loss_pct":cumulative_loss_particles.mean(axis=0)*100,
                        "cum_loss_pct_high": np.percentile(cumulative_loss_particles, 95, axis=0) * 100,
                        "cum_loss_pct_low": np.percentile(cumulative_loss_particles, 5, axis=0) * 100,
                    })
                # Non-triggered particles: accumulate indices
                for m in not_trigger_idx:
                    ##
                    ##soc_indices[m].append(i)
                    ##
                    soc_val = soc_particles[m, i]
                    soc_sum[m] += soc_val
                    soc_count[m] += 1

                    dod_val = abs(DoD_particles_arr[m, i])
                    if dod_val > dod_max[m]:
                        dod_max[m] = dod_val

                    charge_val = charge_rate_particles_arr[m, i]
                    if charge_val != 0:
                        charge_sum[m] += charge_val
                        charge_count[m] += 1

                    discharge_val = abs(discharge_rate_particles_arr[m, i])
                    if discharge_val != 0:
                        discharge_sum[m] += discharge_val
                        discharge_count[m] += 1




                pbar.update(1)

        # append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            #print(f"Appended {len(logs)} rows -> {os.path.basename(log_file)}")

        return (
            loss_array_particles.astype(np.float32, copy=False),
            cumulative_loss_particles.astype(np.float32, copy=False),
            fec_array_particles.astype(np.float32, copy=False),
            max_spread,
            det_loss_array_particles.astype(np.float32, copy=False) if return_det else None,
        )



    def lfp_cycle_Naumann(self, profile: pd.DataFrame, years: int = 1, fec_window: int = 10) -> np.ndarray:
        """
        Calculates aging using the Naumann method.

        Args:
            profile (pd.DataFrame): SOC profile dataframe.
            years (int, optional): Number of years to iterate the SOC profile. Defaults to 1.
            fec_window (int, optional): Intervals for calculating the cycle aging. Defaults to 10.

        Returns:
            np.ndarray: Array of capacity loss for each timestep.
        """
        num_steps = len(profile)
        cumulative_loss = 0.0
        loss_array = np.zeros(num_steps * years)
        soc_indices = []
        fec_t0, fec_year = 0, 0

        # Constants for the Naumann method
        A_QLOSS, B_QLOSS, C_QLOSS, D_QLOSS = 0.0630, 0.0971, 4.0253, 1.0923

        # Add progress bar using tqdm
        with tqdm(total=num_steps * years, desc="Calculating Cyclic Aging (Naumann)") as pbar:
            for year in range(years):
                for i, idx in enumerate(profile.index):
                    fec_t1 = fec_year + profile.loc[idx, "FEC"]
                    delta_fec = fec_t1 - fec_t0

                    if delta_fec >= fec_window:
                        # doc = profile.loc[soc_indices, "Half_Cycle_Depth"].abs().mean()
                        doc = profile.loc[soc_indices, "soc"].max() - profile.loc[soc_indices, "soc"].min()
                        c_rate = profile.loc[soc_indices, "C_Rate"].abs().mean()

                        # Calculate stress factor dependent coefficients
                        k_c_rate = A_QLOSS * c_rate + B_QLOSS
                        k_doc = C_QLOSS * (doc - 0.6) ** 3 + D_QLOSS

                        # Calculate capacity loss per step, based on virtual FEC and past total degradation
                        virtual_fec = (cumulative_loss * 100 / (k_c_rate * k_doc)) ** 2
                        loss = k_c_rate * k_doc * math.sqrt(virtual_fec + delta_fec) / 100
                        loss -= cumulative_loss  # Relative capacity loss in current timestep
                        loss_array[i + num_steps * year] = loss

                        # Update variables
                        cumulative_loss += loss
                        fec_t0 = fec_t1
                        soc_indices = []  # Reset index count
                    else:
                        soc_indices.append(idx)

                    pbar.update(1)  # Update the progress bar

                fec_year = fec_t1

        return loss_array

    def _get_network(self, is_virtual: bool = False) -> Sequential:
        """
        Retrieves the underlying neural network or computational model used in the mechanism.

        Args:
            is_virtual (bool, optional): Whether the model is virtual or not. Defaults to False.

        Returns:
            Sequential: The neural network or model.
        """
        print("Retrieving the neural network.")
        if is_virtual:
            return Sequential([
                Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
                LSTM(64, return_sequences=True, activation='relu', kernel_initializer='he_normal', use_bias=False),
                LSTM(64, return_sequences=True, activation='relu', kernel_initializer='he_normal', use_bias=False),
                Dense(1, kernel_initializer='zeros', use_bias=False)
            ])
        else:
            return Sequential([
                Input(shape=(self.X_train.shape[1], self.X_train.shape[2])),
                Dense(512, kernel_initializer='he_normal', activation='relu'),
                Dense(256, kernel_initializer='he_normal', activation='relu'),
                Dense(tfpl.IndependentNormal.params_size(event_shape=1), kernel_initializer='zeros'),
                tfpl.IndependentNormal(event_shape=1) #consider changing into event_shape=1, convert_to_tensor_fn=None for generating not only mean
            ])
            # IMPORTANT: The Dense layer outputs parameters for the probabilistic distribution:
            # It produces both mean and scale parameters.
            # Using kernel_initializer='zeros' here means the initial outputs are zero,
            # which can cause scale (stddev) to be very small or zero after the softplus transform.
            #
            # To improve training and avoid near-zero stddev outputs:
            # - Consider adding a bias_initializer with a small positive constant (e.g., bias_initializer=Constant(1.0))
            #   so the scale parameter starts from a reasonable positive value.
            # - This helps the model learn meaningful uncertainty and avoids the stddev collapsing to zero.
            #Dense(tfpl.IndependentNormal.params_size(event_shape=1), kernel_initializer='zeros',
                  #bias_initializer=tf.keras.initializers.Constant(1.0)),  # <-- Add bias initializer here

            # Probabilistic output layer:
            # This layer parameterizes a Normal distribution with learned mean and stddev.
            # Optional: you can add convert_to_tensor_fn=None to prevent the layer
            # from automatically converting distribution to tensor during prediction,
            # which allows you to call .mean() and .stddev() on the output distribution object.
            #
            # Example:
            # tfpl.IndependentNormal(event_shape=1, convert_to_tensor_fn=None)
            #
            # This is useful if you want to separately access mean and uncertainty at inference.
            #tfpl.IndependentNormal(event_shape=1)  # Consider adding convert_to_tensor_fn=None if needed


    def visualize(self, data: pd.DataFrame):
        """
        Visualizes the results or status of the mechanism using the provided data.

        Args:
            data (pd.DataFrame): Data to be visualized.
        """
        # Placeholder for visualization logic
        pass

    def get_trained_model(self, file_path: str, **kwargs) -> tf.keras.Model:
        """
        Loads a trained model from a file.

        Args:
            file_path (str): The path from which to load the model.
            **kwargs: Additional keyword arguments for custom objects.

        Returns:
            tf.keras.Model: The loaded model.
        """
        custom_objects = kwargs.get('custom_objects', None)
        load_epistemic = kwargs.get('load_epistemic', False)
        print(f"Loading trained model from {file_path}.")
        return load_latest_model(    file_path,    custom_objects=custom_objects,    load_epistemic=load_epistemic)

    def preprocess_data(self, data: dict, split_percentages: list = None, **kwargs):
        """
        Preprocesses data for training or analysis.

        Args:
            data (dict): The data to be processed.
            split_percentages (list, optional): The percentages of the data to be used for training, validation, and
            testing. Defaults to [0.95, 0, 0.05].
        """
        df_comb = data['combined']
        df_calendar = data['calendar']
        if split_percentages is None:
            split_percentages = [0.95, 0, 0.05]
        print("Preprocessing data.")

        interpolate_rows = kwargs.get('interpolate_rows', True)
        fec_step_size = kwargs.get('fec_step_size', 50)
        self.conditions = extract_conditions(df_comb, degradation_type=self.degradation_type,
                                             complete_data=self.complete_data)

        self.train_conditions, self.dev_conditions, self.test_conditions = shuffle_and_split(self.conditions,
                                                                                             split_percentages)

        self.input_train, self.labels_train = preprocess_data(df_comb,
                                                              self.train_conditions,
                                                              df_calendar,
                                                              degradation_type=self.degradation_type,
                                                              interpolate_rows=interpolate_rows,
                                                              fec_step_size=fec_step_size,
                                                              complete_data=self.complete_data)
        self.input_test, self.labels_test = preprocess_data(df_comb,
                                                            self.test_conditions,
                                                            df_calendar,
                                                            degradation_type=self.degradation_type,
                                                            interpolate_rows=interpolate_rows,
                                                            fec_step_size=fec_step_size,
                                                            complete_data=self.complete_data)

        scaler = MinMaxScaler()

        self.input_train_scaled = scale_data(scaler, self.input_train, path=self.normalizer_path_cycle)
        self.input_test_scaled = scale_data(scaler, self.input_test, path=self.normalizer_path_cycle,
                                            val_test_flag=True)

        self.df_train_inverse = create_inverse_df(self.input_train, self.labels_train,
                                                  degradation_type=self.degradation_type)
        self.df_test_inverse = create_inverse_df(self.input_test, self.labels_test,
                                                 degradation_type=self.degradation_type)

        self.df_train_inverse_scaled = create_inverse_df(self.input_train_scaled, self.labels_train,
                                                         degradation_type=self.degradation_type)
        self.df_test_inverse_scaled = create_inverse_df(self.input_test_scaled, self.labels_test,
                                                        degradation_type=self.degradation_type)

        if self.dev_conditions != {}:
            self.input_dev, self.labels_dev = preprocess_data(df_comb,
                                                              self.dev_conditions,
                                                              df_calendar,
                                                              degradation_type=self.degradation_type,
                                                              interpolate_rows=interpolate_rows,
                                                              fec_step_size=fec_step_size,
                                                              complete_data=self.complete_data)
            self.input_dev_scaled = scale_data(scaler, self.input_dev, path=self.normalizer_path_cycle,
                                               val_test_flag=True)
            self.df_dev_inverse = create_inverse_df(self.input_dev, self.labels_dev,
                                                    degradation_type=self.degradation_type)
            self.df_dev_inverse_scaled = create_inverse_df(self.input_dev_scaled, self.labels_dev,
                                                           degradation_type=self.degradation_type)

        self.inputs, self.labels = preprocess_data(df_comb,
                                                   self.conditions,
                                                   df_calendar,
                                                   degradation_type=self.degradation_type,
                                                   interpolate_rows=interpolate_rows,
                                                   fec_step_size=fec_step_size,
                                                   complete_data=self.complete_data)

        self.inputs_scaled = scale_data(scaler, self.inputs, path=self.normalizer_path_cycle, val_test_flag=True)

        self.df_inverse = create_inverse_df(self.inputs, self.labels, degradation_type=self.degradation_type)

        self.df_inverse_scaled = create_inverse_df(self.inputs_scaled, self.labels,
                                                   degradation_type=self.degradation_type)

        self.X_train = np.array(self.input_train_scaled.tolist(), dtype=np.float32)
        self.Y_train = np.array(self.labels_train.tolist(), dtype=np.float32)
        if self.dev_conditions != {}:
            self.X_val = np.array(self.input_dev_scaled.tolist(), dtype=np.float32)
            self.Y_val = np.array(self.labels_dev.tolist(), dtype=np.float32)
        self.X_test = np.array(self.input_test_scaled.tolist(), dtype=np.float32)
        self.Y_test = np.array(self.labels_test.tolist(), dtype=np.float32)
