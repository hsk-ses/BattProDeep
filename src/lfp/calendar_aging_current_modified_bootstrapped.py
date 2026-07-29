import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tqdm import tqdm
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


class CalendarAging:
    """Implements calendar aging simulation based on the Mechanism protocol."""

    def __init__(self, load_trained_models: bool = True, complete_data: bool = False, **kwargs):
        """
        Initializes the CalendarAging with a configuration dictionary.

        Args:
            load_trained_models (bool): Whether to load trained models or not.
            complete_data (bool, optional): Whether data is completed or not. The public version of calendar aging
                dataset does not include all cells' data. Defaults to True.
            **kwargs: Additional keyword arguments for model paths.
        """
        # super().__init__(load_trained_models)
        self.degradation_type = "calendar"
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
        self.model_path_calendar = os.path.join(self.models_folder,
                                                kwargs.get('calendar_trained_folder', self.degradation_type))
        self.fig_save_path = os.path.join(parent_directory, kwargs.get('results_relative_folder', "results"))
        calendar_results_folder = kwargs.get('calendar_results_folder', self.degradation_type)
        self.fig_save_path_calendar = os.path.join(parent_directory, kwargs.get('calendar_figures_folder',
                                                                                os.path.join(calendar_results_folder,
                                                                                             "figures")))
        self.normalizer_path_calendar = os.path.join(self.model_path_calendar,
                                                     kwargs.get('calendar_normalizers_folder', "normalizer"))
        self.model_folder_virtual = os.path.join(self.model_path_calendar,
                                                 kwargs.get('model_relative_path_virtual', "virtual_time"))
        self.model_folder_det = os.path.join(self.model_path_calendar,
                                             kwargs.get('model_relative_path_det', "deterministic"))
        aleatoric_model_folder = (
            kwargs.get("aleatoric_model_folder")
            or kwargs.get("calendar_aleatoric_model_folder")
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
                    else Path(self.model_path_calendar) / aleatoric_model_folder
                )
        else:
            self.model_folder = os.path.join(
                self.model_path_calendar,
                kwargs.get(
                    'model_relative_path',
                    os.path.join('probabilistic', '01-aleatoric'),
                ),
            )
        epistemic_model_folder = (
            kwargs.get("epistemic_model_folder")
            or kwargs.get("calendar_epistemic_model_folder")
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
                    else Path(self.model_path_calendar) / epistemic_model_folder
                )
        else:
            self.model_folder_ep = os.path.join(
                self.model_path_calendar,
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
                                                                 'negative_log_likelihood': negative_log_likelihood}) #

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

    def train(self, df_calendar: pd.DataFrame, is_virtual: bool = False, save_model: bool = True,
              **kwargs):
        """
        Trains the mechanism's model on provided data.

        Args:
            df_calendar (pd.DataFrame): Dataframe containing the calendar data for training.
            labels (pd.DataFrame): Dataframe containing the labels for training.
            is_virtual (bool): Whether the model is virtual or not.
            save_model (bool): Whether to save the trained model or not.
            **kwargs: Additional keyword arguments for training configuration.
        """
        epochs = kwargs.get('epochs', 100)
        learning_rate = kwargs.get('learning_rate', 0.0001)
        loss = kwargs.get('loss', 'mae')
        batch_size = kwargs.get('batch_size', 1)
        verbose = kwargs.get('verbose', 1)
        file_path = kwargs.get('file_path', None)
        data = {
            'calendar': df_calendar
        }

        self.preprocess_data(data)
        print("Training model with provided data.")
        self.model = self._get_network(is_virtual=is_virtual)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_function = loss if is_virtual else negative_log_likelihood
        self.model.compile(optimizer=optimizer, loss=loss_function)

        fit_args = {'batch_size': batch_size, 'epochs': epochs, 'verbose': verbose}
        if self.dev_conditions:
            fit_args['validation_data'] = (self.X_val, self.Y_val)

        self.model.fit(self.X_train, self.Y_train, **fit_args)

        if save_model:
            self._save_model(file_path, is_virtual)

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


    def weighted_average_temperature(self, profile: pd.DataFrame,  soc_indices: list, temperature_column: str = "Max. Cell Temperature in Ã‚Â°C") -> float:
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
            weight = np.exp(temperature / 25)  # arrhenius based scaling factor: 25Ã‚Â°C gives a weight of e**(25/25)=2.718.
            # A temperature of 50Ã‚Â°C gives a weight of e**(50/25)=7.389.

            # Add the weighted temperature to the sum
            weighted_sum += temperature * weight
            total_weight += weight

        # Compute the weighted average temperature
        weighted_avg_temp = weighted_sum / total_weight if total_weight != 0 else 0
        return weighted_avg_temp

    def lfp_calendar_BattProDeep(
        self,
        profile: pd.DataFrame,
        variant_name: str,
        years: int = 1,
        vt_scale: float = 1.0,
        window_size: int = 60,
        temperature: float = 25,
        temperature_type: str = "constant",
        temperature_column: str = "Temperature",
        virtual_time0: float = 0.0,
        cumulative_loss0: float = 0.0,
        last_std0: float = 0.0,
        log_file: str = None,
        eps: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        """
        Calendar aging model (BattProDeep) - looped-safe version.
        Keeps virtual-time continuity across blocks.
        Scales total exposure using vt_scale (global 'years' multiplier).
        """

        if log_file is None:
            log_file = f"calendar_inputs_{variant_name}.pkl"


        num_steps = len(profile)
        if num_steps < 2:
            vt_trace = np.full(num_steps, virtual_time0, dtype=float)
            return (
                np.zeros(num_steps),
                np.full(num_steps, last_std0),
                virtual_time0, cumulative_loss0, last_std0, vt_trace
            )

        # --- timestep resolution ---
        dt_minutes = (profile.index[1] - profile.index[0]).total_seconds() / 60.0
        elapsed_minutes = (profile.index - profile.index[0]).total_seconds() / 60.0
        total_minutes = elapsed_minutes[-1]

        # --- initial states ---
        cumulative_loss_pct = float(cumulative_loss0)
        vt_t0 = float(virtual_time0)
        last_std = float(last_std0)
        det_cumulative_loss = float(cumulative_loss0)
        det_loss_array = np.zeros(num_steps)
        delta_std_array = np.zeros(num_steps)

        loss_array = np.zeros(num_steps)
        std_array = np.full(num_steps, last_std, dtype=float)
        vt_trace = np.full(num_steps, vt_t0, dtype=float)
        logs = []

        with tqdm(total=num_steps, desc="Calculating Calendar Aging (BattProDeep)") as pbar:
            soc_indices = []

            for i, idx in enumerate(profile.index):
                std_array[i] = last_std
                vt_trace[i] = vt_t0
                #soc_indices.append(idx) #moved to else branch

                is_boundary = (elapsed_minutes[i] % window_size == 0)
                if (is_boundary and i > 0) or (i == num_steps - 1):
                    soc_t1 = profile.loc[soc_indices, "soc"].mean()
                    temp_t1 = (
                        temperature
                        if temperature_type == "constant"
                        else self.weighted_average_temperature(profile, soc_indices, temperature_column)
                    )

                    # Scaled time increment
                    vt_t1 = vt_t0 + len(soc_indices) * (dt_minutes / 60.0) * vt_scale

                    # --- prepare scaled inputs ---
                    x0 = np.array([vt_t0, temp_t1, soc_t1 * 100.0], dtype=float)
                    x1 = np.array([vt_t1, temp_t1, soc_t1 * 100.0], dtype=float)

                    x0 = scale_data(None, x0.reshape(-1, 1, x0.shape[0]),
                                    path=self.normalizer_path_calendar, val_test_flag=True)
                    x1 = scale_data(None, x1.reshape(-1, 1, x1.shape[0]),
                                    path=self.normalizer_path_calendar, val_test_flag=True)

                    # --- model predictions probab.---
                    # loss_t0_pct = self.model_prob(x0).mean().numpy().reshape(-1)[0]
                    # loss_t1_pct = self.model_prob(x1).mean().numpy().reshape(-1)[0]
                    # --- model predictions deterministic instead bc of virtual time.---
                    loss_t0_pct = self.model_det.predict(x0, verbose=False).reshape(-1)[0]
                    loss_t1_pct = self.model_det.predict(x1, verbose=False).reshape(-1)[0]

                    _, sigma_t1_pct = self._probabilistic_mean_std(x1)
                    _, sigma_t0_pct = self._probabilistic_mean_std(x0)
                    stddev_frac = sigma_t1_pct[0] / 100

                    ####
                    stddev_t0_frac = sigma_t0_pct[0] / 100
                    if eps is None:
                        loss_delta_pct = loss_t1_pct - loss_t0_pct
                        delta_stddev = np.sqrt(np.maximum(stddev_frac**2 - stddev_t0_frac**2, 0.0))
                        delta_std_array[i] = delta_stddev
                    else:
                        L0 = loss_t0_pct + eps * (stddev_t0_frac*100)
                        L1 = loss_t1_pct + eps * (stddev_frac*100)
                        loss_delta_pct = L1 - L0

                    det_mu0=self.model_det.predict(x0, verbose=False).reshape(-1)[0]
                    det_mu1= self.model_det.predict(x1, verbose=False).reshape(-1)[0]
                    det_delta_loss=(det_mu1-det_mu0)/100
                    det_cumulative_loss += det_delta_loss
                    det_loss_array[i] = det_delta_loss

                    #####


                    # --- update arrays ---
                    loss_array[i] = loss_delta_pct / 100
                    std_array[i] = stddev_frac
                    cumulative_loss_pct += loss_delta_pct
                    last_std = stddev_frac

                    # --- update virtual time continuously ---
                    vt_in = np.array([cumulative_loss_pct, temp_t1, soc_t1 * 100.0])
                    #vt_in = np.array([det_cumulative_loss*100, temp_t1, soc_t1 * 100.0]) # make sure it uses deterministic loss
                    vt_in = scale_data(None, vt_in.reshape(-1, 1, vt_in.shape[0]),
                                    path=self.normalizer_path_calendar, is_inverse=True, val_test_flag=True)
                    vt_pred_norm = self.model_virtual.predict(vt_in, verbose=False).reshape(-1, 1)
                    vt_pred = inverse_scale(vt_pred_norm, path=self.normalizer_path_calendar,
                                            is_inverse=True, is_label=True)
                    vt_t0 = float(vt_pred)
                    vt_trace[i] = vt_t0

                    # --- store logs ---
                    logs.append({
                        "timestep": idx,
                        "variant": variant_name,
                        "soc": soc_t1,
                        "temp": temp_t1,
                        "vt0": vt_t0,
                        "vt1": vt_t1,
                        "vt_scale": vt_scale
                    })

                    soc_indices = []
                else:
                    loss_array[i] = 0
                    std_array[i] = last_std
                    soc_indices.append(idx)
                    delta_std_array[i] = 0


                pbar.update(1)

        # Append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            print(f"Appended {len(logs)} rows -> {os.path.basename(log_file)}")

        return loss_array, std_array, vt_t0, cumulative_loss_pct, last_std, vt_trace, det_loss_array,delta_std_array



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

    def mu_epistemic(self, x, model_particle_idx):

        mu = np.empty(x.shape[0], dtype=np.float32)

        for k, idx in model_particle_idx.items():
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

        return np.stack(preds, axis=0) / 100.0  # (B, M)

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
        epistemic_scale: float =1,# 1.0,
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


    def lfp_calendar_BattProDeep_vectorized(
        self,
        M,
        profile,
        soc_particles: np.ndarray,         # shape (M, timesteps)
        variant_name: str,
        years: int = 1,
        vt_scale: float = 1.0,
        window_size: int = 60,
        temperature: float = 25,
        temperature_type: str = "profile",
        temperature_column: str = "Temperature",
        max_spread0:float=0,
        virtual_time0: np.ndarray=0,  #exists for M particles
        cumulative_loss0: np.ndarray=0,  #exists for M particles
        last_std0: np.ndarray=0,  #fractional
        log_file: str = None,
        eps: np.ndarray | None = None,
        zeta: np.ndarray | None = None,  # epistemic scenario (per particle or per run)
        n_boot: int = 32,
        bootstrap_seed: int | None = 42,
        epistemic_mode: str = "anchored",
        epistemic_support_scale: float = 1.0,
        return_trace: bool = False,
        return_det: bool = False,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """
            Vectorized calendar aging computation for M particles.
            Returns:
                loss_array: (M, timesteps)
                vt_t0: final virtual time (scalar)
                cumulative_loss_pct: final cumulative calendar loss per particle (array)

            """


        if log_file is None:
            log_file = f"calendar_inputs_{variant_name}.pkl"


        num_steps = soc_particles.shape[1]
        if num_steps < 2:
            return (
                np.zeros((M, num_steps), dtype=np.float32),
                np.full(M, virtual_time0, dtype=np.float32),
                np.full(M, cumulative_loss0, dtype=np.float32),
                np.full(M, last_std0, dtype=np.float32),
                np.zeros((M, num_steps), dtype=np.float32) if return_trace else None,
                np.float32(max_spread0),
                np.zeros((M, num_steps), dtype=np.float32) if return_det else None,
            )



        # --- timestep resolution ---
        dt_minutes = (profile.index[1] - profile.index[0]).total_seconds() / 60.0
        elapsed_minutes = (profile.index - profile.index[0]).total_seconds() / 60.0

        # --- initial states ---
        vt0_particles = np.full(M, virtual_time0, dtype=np.float32)
        cumulative_loss_pct = np.array(cumulative_loss0, dtype=np.float32).reshape(M,)
        max_spread = np.float32(max_spread0)

        last_std = np.array(last_std0, dtype=np.float32).reshape(M,)
        loss_array = np.zeros((M, num_steps), dtype=np.float32)
        det_loss_particles_array = np.zeros((M, num_steps), dtype=np.float32) if return_det else None

        vt_trace = np.zeros((M, num_steps), dtype=np.float32) if return_trace else None
        logs = []
        if eps is None:
            print("vectorized version needs epsilon")
            eps = np.zeros(M, dtype=np.float32)
        else:
            eps = np.asarray(eps, dtype=np.float32).reshape(M,)
        if zeta is None:
            zeta = np.zeros(M, dtype=np.float32)
        else:
            zeta = np.asarray(zeta, dtype=np.float32).reshape(M,)
        window_start_soc_indices = 0 # The current calendar-aging window starts at this timestep.
        with tqdm(total=num_steps, desc="Calculating Calendar Aging (BattProDeep)") as pbar:


            #for i in range(num_steps):
            for i, idx in enumerate(profile.index):


                if vt_trace is not None:
                    vt_trace[:, i] = vt0_particles
                #soc_indices.append(idx) #moved to else branch

                is_boundary = ((elapsed_minutes[i] % window_size == 0) and i > 0) or (i == num_steps - 1)
                if is_boundary:
                    soc_window = soc_particles[:, window_start_soc_indices:i+1]  # shape (M, window_len)
                    soc_t1 = soc_window.mean(axis=1)
                    window_len = i + 1 - window_start_soc_indices #Number of timesteps in the current window
                    window_start_soc_indices = i + 1 # Reset; next calendar-aging window starts after this boundary.
                    calendar_time_indices = profile.index[     window_start_soc_indices - window_len : window_start_soc_indices]



                    if temperature_type == "constant":
                        temp_t1_vec = np.full(M, temperature, dtype=np.float32)

                    elif temperature_type == "profile":
                        # indices corresponding to the SOC window


                        temp_t1 = self.weighted_average_temperature(
                            profile=profile,
                            soc_indices=calendar_time_indices ,
                            temperature_column=temperature_column,
                        )
                        temp_t1_vec = np.full(M, temp_t1, dtype=np.float32)

                    else:
                        raise ValueError(f"Invalid temperature_type: {temperature_type}")


                    vt_t1_particles = vt0_particles + np.float32(window_len * (dt_minutes / 60.0) * vt_scale)
                    x0 = np.stack([vt0_particles, temp_t1_vec, soc_t1 * 100.0], axis=1).astype(np.float32, copy=False)
                    if not np.all(np.isfinite(x0)):
                        print("INVALID x0 DETECTED")
                        print(x0)
                        raise ValueError("Bad input before scaling")
                    x1 = np.stack([vt_t1_particles, temp_t1_vec, soc_t1 * 100.0], axis=1).astype(np.float32, copy=False)

                    x0 = scale_data(None, x0.reshape(M,1,-1), path=self.normalizer_path_calendar, val_test_flag=True)
                    x1 = scale_data(None, x1.reshape(M,1,-1), path=self.normalizer_path_calendar, val_test_flag=True)
                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        det_pair = self.model_det.predict(
                            np.concatenate([x0, x1], axis=0),
                            verbose=False,
                        ).reshape(-1) / 100.0
                        mu_t0 = det_pair[:M]
                        mu_t1 = det_pair[M:]
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
                    elif epistemic_mode == "full":
                        f_t0 = self.mu_epistemic_fast(x0, zeta, n_boot=n_boot, subset_seed=bootstrap_seed)
                        f_t1 = self.mu_epistemic_fast(x1, zeta, n_boot=n_boot, subset_seed=bootstrap_seed)
                        delta_mu = f_t1 - f_t0
                    elif epistemic_mode in (
                        "full_centered",
                        "full_centered_rel",
                        "full_centered_minchange",
                        "full_centered_minchange_reference",
                        "full_centered_minchange_reference_refsigma",
                        "full_centered_support",
                    ):
                        det_pair = self.model_det.predict(
                            np.concatenate([x0, x1], axis=0),
                            verbose=False,
                        ).reshape(-1) / 100.0
                        det_t0 = det_pair[:M]
                        det_t1 = det_pair[M:]
                        delta_det = np.maximum(det_t1 - det_t0, 0.0).astype(np.float32, copy=False)
                        if epistemic_mode in (
                            "full_centered_minchange_reference",
                            "full_centered_minchange_reference_refsigma",
                        ):
                            delta_mu = self.reference_centered_bootstrap_increment(
                                x0,
                                x1,
                                delta_det,
                                zeta,
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                        else:
                            delta_mu = self.centered_bootstrap_increment(
                                x0,
                                x1,
                                delta_det,
                                zeta,
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
                        det_t0 = det_pair[:M]
                        det_t1 = det_pair[M:]
                        delta_det = np.maximum(det_t1 - det_t0, 0.0).astype(np.float32, copy=False)

                        x0_ref_raw = np.array(
                            [[vt0_particles.mean(), temp_t1_vec.mean(), soc_t1.mean() * 100.0]],
                            dtype=np.float32,
                        )
                        x1_ref_raw = np.array(
                            [[vt_t1_particles.mean(), temp_t1_vec.mean(), soc_t1.mean() * 100.0]],
                            dtype=np.float32,
                        )
                        x0_ref = scale_data(
                            None,
                            x0_ref_raw.reshape(1, 1, -1),
                            path=self.normalizer_path_calendar,
                            val_test_flag=True,
                        )
                        x1_ref = scale_data(
                            None,
                            x1_ref_raw.reshape(1, 1, -1),
                            path=self.normalizer_path_calendar,
                            val_test_flag=True,
                        )
                        epi_factors = self.reference_epistemic_factors(
                            x0_ref,
                            x1_ref,
                            n_boot=n_boot,
                            subset_seed=bootstrap_seed,
                        )
                        zeta_mod = zeta.astype(np.int32) % epi_factors.shape[0]
                        delta_mu = delta_det * epi_factors[zeta_mod]
                    else:
                        raise ValueError(f"Unknown epistemic_mode: {epistemic_mode}")
                    # ----------------------------
                    # aleatoric uncertainty
                    # ----------------------------
                    if epistemic_mode in (
                        "full_centered_minchange_reference_refsigma",
                        "sigma_e_reference_refsigma",
                    ):
                        _, sigma_pair_ref = self._probabilistic_mean_std(
                            np.concatenate([x0[:1], x1[:1]], axis=0)
                        )
                        sigma_pair_ref = sigma_pair_ref / 100.0
                        sigma_a_t0 = np.full(M, sigma_pair_ref[0], dtype=np.float32)
                        sigma_a_t1 = np.full(M, sigma_pair_ref[1], dtype=np.float32)
                    else:
                        _, sigma_pair = self._probabilistic_mean_std(
                            np.concatenate([x0, x1], axis=0)
                        )
                        sigma_pair = sigma_pair / 100.0
                        sigma_a_t0 = sigma_pair[:M]
                        sigma_a_t1 = sigma_pair[M:]

                    # ----------------------------
                    # epistemic uncertainty (bootstrap)
                    # ----------------------------
                    #alternatively
                    # sigma_e_t0 = self.epistemic_std(x0)
                    # sigma_e_t1 = self.epistemic_std(x1)
                    ### full epistemic
                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        if epistemic_mode == "sigma_e_reference_refsigma":
                            _, sigma_e_pair_ref = self.epistemic_stats(
                                np.concatenate([x0[:1], x1[:1]], axis=0),
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                            sigma_e_pair_ref = sigma_e_pair_ref / 100.0
                            sigma_e_t0 = np.full(M, sigma_e_pair_ref[0], dtype=np.float32)
                            sigma_e_t1 = np.full(M, sigma_e_pair_ref[1], dtype=np.float32)
                        else:
                            _, sigma_e_pair = self.epistemic_stats(
                                np.concatenate([x0, x1], axis=0),
                                n_boot=n_boot,
                                subset_seed=bootstrap_seed,
                            )
                            sigma_e_pair = sigma_e_pair / 100.0
                            sigma_e_t0 = sigma_e_pair[:M]
                            sigma_e_t1 = sigma_e_pair[M:]
                    # #epistemic only as mean
                    # # x1_mean = np.mean(x1, axis=0, keepdims=True)
                    # # _, sigma_e_t1 = self.epistemic_stats(x1_mean)
                    # # sigma_e_t1 = np.full(M, sigma_e_t1 / 100.0)
                    # # x0_mean = np.mean(x0, axis=0, keepdims=True)
                    # # _, sigma_e_t0 = self.epistemic_stats(x0_mean)
                    # # sigma_e_t0 = np.full(M, sigma_e_t0 / 100.0)
                    # # =========================================================
                    # # 3) OPTION 3: UNCERTAINTY GROWTH MODEL
                    # # =========================================================
                    sigma_growth_a = np.maximum(sigma_a_t1 - sigma_a_t0, 0.0)
                    if epistemic_mode in ("sigma_e", "sigma_total", "sigma_e_reference_refsigma"):
                        sigma_growth_e = np.maximum(sigma_e_t1 - sigma_e_t0, 0.0)

                    # sigma_delta = np.sqrt(
                    #     (eps**2) *sigma_growth_a**2 + sigma_growth_e**2
                    # )
                    if epistemic_mode == "sigma_total":
                        sigma_delta = np.sqrt(sigma_growth_a**2 + sigma_growth_e**2)
                    else:
                        sigma_delta = sigma_growth_a

                    # =========================================================
                    # 5) stochastic increment (single injection)
                    # =========================================================
                    #delta = delta_mu +  sigma_delta
                    # = np.sign(eps)
                    delta = delta_mu + eps * sigma_delta
                    if epistemic_mode in ("sigma_e", "sigma_e_reference_refsigma"):
                        delta = delta + float(epistemic_support_scale) * zeta * sigma_growth_e
                    if epistemic_mode in (
                        "full_centered",
                        "full_centered_rel",
                        "full_centered_minchange",
                        "full_centered_minchange_reference",
                        "full_centered_minchange_reference_refsigma",
                        "full_centered_support",
                    ) and M > 0:
                        delta[0] = delta_mu[0]

                    # =========================================================
                    # 6) enforce physical constraints
                    # =========================================================
                    delta = np.maximum(delta, 0.0)

                    # optional: keep ensemble mass stable (recommended if needed)
                    # delta = delta / (np.mean(delta) + 1e-12) * np.mean(delta)

                    # =========================================================
                    # 7) output
                    # =========================================================
                    loss_particles = delta
                    det_loss_particles = delta_mu



                    # --- update arrays ---
                    loss_array[:, i] = loss_particles
                    if return_det:
                        det_loss_particles_array[:, i] = det_loss_particles

                    cumulative_loss_pct += loss_particles*100 #M particlse,

                    last_std=sigma_a_t1

                    # --- update virtual time continuously ---
                    vt_in = np.stack([cumulative_loss_pct, temp_t1_vec, soc_t1 * 100.0], axis=1).astype(np.float32, copy=False)
                    #print(vt_in.shape)#(100, 3)
                    vt_in = scale_data(None, vt_in.reshape(M, 1, 3), # (M, 1, 3): particles, one timestep, three features
                                    path=self.normalizer_path_calendar, is_inverse=True, val_test_flag=True)
                    #print(vt_in.shape) #(100, 1, 3)
                    vt_pred_norm_particles = self.model_virtual.predict(vt_in,verbose=False).reshape(-1,1)
                    #print(vt_pred_norm_particles.shape) #(100, 1)

                    vt0_particles = inverse_scale_values(
                        vt_pred_norm_particles,
                        path=self.normalizer_path_calendar,
                        is_inverse=True,
                        is_label=True,
                    ).astype(np.float32, copy=False)
                    #print(vt0_particles.shape) #()






                    if vt_trace is not None:
                        vt_trace[:, i] = vt0_particles
                    # --- store logs ---
                    logs.append({
                        "timestep": idx,#i,
                        "variant": variant_name,
                        "soc": soc_t1.mean(axis=0),
                        "temp": temp_t1_vec.mean(),
                        "vt0": vt0_particles.mean(axis=0),
                        "last_std1":last_std.mean(axis=0),
                        #"last_std_e1":sigma_e_t1.mean(axis=0),
                        "vt_scale": vt_scale,
                        "loss_inc":loss_particles.mean(axis=0),
                        #"L1":mu_t1.mean(axis=0),#L1.mean(axis=0),
                        "cum_loss_pct":cumulative_loss_pct.mean(axis=0),
                        "cum_loss_pct_high": np.percentile(cumulative_loss_pct, 95, axis=0) ,
                        "cum_loss_pct_low": np.percentile(cumulative_loss_pct, 5, axis=0),
                    })

                #ou already initialized loss_array = np.zeros((M, num_steps)), so no else is neccesary



                pbar.update(1)

        # Append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            #print(f"Appended {len(logs)} rows -> {os.path.basename(log_file)}")

        return (
            loss_array.astype(np.float32, copy=False),
            vt0_particles.astype(np.float32, copy=False),
            cumulative_loss_pct.astype(np.float32, copy=False),
            last_std.astype(np.float32, copy=False),
            vt_trace.astype(np.float32, copy=False) if vt_trace is not None else None,
            max_spread,
            det_loss_particles_array.astype(np.float32, copy=False) if return_det else None,
        )




    def lfp_calendar_Naumann(self, profile: pd.DataFrame, years: int = 1, temperature: float = 25) -> np.ndarray:
        """
        Calculates aging using the Naumann method.

        Args:
            profile (pd.DataFrame): SOC profile dataframe.
            years (int, optional): Number of years to iterate the SOC profile. Defaults to 1.
            temperature (float, optional): Cell temperature profile. Defaults to 25.

        Returns:
            np.ndarray: Array of capacity loss for each timestep.
        """
        num_steps = len(profile)
        dt = (profile.index[1] - profile.index[0]).seconds
        C_QLOSS = 2.8575
        D_QLOSS = 0.60225
        temp = temperature + 273.15
        Eact = 17126
        R = 8.314
        k_ref = 1.2571e-5
        k_temp_base = k_ref * math.exp(-(Eact / R) * ((1 / temp) - (1 / 298.15)))

        cumulative_loss = 0
        loss_array = np.zeros(num_steps * years)

        # Add progress bar using tqdm
        with tqdm(total=num_steps * years, desc="Calculating Calendar Aging (Naumann)") as pbar:
            for year in range(years):
                for i, idx in enumerate(profile.index):
                    t = i + num_steps * year
                    soc = profile.loc[idx, "soc"]
                    k_soc = C_QLOSS * (soc - 0.5) ** 3 + D_QLOSS
                    k = k_temp_base * k_soc
                    virtual_time = (cumulative_loss / k) ** 2
                    loss = k_soc * k_temp_base * math.sqrt(virtual_time + dt) - cumulative_loss
                    loss_array[t] = loss
                    cumulative_loss += loss

                    pbar.update(1)  # Update the progress bar

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

    def preprocess_data(self, data: dict, split_percentages: list = None):
        """
        Preprocesses data for training or analysis.

        Args:
            data (dict): The data to be processed.
            split_percentages (list, optional): The percentages of the data to be used for training, validation, and
            testing. Defaults to [0.95, 0, 0.05].
        """
        df_calendar = data['calendar']
        if split_percentages is None:
            split_percentages = [0.95, 0, 0.05]
        print("Preprocessing data.")

        self.conditions = extract_conditions(df_calendar, degradation_type=self.degradation_type,
                                             complete_data=self.complete_data)
        self.train_conditions, self.dev_conditions, self.test_conditions = shuffle_and_split(self.conditions,
                                                                                             split_percentages)

        self.input_train, self.labels_train = preprocess_data(df_calendar,
                                                              self.train_conditions,
                                                              df_calendar,
                                                              degradation_type=self.degradation_type,
                                                              complete_data=self.complete_data)
        self.input_test, self.labels_test = preprocess_data(df_calendar,
                                                            self.test_conditions,
                                                            df_calendar,
                                                            degradation_type=self.degradation_type,
                                                            complete_data=self.complete_data)

        scaler = MinMaxScaler()
        self.input_train_scaled = scale_data(scaler, self.input_train, path=self.normalizer_path_calendar)
        self.input_test_scaled = scale_data(scaler, self.input_test, path=self.normalizer_path_calendar,
                                            val_test_flag=True)

        self.df_train_inverse = create_inverse_df(self.input_train, self.labels_train,
                                                  degradation_type=self.degradation_type)
        self.df_test_inverse = create_inverse_df(self.input_test, self.labels_test,
                                                 degradation_type=self.degradation_type)

        self.df_train_inverse_scaled = create_inverse_df(self.input_train_scaled, self.labels_train,
                                                         degradation_type=self.degradation_type)
        self.df_test_inverse_scaled = create_inverse_df(self.input_test_scaled, self.labels_test,
                                                        degradation_type=self.degradation_type)

        if self.dev_conditions:
            self.input_dev, self.labels_dev = preprocess_data(df_calendar,
                                                              self.dev_conditions,
                                                              df_calendar,
                                                              degradation_type=self.degradation_type,
                                                              complete_data=self.complete_data)
            self.input_dev_scaled = scale_data(scaler, self.input_dev, path=self.normalizer_path_calendar,
                                               val_test_flag=True)
            self.df_dev_inverse = create_inverse_df(self.input_dev, self.labels_dev,
                                                    degradation_type=self.degradation_type)
            self.df_dev_inverse_scaled = create_inverse_df(self.input_dev_scaled, self.labels_dev,
                                                           degradation_type=self.degradation_type)

        self.inputs, self.labels = preprocess_data(df_calendar,
                                                   self.conditions,
                                                   df_calendar,
                                                   degradation_type=self.degradation_type,
                                                   complete_data=self.complete_data)
        self.inputs_scaled = scale_data(scaler, self.inputs, path=self.normalizer_path_calendar, val_test_flag=True)

        self.df_inverse = create_inverse_df(self.inputs, self.labels, degradation_type=self.degradation_type)
        self.df_inverse_scaled = create_inverse_df(self.inputs_scaled, self.labels,
                                                   degradation_type=self.degradation_type)

        self.X_train = np.array(self.input_train_scaled.tolist(), dtype=np.float32)
        self.Y_train = np.array(self.labels_train.tolist(), dtype=np.float32)

        if self.dev_conditions:
            self.X_val = np.array(self.input_dev_scaled.tolist(), dtype=np.float32)
            self.Y_val = np.array(self.labels_dev.tolist(), dtype=np.float32)

        self.X_test = np.array(self.input_test_scaled.tolist(), dtype=np.float32)
        self.Y_test = np.array(self.labels_test.tolist(), dtype=np.float32)
