import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tqdm import tqdm
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
        parent_directory = str(Path(os.getcwd()).parent.absolute())
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
        self.model_folder = os.path.join(self.model_path_calendar, kwargs.get('model_relative_path',
                                                                              os.path.join('probabilistic',
                                                                                           '01-aleatoric')))

    def _load_models(self):
        """Loads trained models if specified during initialization."""
        self.model_virtual = self.get_trained_model(self.model_folder_virtual)
        self.model_det = self.get_trained_model(self.model_folder_det)
        self.model_prob = self.get_trained_model(self.model_folder,
                                                 custom_objects={'IndependentNormal': tfpl.IndependentNormal,
                                                                 'negative_log_likelihood': negative_log_likelihood}) #

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


    def weighted_average_temperature(self, profile: pd.DataFrame,  soc_indices: list, temperature_column: str = "Max. Cell Temperature in °C") -> float:
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
            weight = np.exp(temperature / 25)  # arrhenius based scaling factor: A temperature of 25°C would give a weight of 𝑒**(25/25)=2.718. 
            #A temperature of 50°C would give a weight of 𝑒**(50/25)=7.389. This shows how the weight increases much faster for higher temperatures.
            
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
        Calendar aging model (BattProDeep) – looped-safe version.
        ✅ Keeps virtual-time continuity across blocks.
        ✅ Scales total exposure using vt_scale (global 'years' multiplier).
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

                    # 🔹 scaled time increment
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
                    
                    stddev_frac = self.model_prob(x1).stddev().numpy().reshape(-1)[0] / 100

                    
                    ####
                    stddev_t0_frac = self.model_prob(x0).stddev().numpy().reshape(-1)[0] / 100
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

        # ✅ append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            print(f"🗂️ Appended {len(logs)} rows → {os.path.basename(log_file)}")

        return loss_array, std_array, vt_t0, cumulative_loss_pct, last_std, vt_trace, det_loss_array,delta_std_array
    

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
        virtual_time0: np.ndarray=0,  #exists for M particles
        cumulative_loss0: np.ndarray=0,  #exists for M particles
        last_std0: np.ndarray=0,  #fractional
        log_file: str = None,
        eps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """
            Vectorized calendar aging computation for M particles.
            Returns:
                loss_array: (M, timesteps)
                vt_t0: final virtual time (scalar)
                cumulative_loss_pct: final cumulative calendar loss per particle (array)
                vt_trace: (M, timesteps) virtual time trace per particle
            """

        if log_file is None:
            log_file = f"calendar_inputs_{variant_name}.pkl"
        

        num_steps = soc_particles.shape[1]
        if num_steps < 2:
            return (
                np.zeros((M, num_steps)),
                np.full(M, virtual_time0),
                np.full(M, cumulative_loss0),
                np.zeros(num_steps),
            )

            
        
        # --- timestep resolution ---
        dt_minutes = (profile.index[1] - profile.index[0]).total_seconds() / 60.0
        elapsed_minutes = (profile.index - profile.index[0]).total_seconds() / 60.0

        # --- initial states ---
        vt0_particles = np.full(M, virtual_time0, dtype=float) #is 0 initially too) how to declare it here
        cumulative_loss_pct = np.array(cumulative_loss0, dtype=float).reshape(M,) 
        
        
        last_std=np.array(last_std0, dtype=float).reshape(M,) 
        loss_array = np.zeros((M, num_steps))
        
        vt_trace = np.zeros((M, num_steps))
        logs = []
        window_start_soc_indices = 0 #The current calendar-aging window starts at timestep window_start_soc_indices.”
        with tqdm(total=num_steps, desc="Calculating Calendar Aging (BattProDeep)") as pbar:
            

            #for i in range(num_steps):
            for i, idx in enumerate(profile.index):
                
                
                vt_trace[:,i] = vt0_particles # just for tracking, append always the updated virtual time
                #soc_indices.append(idx) #moved to else branch

                is_boundary = (elapsed_minutes[i] % window_size == 0) or (i == num_steps - 1)
                if is_boundary:
                    soc_window = soc_particles[:, window_start_soc_indices:i+1]  # shape (M, window_len) “All SOC values for all particles from the start of the current window up to now.”
                    soc_t1 = soc_window.mean(axis=1)
                    window_len = i + 1 - window_start_soc_indices #Number of timesteps in the current window
                    window_start_soc_indices = i + 1 #reset. The next calendar-aging window starts after this boundary.”
                    calendar_time_indices = profile.index[     window_start_soc_indices - window_len : window_start_soc_indices]


                    
                    if temperature_type == "constant":
                        temp_t1_vec = np.full(M, temperature)

                    elif temperature_type == "profile":
                        # indices corresponding to the SOC window
                        

                        temp_t1 = self.weighted_average_temperature(
                            profile=profile,
                            soc_indices=calendar_time_indices ,
                            temperature_column=temperature_column,
                        )
                        temp_t1_vec = np.full(M, temp_t1)

                    else:
                        raise ValueError(f"Invalid temperature_type: {temperature_type}")
                    
                    # # --- HARD BO L CONDITION ---
                    # is_bol = cumulative_loss_pct <= 0.0 + 1e-12  # float-safe

                    # if np.all(is_bol):
                    #     vt0_particles = np.zeros(M)
                    #     vt_t1_particles = vt0_particles+ window_len * (dt_minutes / 60.0) * vt_scale
                        

                    # else: 
                    #     # 🔹 scaled time increment
                    #     vt_in = np.stack( [cumulative_loss_pct, temp_t1_vec, soc_t1 * 100.0],  axis=1)  # shape (M, 3)
                    #     #print(vt_in.shape)#(100, 3)
                    #     vt_in = scale_data(None, vt_in.reshape(M, 1, 3), #→ (M, 1, 3) M samples (particles), 1 timestep (current window), 3 features (cumulative_loss, temp, soc)
                    #                     path=self.normalizer_path_calendar, is_inverse=True, val_test_flag=True)
                    #     #print(vt_in.shape) #(100, 1, 3)
                    #     vt_pred_norm_particles = self.model_virtual.predict(vt_in,verbose=False).reshape(-1,1)

                    #     def inverse_scale_particles(sequences_scaled: np.ndarray, path: str, is_label: bool = False,
                    #             is_inverse: bool = False) -> np.ndarray:
                    #         """
                    #         Like inverse_scale but returns all values (array) instead of a single scalar.
                    #         """
                    #         if is_label:
                    #             identifier = 'labels_inverse' if is_inverse else 'labels'
                            
                    #         scaler_files = os.listdir(path)
                    #         scaler_file = os.path.join(path,
                    #                                 sorted([f for f in scaler_files if f.startswith(f'scalar_{identifier}')], reverse=True)[0])

                    #         loaded_scaler = joblib.load(scaler_file)
                    #         inverse_value = loaded_scaler.inverse_transform(sequences_scaled)

                    #         return inverse_value.reshape(-1)  # <-- no [0], keep all values 
                    #     vt0_particles= inverse_scale_particles(vt_pred_norm_particles, path=self.normalizer_path_calendar,
                    #                             is_inverse=True, is_label=True)#.reshape(-1)  # now shape (M,)

                    #     vt_t1_particles = vt0_particles + window_len * (dt_minutes / 60.0) * vt_scale  # hours #needs to use M particles for Virtoual time 0

                    # # --- prepare scaled inputs ---
                    # #print(vt0_particles.shape, temp_t1_vec.shape, soc_t1.shape)
                    # # sanitize virtual time
                    # vt0_particles = np.array(vt0_particles, dtype=float)  # ensure float

                    # # detect invalid values (NaN or Inf)
                    # invalid_mask = ~np.isfinite(vt0_particles)  # True where NaN or Inf

                    # if np.any(invalid_mask):
                    #     # compute mean of valid particles
                    #     valid_mean = vt0_particles[np.isfinite(vt0_particles)].mean() if np.any(np.isfinite(vt0_particles)) else 0.0

                    #     # set NaN to 0, Inf to mean of valid particles
                    #     vt0_particles[np.isnan(vt0_particles)] = 0.0
                    #     vt0_particles[np.isinf(vt0_particles)] = valid_mean
                    # if np.any(np.isnan(vt0_particles)) or np.any(np.isinf(vt0_particles)):
                    #     print("DEBUG: vt0_particles contains invalid numbers!")
                    #     print(vt0_particles)
                    # if np.any(np.isnan(soc_t1)):
                    #     print("DEBUG: soc contains invalid numbers!")
                    #     print(vt0_particles)
                    # # print(vt0_particles)
                    # # print(vt0_particles.shape)
                    # # print(vt_t1_particles)
                    # # print(vt_t1_particles.shape)
                    # # print(soc_t1)
                    # # print(soc_t1.shape)
                    vt_t1_particles = vt0_particles + window_len * (dt_minutes / 60.0) * vt_scale  # hours #needs to use M particles for Virtoual time 0
                    x0 = np.stack([vt0_particles, temp_t1_vec, soc_t1*100.0], axis=1)
                    x1 = np.stack([vt_t1_particles, temp_t1_vec, soc_t1*100.0], axis=1) # virtual time needs to be per particle!

                    x0 = scale_data(None, x0.reshape(M,1,-1), path=self.normalizer_path_calendar, val_test_flag=True)
                    x1 = scale_data(None, x1.reshape(M,1,-1), path=self.normalizer_path_calendar, val_test_flag=True)

                    loss_t0_pct = self.model_det.predict(x0, verbose=False).reshape(-1)  # shape (M,)
                    loss_t1_pct = self.model_det.predict(x1, verbose=False).reshape(-1) # shape (M,)
                    
                    stddev_frac = self.model_prob(x1).stddev().numpy().reshape(-1)/ 100# shape (M,) 
                    #If M > 1 and output shape (M, 1), .reshape(-1)[0] only picks particle 0.You lose all other M-1 particle predictions.
                    #For multiple particles, the correct way is .squeeze(axis=-1) or [:,0]: .reshape(-1)
                    if eps is None:
                        print("vectorized version needs epsilon")
                    else:
                        eps = np.asarray(eps).reshape(M,)
                    
                    ####
                    stddev_t0_frac = self.model_prob(x0).stddev().numpy().reshape(-1)/ 100  # shape (M,) 
                    
                    #loss_particles=( loss_t1_pct -loss_t0_pct)/100
                    delta_stddev = np.sqrt(np.maximum(stddev_frac**2 - stddev_t0_frac**2, 0.0))
                    ##loss_particles = loss_particles + eps * delta_stddev # vectorized like earlier in cycle script
                    L0 = loss_t0_pct + eps * (stddev_t0_frac*100)
                    #L0=cumulative_loss_pct + eps * (last_std*100) #vt troubleshoot
                    L1 = loss_t1_pct + eps * (stddev_frac*100)
                    #loss_particles = np.maximum(((L1 - L0)/100),0) #Instead of clamping ΔL_i ≥ 0, shift all particles at each timestep so that the ensemble mean incremental loss matches the deterministic Δμ_i.
                    #make sure deterministic mean loss is met even when σ_i < σ_{i-1} 11.02.
                    det_delta_loss_ptc=(loss_t1_pct-loss_t0_pct)
                    ## prevent shrinking CI 13.02.
                    delta_std = stddev_frac*100 - stddev_t0_frac*100
                    delta_std = np.maximum(delta_std, 0) 
                    # particle incremental loss
                    delta_L_particles_ptc = det_delta_loss_ptc + eps * delta_std
                    loss_particles =np.maximum((delta_L_particles_ptc/100),0)
                    ####13.02.
                    #delta_L_particles_ptc = L1 - L0  
                    #mean_particles_ptc = np.mean(delta_L_particles_ptc)
                    #loss_particles =np.maximum(((delta_L_particles_ptc+np.maximum((det_delta_loss_ptc - mean_particles_ptc),0))/100),0) #only correct if the difference is positive
                    #####
                    
                    
                    # --- update arrays ---
                    loss_array[:, i] = loss_particles

                    
                    cumulative_loss_pct += loss_particles*100 #M particlse, 
                    last_std=stddev_frac

                    # --- update virtual time continuously ---
                    vt_in = np.stack( [cumulative_loss_pct, temp_t1_vec, soc_t1 * 100.0],  axis=1)  # shape (M, 3)
                    #print(vt_in.shape)#(100, 3)
                    vt_in = scale_data(None, vt_in.reshape(M, 1, 3), #→ (M, 1, 3) M samples (particles), 1 timestep (current window), 3 features (cumulative_loss, temp, soc)
                                    path=self.normalizer_path_calendar, is_inverse=True, val_test_flag=True)
                    #print(vt_in.shape) #(100, 1, 3)
                    vt_pred_norm_particles = self.model_virtual.predict(vt_in,verbose=False).reshape(-1,1)
                    #print(vt_pred_norm_particles.shape) #(100, 1)

                    def inverse_scale_particles(sequences_scaled: np.ndarray, path: str, is_label: bool = False,
                            is_inverse: bool = False) -> np.ndarray:
                        """
                        Like inverse_scale but returns all values (array) instead of a single scalar.
                        """
                        if is_label:
                            identifier = 'labels_inverse' if is_inverse else 'labels'
                        
                        scaler_files = os.listdir(path)
                        scaler_file = os.path.join(path,
                                                sorted([f for f in scaler_files if f.startswith(f'scalar_{identifier}')], reverse=True)[0])

                        loaded_scaler = joblib.load(scaler_file)
                        inverse_value = loaded_scaler.inverse_transform(sequences_scaled)

                        return inverse_value.reshape(-1)  # <-- no [0], keep all values 
                    vt0_particles= inverse_scale_particles(vt_pred_norm_particles, path=self.normalizer_path_calendar,
                                            is_inverse=True, is_label=True)#.reshape(-1)  # now shape (M,)
                    #print(vt0_particles.shape) #()
                    
                    


                    
                    
                    vt_trace[:,i] = vt0_particles
                    # --- store logs ---
                    logs.append({
                        "timestep": idx,#i,
                        "variant": variant_name,
                        "soc": soc_t1.mean(axis=0),
                        "temp": temp_t1_vec.mean(),
                        "vt0": vt0_particles.mean(axis=0),
                        "last_std1":last_std.mean(axis=0),
                        "vt_scale": vt_scale
                    })

                #ou already initialized loss_array = np.zeros((M, num_steps)), so no else is neccesary
                    
                        

                pbar.update(1)

        # ✅ append logs safely
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                old = pd.read_pickle(log_file)
                df_logs = pd.concat([old, df_logs], ignore_index=True)
            df_logs.to_pickle(log_file)
            #print(f"🗂️ Appended {len(logs)} rows → {os.path.basename(log_file)}")

        return loss_array,  vt0_particles, cumulative_loss_pct,  vt_trace, last_std
    
    def lfp_calendar_BattProDeep_particles(
        self,
        profile: pd.DataFrame,
        variant_name: str,
        M: int,
        years: int = 1,
        vt_scale: float = 1.0,
        window_size: int = 60,
        temperature: float = 25,
        temperature_type: str = "constant",
        temperature_column: str = "Temperature",
        virtual_time0: float = 0.0,
        cumulative_loss0: float = 0.0,
        last_std0: float = 0.0,
        log_file: str | None = None,
        eps_particles: np.ndarray = None,
        seed: int = 42,
    ):
        """
        Particle-based stochastic calendar aging.
        Fully equivalent to lfp_calendar_BattProDeep, but with M particles.
        """

        rho=0.995   # correlation strength (hours–days memory)
        rng = np.random.default_rng(seed)

        if log_file is None:
            log_file = f"calendar_inputs_{variant_name}.pkl"

        num_steps = len(profile)
        dt_minutes = (profile.index[1] - profile.index[0]).total_seconds() / 60.0
        elapsed_minutes = (profile.index - profile.index[0]).total_seconds() / 60.0

        # --- particle states ---
        cumulative_loss = np.full(M, cumulative_loss0, dtype=float)
        #vt = np.full(M, virtual_time0, dtype=float)
        virtual_time_t0 = np.full(M, virtual_time0, dtype=float)
        last_std = np.full(M, last_std0, dtype=float)
        det_cumulative_loss = cumulative_loss0
        det_loss_array = np.zeros(num_steps)

        # --- outputs ---
        loss_particles = np.zeros((M, num_steps))
        std_particles = np.zeros((M, num_steps))
        vt_trace = np.zeros((M, num_steps))

        soc_indices = []
        logs = []

        for i, idx in enumerate(profile.index):
            soc_indices.append(idx)

            #vt_trace[:, i] = vt
            vt_trace[:, i] = virtual_time_t0
            #std_particles[:, i] = last_std

            is_boundary = (elapsed_minutes[i] % window_size == 0)
            if (is_boundary and i > 0) or (i == num_steps - 1):

                # --- SOC ---
                soc_t = profile.loc[soc_indices, "soc"].mean()

                # --- temperature selection ---
                if temperature_type == "constant":
                    temp_t = temperature
                elif temperature_type == "profile":
                    temp_t = self.weighted_average_temperature(
                        profile, soc_indices, temperature_column
                    )
                else:
                    raise ValueError(f"Invalid temperature_type: {temperature_type}")

                # --- virtual time increment ---
                #vt_t1 = vt + len(soc_indices) * (dt_minutes / 60.0) * vt_scale
                delta = len(soc_indices) * (dt_minutes / 60.0) * vt_scale
                virtual_time_t1 = virtual_time_t0 + delta   # <-- SAME AS OLD CODE
                # --- ensure vt and vt_t1 are M-dimensional arrays ---
                # vt = np.atleast_1d(vt)
                # if vt.size == 1:
                #     vt = np.full(M, vt[0])

                # vt_t1 = np.atleast_1d(vt_t1)
                # if vt_t1.size == 1:
                #     vt_t1 = np.full(M, vt_t1[0])

                # --- prepare NN inputs (vectorized over particles) ---
                # x0 = np.stack([vt, np.full(M, temp_t), np.full(M, soc_t * 100)], axis=1)
                # x1 = np.stack([vt_t1, np.full(M, temp_t), np.full(M, soc_t * 100)], axis=1)
                x0 = np.stack(  [np.full(M,virtual_time_t0), np.full(M, temp_t), np.full(M, soc_t * 100)], axis=1,  )
                x1 = np.stack( [np.full(M,virtual_time_t1), np.full(M, temp_t), np.full(M, soc_t * 100)],  axis=1,   )

                x0 = scale_data(None, x0.reshape(M, 1, 3),
                                path=self.normalizer_path_calendar, val_test_flag=True)
                x1 = scale_data(None, x1.reshape(M, 1, 3),
                                path=self.normalizer_path_calendar, val_test_flag=True)

                dist0 = self.model_prob(x0)
                dist1 = self.model_prob(x1)

                mu0 = dist0.mean().numpy().reshape(-1)
                mu1 = dist1.mean().numpy().reshape(-1)
                #deterministic model instead for better vt prediction
                det_mu0=self.model_det.predict(x0, verbose=False).reshape(-1)[0]
                det_mu1= self.model_det.predict(x1, verbose=False).reshape(-1)[0]
                det_delta_loss=(det_mu1-det_mu0)/100
                det_cumulative_loss += det_delta_loss
                det_loss_array[i] = det_delta_loss
                std0 = dist0.stddev().numpy().reshape(-1) / 100
                std1 = dist1.stddev().numpy().reshape(-1) / 100

                eps_particles = eps_particles.reshape(-1)  # ensure (M,)
                eps = eps_particles.copy()
                #eps = eps_particles
                # delta_loss = (mu1 + eps * std1 * 100) - (mu0 + eps * std0 * 100)
                # delta_loss= np.ravel(delta_loss/100)
                # loss_particles[:, i] = delta_loss
                # std_particles[:, i] = std1
                # cumulative_loss += delta_loss
                last_std = std1

                # --- incremental mean ---
                delta_mu = mu1 - mu0
                # --- incremental uncertainty ---
                delta_sigma = np.sqrt(np.maximum(std1**2 - std0**2, 0.0))
                # --- correlated noise update --- maybe add later
                #eps = rho * eps + np.sqrt(1.0 - rho**2) * rng.normal(size=M)
                # --- stochastic incremental loss ---
                delta_loss = delta_mu + eps * delta_sigma
                delta_loss = np.maximum(delta_loss, 0.0)  # physical constraint
                delta_loss /= 100.0
                # --- store ---
                loss_particles[:, i] = delta_loss
                std_particles[:, i] = std1 
                cumulative_loss += delta_loss

                # --- virtual time update (particle-wise) ---
                #vt_in = np.stack([cumulative_loss * 100, np.full(M, temp_t), np.full(M, soc_t * 100)], axis=1)
                #Unless your virtual-time model was explicitly trained with stochastic loss inputs, you should instead do:
                #vt_in = np.stack([     np.full(M, cumulative_loss.mean() * 100),      np.full(M, temp_t),    np.full(M, soc_t * 100)], axis=1) # <-- replicate scalar
                vt_in = np.stack([     np.full(M, det_cumulative_loss * 100),      np.full(M, temp_t),    np.full(M, soc_t * 100)], axis=1) # <-- ruse deterministic mean
                vt_in = scale_data(None, vt_in.reshape(M, 1, 3),
                                path=self.normalizer_path_calendar,
                                is_inverse=True, val_test_flag=True)

                vt_pred_norm = self.model_virtual.predict(vt_in, verbose=False).reshape(-1, 1)
                # vt = inverse_scale(vt_pred_norm,
                #                 path=self.normalizer_path_calendar,
                #                 is_inverse=True, is_label=True).reshape(-1)
                virtual_time_t0 = inverse_scale( vt_pred_norm,   path=self.normalizer_path_calendar,  is_inverse=True,  is_label=True,  ).reshape(-1)

                

                # --- logging (deterministic quantities only) ---
                logs.append({
                    "timestep": idx,
                    "variant": variant_name,
                    "soc": soc_t,
                    "temp": temp_t,
                    "cumulative_loss_mean": float(cumulative_loss.mean()),
                    # "vt_mean": float(vt.mean()),
                    # "vt_std": float(vt.std()),
                    "vt_mean": float(virtual_time_t0.mean()),
                    "vt_std": float(virtual_time_t0.std()),
                })
                
                soc_indices = []

        # --- save logs ---
        if logs:
            df_logs = pd.DataFrame(logs)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if os.path.exists(log_file):
                df_logs = pd.concat([pd.read_pickle(log_file), df_logs])
            df_logs.to_pickle(log_file)

        return loss_particles, std_particles, cumulative_loss, last_std, vt_trace, det_loss_array


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
        print(f"Loading trained model from {file_path}.")
        return load_latest_model(file_path, custom_objects)

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
