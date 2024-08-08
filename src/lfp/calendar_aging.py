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
                                                                 'nll': negative_log_likelihood})

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

    def lfp_calendar_BattProDeep(self, profile: pd.DataFrame, years: int = 1, window_size: int = 24,
                                 temperature: float = 25) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates aging using the BattProDeep method.

        Args:
            profile (pd.DataFrame): SOC profile dataframe.
            years (int): Number of years to iterate the SOC profile.
            window_size (int): Intervals for calculating the calendar aging.
            temperature (float): Cell temperature profile.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of capacity loss and its standard deviation for each timestep.
        """
        num_steps = len(profile)
        dt = (profile.index[1] - profile.index[0]).seconds / 3600
        delta = window_size * dt
        cumulative_loss = 0
        loss_array = np.zeros(num_steps * years)
        loss_array_stddev = np.zeros(num_steps * years)
        virtual_time_t0 = 0
        soc_indices = []
        stddev = 0

        # Add progress bar using tqdm
        with tqdm(total=num_steps * years, desc="Calculating Calendar Aging (BattProDeep)") as pbar:
            for year in range(years):
                for i, idx in enumerate(profile.index):
                    t = i + num_steps * year
                    if (t != 0 and t % window_size == 0) or (idx == profile.index[-1] and year == range(years)[-1]):
                        soc_t1 = profile.loc[soc_indices, "soc"].mean()
                        virtual_time_t1 = virtual_time_t0 + delta
                        new_data_t0 = np.array([virtual_time_t0, temperature, soc_t1 * 100])
                        new_data_t1 = np.array([virtual_time_t1, temperature, soc_t1 * 100])

                        new_data_t0 = scale_data(None, new_data_t0.reshape(-1, 1, new_data_t0.shape[0]),
                                                 path=self.normalizer_path_calendar, val_test_flag=True)
                        new_data_t1 = scale_data(None, new_data_t1.reshape(-1, 1, new_data_t1.shape[0]),
                                                 path=self.normalizer_path_calendar, val_test_flag=True)

                        loss_t0_mean = self.model_det.predict(new_data_t0, verbose=False).reshape(-1)[0]
                        loss_t1_mean = self.model_det.predict(new_data_t1, verbose=False).reshape(-1)[0]
                        loss = loss_t1_mean - loss_t0_mean
                        loss_array[t] = loss / 100
                        stddev = self.model_prob(new_data_t1).stddev().numpy().reshape(-1)[0]
                        loss_array_stddev[t] = stddev / 100
                        cumulative_loss += loss

                        new_data_t = np.array([cumulative_loss, temperature, soc_t1 * 100])
                        new_data_t = scale_data(None, new_data_t.reshape(-1, 1, new_data_t.shape[0]),
                                                path=self.normalizer_path_calendar, is_inverse=True, val_test_flag=True)
                        predicted_y_normalized = self.model_virtual.predict(new_data_t, verbose=False).reshape(-1, 1)
                        virtual_time_t0 = inverse_scale(predicted_y_normalized, path=self.normalizer_path_calendar,
                                                        is_inverse=True, is_label=True)

                        soc_indices = []
                    else:
                        loss_array[t] = 0
                        loss_array_stddev[t] = stddev / 100
                        soc_indices.append(idx)

                    pbar.update(1)  # Update the progress bar

        return loss_array, loss_array_stddev

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
                tfpl.IndependentNormal(event_shape=1)
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
