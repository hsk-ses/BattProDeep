import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.utils import resample
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
        parent_directory = str(Path(os.getcwd()).parent.absolute())
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
        self.model_folder = os.path.join(self.model_path_cycle, kwargs.get('model_relative_path',
                                                                              os.path.join('probabilistic',
                                                                                           '01-aleatoric')))

    def _load_models(self):
        """Loads trained models if specified during initialization."""
        self.model_virtual = self.get_trained_model(self.model_folder_virtual)
        self.model_det = self.get_trained_model(self.model_folder_det)
        self.model_prob = self.get_trained_model(self.model_folder,
                                                 custom_objects={'IndependentNormal': tfpl.IndependentNormal,
                                                                 'nll': negative_log_likelihood})

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

    def lfp_cycle_BattProDeep(self, profile: pd.DataFrame, years: int = 1, fec_window: int = 10,
                              temperature: float = 25) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates aging using the BattProDeep method.

        Args:
            profile (pd.DataFrame): SOC profile dataframe.
            years (int, optional): Number of years to iterate the SOC profile. Defaults to 1.
            fec_window (int, optional): Intervals for calculating the cycle aging. Defaults to 10.
            temperature (float, optional): Cell temperature profile. Defaults to 25.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of capacity loss and its standard deviation for each timestep.
        """
        num_steps = len(profile)
        cumulative_loss = 0.0
        loss_array = np.zeros(num_steps * years)
        loss_array_stddev = np.zeros(num_steps * years)
        soc_indices = []
        fec_t0, fec_t1, fec_year = 0, 0, 0
        stddev = 0.0

        # Add progress bar using tqdm
        with tqdm(total=num_steps * years, desc="Calculating Cyclic Aging (BattProDeep)") as pbar:
            for year in range(years):
                for i, idx in enumerate(profile.index):
                    fec_t1 = fec_year + profile.loc[idx, "FEC"]
                    delta_fec = fec_t1 - fec_t0

                    if (delta_fec >= fec_window) or (idx == profile.index[-1] and year == range(years)[-1]):
                        mask = profile.index.isin(soc_indices)
                        soc_t1 = profile.loc[soc_indices, "soc"].mean()
                        depth_of_charge_t1 = profile.loc[
                            mask & (profile["Half_Cycle_Depth"] > 0), "Half_Cycle_Depth"].max()
                        depth_of_discharge_t1 = profile.loc[
                            mask & (profile["Half_Cycle_Depth"] < 0), "Half_Cycle_Depth"].abs().max()
                        doc_t1 = max(depth_of_charge_t1, depth_of_discharge_t1)
                        charge_rate = profile.loc[soc_indices, "Charge_Rate"].replace(0, np.nan).mean()
                        discharge_rate = profile.loc[soc_indices, "Discharge_Rate"].abs().replace(0, np.nan).mean()

                        new_data_t0 = np.array(
                            [fec_t0, charge_rate, discharge_rate, temperature, soc_t1 * 100, doc_t1 * 100])
                        new_data_t1 = np.array(
                            [fec_t1, charge_rate, discharge_rate, temperature, soc_t1 * 100, doc_t1 * 100])
                        new_data_t0 = scale_data(None, new_data_t0.reshape(-1, 1, new_data_t0.shape[0]),
                                                 path=self.normalizer_path_cycle, val_test_flag=True)
                        new_data_t1 = scale_data(None, new_data_t1.reshape(-1, 1, new_data_t1.shape[0]),
                                                 path=self.normalizer_path_cycle, val_test_flag=True)

                        loss_t0_mean = self.model_prob(new_data_t0).mean().numpy().reshape(-1)[0]
                        loss_t1_mean = self.model_prob(new_data_t1).mean().numpy().reshape(-1)[0]
                        loss = loss_t1_mean - loss_t0_mean
                        stddev = self.model_prob(new_data_t1).stddev().numpy().reshape(-1)[0]

                        loss_array[i + num_steps * year] = loss / 100
                        cumulative_loss += loss
                        loss_array_stddev[i + num_steps * year] = stddev / 100
                        fec_t0 = fec_t1
                        soc_indices = []
                    else:
                        loss_array[i + num_steps * year] = 0
                        loss_array_stddev[i + num_steps * year] = stddev / 100
                        soc_indices.append(idx)

                    pbar.update(1)  # Update the progress bar

                fec_year = fec_t1

        return loss_array, loss_array_stddev

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
