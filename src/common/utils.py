import os
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import pandas as pd
import math
import re
import shutil
import zipfile
import tempfile
import random
from datetime import datetime

from pandas import DataFrame
from plotly.graph_objs import Figure
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Any, Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.reset_defaults()
# sns.set_style('whitegrid')
# sns.set_context('talk')
sns.set_context(context='talk', font_scale=0.7)


def calculate_soc_profile(current_profile: np.ndarray, initial_soc: float, battery_capacity: float) -> np.ndarray:
    """
    Calculate the state of charge (SoC) profile for a battery given a current profile.

    Args:
        current_profile (np.ndarray): Array of current values (in amperes) over time.
        initial_soc (float): Initial state of charge (in percentage).
        battery_capacity (float): Battery capacity (in ampere-hours).

    Returns:
        np.ndarray: Array of state of charge values over time.
    """
    # Convert current in amperes to ampere-minutes (since the time step is one minute)
    # and then to ampere-hours
    ampere_hours = current_profile / 60.0  # There are 60 minutes in an hour

    # Calculate the charge transferred for each minute in ampere-hours
    charge_transferred = np.cumsum(ampere_hours)

    # Calculate the change in SoC: ΔSoC = (Charge Transferred / Battery Capacity)
    # Subtract from initial SoC because positive current means discharge
    soc_profile = initial_soc / 100 + (charge_transferred / battery_capacity)

    return soc_profile


def repeat_and_align_soc_profiles_to_dict(profiles_dict: Dict[str, List[Any]], repeat_dict: Dict[str, int]) -> Dict[
    str, pd.DataFrame]:
    """
    Repeat and align state of charge (SoC) profiles based on provided repetition counts and align them to the current time.

    Args:
        profiles_dict (Dict[str, List[Any]]): Dictionary where keys are profile names and values are lists containing metadata and SoC profiles.
        repeat_dict (Dict[str, int]): Dictionary where keys are profile names and values are integers specifying the number of times to repeat each profile.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of aligned SoC profiles as pandas DataFrames with datetime indices.
    """
    # Initialize a dictionary to store final profiles DataFrames
    final_dataframes = {}

    # Extract keys and ensure they match
    keys = profiles_dict.keys()
    assert keys == repeat_dict.keys(), "Keys of profiles and repeat times must match"

    # Initialize a list to store repeated profiles and lengths
    repeated_profiles = {}
    lengths = []

    # Repeat each profile according to the specified number of times and store
    for key in keys:
        profile = profiles_dict[key][1]
        times = repeat_dict[key]
        repeated_profile = np.tile(profile, times)
        repeated_profiles[key] = repeated_profile
        lengths.append(len(repeated_profile))

    # Determine the minimum length to ensure all profiles are the same length
    min_length = min(lengths)

    # Calculate the start time as now minus the length of the series in minutes
    start_time = pd.Timestamp.now() - pd.Timedelta(minutes=min_length)

    # Create a datetime index once for all profiles, starting now, with one-minute intervals
    index = pd.date_range(start=start_time, periods=min_length, freq='min')  # Updated to 'min'

    # Trim the profiles to the minimum length and create DataFrames
    for key in repeated_profiles:
        trimmed_profile = repeated_profiles[key][:min_length]
        final_dataframes[key] = pd.DataFrame(trimmed_profile, index=index, columns=['soc'])

    return final_dataframes


def replace_outliers_with_mean(data, threshold=3.0):
    """
    Replaces outliers in the data with the mean of the non-outlier values.

    Args:
        data (numpy array): The data points to be processed.
        threshold (float): The number of standard deviations to use for defining outliers.

    Returns:
        numpy array: The data with outliers replaced by the mean of non-outlier data.
    """
    mean = np.mean(data)
    std = np.std(data)
    non_outliers = [d for d in data if abs(d - mean) < threshold * std]
    mean_non_outliers = np.mean(non_outliers)

    # Replace outliers with the mean of non-outliers
    replaced_data = np.where(abs(data - mean) < threshold * std, data, mean_non_outliers)
    return replaced_data


def low_pass_fft(y, cutoff_frequency):
    """
    Apply a low-pass Fourier transform filter to the data.

    Args:
        y (numpy array): Data to be smoothed.
        cutoff_frequency (float): The cutoff frequency for filtering; values above this will be removed.

    Returns:
        numpy array: The smoothed data.
    """
    # Compute the FFT
    w = np.fft.fft(y)
    frequency = np.fft.fftfreq(len(y))

    # Remove high frequencies
    w[np.abs(frequency) > cutoff_frequency] = 0

    # Compute the inverse FFT
    smoothed = np.fft.ifft(w)

    return np.real(smoothed)


def apply_savitzky_golay(y, window_length, poly_order, deriv=0, rate=1):
    """
    Applies the Savitzky-Golay filter to smooth data and adjusts the smoothed data
    to start from the same initial value as the original data.

    Args:
        y (numpy array): The data points to be smoothed.
        window_length (int): The length of the filter window (i.e., the number of coefficients).
                             Window_length must be a positive odd integer.
        poly_order (int): The order of the polynomial used to fit the samples.
                          Poly_order must be less than window_length.
        deriv (int, optional): The order of the derivative to compute. Default is 0, which means to smooth the data.
        rate (int, optional): The spacing of the samples to which the filter will be applied. Default is 1.

    Returns:
        numpy array: The smoothed data, adjusted to start from the same initial value as y.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if window_length < 1:
        raise ValueError("window_length must be positive.")
    if poly_order >= window_length:
        raise ValueError("poly_order must be less than window_length.")

    # Apply the Savitzky-Golay filter
    smoothed = savgol_filter(y, window_length, poly_order, deriv, rate)

    # Adjust the smoothed data to start at the same value as the original data
    if len(y) > 0:
        offset = y[0] - smoothed[0]
        smoothed += offset

    return smoothed


def smooth_data_with_bspline(x, y, degree=3, n_knots=10):
    """
    Applies B-Spline smoothing to the provided x and y data.

    Args:
        x: Array-like, the independent variable data points.
        y: Array-like, the dependent variable data points.
        degree: Integer, the degree of the spline polynomial.
        n_knots: Integer, the number of internal knots used for the spline fitting.

    Returns:
        y_smooth: Array of the smoothed y values.
    """
    if len(x) > degree + n_knots + 1:
        # Calculating positions for knots within the range of x, excluding the extremes
        t = np.linspace(x[1], x[-2], num=n_knots)
        try:
            # Create the least squares B-Spline
            spl = LSQUnivariateSpline(x, y, t, k=degree)
            # Evaluate spline across the domain
            y_smooth = spl(x)
            return y_smooth
        except Exception as e:
            # If spline fitting fails, print error and return original y
            print(f"Failed to fit spline: {str(e)}")
            return y
    else:
        # If not enough points, return the original y
        print("Not enough data points to fit B-Spline.")
        return y


def smooth_data_with_lowess(x, y, frac=0.1):
    """
    Applies LOWESS (Locally Weighted Scatterplot Smoothing) to the provided x and y data.

    Args:
        x: Array-like, the independent variable data points.
        y: Array-like, the dependent variable data points.
        frac: The fraction of the data points used to compute each y-value in the smoothed array, representing the size of the neighborhood.

    Returns:
        y_smooth: Array of the smoothed y values.
    """

    # Applying LOWESS smoothing
    lowess_results = lowess(y, x, frac=frac)

    # Using the same x array for interpolation to ensure the output length matches exactly
    y_smooth = np.interp(x, lowess_results[:, 0], lowess_results[:, 1])

    return y_smooth


def smooth_dataframe_with_spline(df, s=1.0):
    """
    Smooth each column in the DataFrame using spline interpolation.

    Args:
        df: A pandas DataFrame with numeric values.
        s: Smoothing factor, specifying the number of knots in the spline.

    Returns:
        A pandas DataFrame with spline-smoothed values.
    """
    smoothed_data = pd.DataFrame(index=df.index)
    for column in df.columns:
        spline = UnivariateSpline(df.index, df[column], frac=0.1)
        smoothed_data[column] = spline(df.index)
    return smoothed_data


def update_dataframe_max_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Updates a DataFrame by replacing rows where any column's value is greater
    than the values in both the previous and next rows with the average of
    these surrounding rows' values. The process iterates until no such row exists.

    This function assumes that the DataFrame does not contain NaN values in
    the relevant columns since handling NaN values might require additional
    checks and potentially alter the method's logic.

    Args:
        df: A pandas DataFrame with numeric values.

    Returns:
        A pandas DataFrame with updated values based on the described condition and a boolean value to determine whether
        hill shaving must stop or not.

    Example:
        >>> data = {'A': [1, 3, 2, 5, 4], 'B': [5, 6, 7, 8, 9], 'C': [9, 8, 2, 1, 5]}
        >>> df = pd.DataFrame(data)
        >>> updated_df, changed = update_dataframe_max_values(df)
        >>> print(updated_df)
    """
    arr = df.values
    changed = False

    # Iterate until there are no more updates
    while True:
        update_needed = np.zeros_like(arr, dtype=bool)

        # We skip the first and last index to avoid boundary issues
        for i in range(1, len(arr) - 1):
            # Find peaks that are greater than both previous and next values
            peak_mask = (arr[i] >= arr[i - 1]) & (arr[i] >= arr[i + 1])
            if np.any(peak_mask):
                update_needed[i, peak_mask] = True

        # If no peaks are found, end the loop
        if not update_needed.any():
            break

        # Update values where peaks were detected
        for i in range(1, len(arr) - 1):
            if update_needed[i].any():
                arr[i, update_needed[i]] = (arr[i - 1, update_needed[i]] + arr[i + 1, update_needed[i]]) / 2
                changed = True

    return pd.DataFrame(arr, columns=df.columns), changed


def update_dataframe_min_values(df: pd.DataFrame) -> Tuple[DataFrame, bool]:
    """
    Updates a DataFrame by replacing rows where any column's value is greater
    than the values in both the previous and next rows with the average of
    these surrounding rows' values. The process iterates until no such row exists.

    This function assumes that the DataFrame does not contain NaN values in
    the relevant columns since handling NaN values might require additional
    checks and potentially alter the method's logic.

    Args:
        df: A pandas DataFrame with numeric values.

    Returns:
        A pandas DataFrame with updated values based on the described condition and a boolean value determine whether
        valley shaving must stop or not.

    Example:
        >>> data = {'A': [1, 3, 2, 5, 4], 'B': [5, 6, 7, 8, 9], 'C': [9, 8, 2, 1, 5]}
        >>> df = pd.DataFrame(data)
        >>> updated_df = update_dataframe_values(df)
        >>> print(updated_df)
    """
    # Convert DataFrame to NumPy array for efficient computation
    arr = df.values

    i = 0
    # Perform iterative update until no condition is met
    while True:
        # Calculate previous and next rows
        prev = np.vstack([arr[0], arr[:-1]])
        next_ = np.vstack([arr[1:], arr[-1]])

        # Calculate the average of previous and next rows
        avg = (prev + next_) / 2

        # Identify cells that should be updated
        condition = (arr < prev) & (arr < next_)

        # Check if there are any cells to update, break if none
        if not condition.any():
            if i == 0:
                valley_shaver = False
            else:
                valley_shaver = True
            break

        # Update values based on the condition
        arr[condition] = avg[condition]
        i += 1

    # Convert the updated NumPy array back to a DataFrame
    return pd.DataFrame(arr, columns=df.columns), valley_shaver


def calculate_cumulative_loss_confidence_interval(
        loss: np.ndarray,
        loss_stddev: np.ndarray,
        confidence_level: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate cumulative loss and confidence intervals from periodic losses and stddev arrays.

    This function calculates cumulative loss, cumulative variance, cumulative standard deviation,
    and confidence intervals based on a specified confidence level (e.g., 95% confidence level),
    given arrays of losses and their standard deviations for each period (e.g., hourly).

    Args:
        loss (np.ndarray): Array of periodic losses.
        loss_stddev (np.ndarray): Array of periodic standard deviations.
        confidence_level (float): Desired confidence level as a decimal (e.g., 0.95 for 95% confidence).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - cumulative_loss (np.ndarray): Cumulative losses over periods.
            - upper_bound (np.ndarray): Upper bound of the confidence interval.
            - lower_bound (np.ndarray): Lower bound of the confidence interval.
    """
    # Calculate the z-score multiplier for the given confidence level
    confidence_multiplier = norm.ppf((1 + confidence_level) / 2)

    cumulative_loss = np.cumsum(loss)
    cumulative_variance = np.cumsum(np.square(loss_stddev))
    cumulative_std = np.sqrt(cumulative_variance)
    periods = np.arange(1, len(loss) + 1)
    cumulative_sem = cumulative_std / np.sqrt(periods)

    # Calculate adjusted confidence intervals using the calculated multiplier
    upper_bound = cumulative_loss + confidence_multiplier * cumulative_sem
    lower_bound = cumulative_loss - confidence_multiplier * cumulative_sem

    # Remove negative values from upper and lower bounds
    upper_bound = np.clip(upper_bound, 0, None)
    lower_bound = np.clip(lower_bound, 0, None)

    return cumulative_loss, upper_bound, lower_bound


def plot_calendar_dynamic_loss_with_confidence_intervals(
        days: List[int],
        cumulative_loss: np.ndarray,
        upper_bound: np.ndarray,
        lower_bound: np.ndarray,
        confidence_level: float,
        save_plot: bool = False,
        save_path: str = None,
        validation_type: str = 'Calendar',
        segmentation_bool: bool = False,
        segment_size: List[int] = None,
        temp_values: List[int] = None,
        soc_values: List[float] = None,
        validation_dict: Dict = None,
) -> None:
    """Plots cumulative loss with confidence intervals and optional segmentation.

    Args:
        days (List[int]): List of days corresponding to the indices of cumulative data.
        cumulative_loss (np.ndarray): Array of cumulative losses over time.
        upper_bound (np.ndarray): Upper bound of the confidence interval.
        lower_bound (np.ndarray): Lower bound of the confidence interval.
        confidence_level (float): Confidence level of the uncertainty.
        save_plot (bool, optional): If True, plot will be saved to the specified path. Defaults to False.
        save_path (str, optional): Path to save. Defaults to None.
        validation_type (str, optional): Type of aging used for validation. Defaults to 'Calendar'.
        segmentation_bool (bool): If True, adds segmentation to the plot.
        segment_size (List[int]): Sizes of each segment for segmentation.
        temp_values (List[int]): Temperature values for segment annotations.
        soc_values (List[float]): State of Charge (SOC) values for segment annotations.
        validation_dict (dict, optional): Dictionary of validation values containing the x and y axes values.
            Defaults to None.

    This function creates a Plotly graph showing cumulative loss with confidence intervals.
    If segmentation is enabled, it also adds colored segment backgrounds and annotations.
    Returns:
        plotly.graph_objs
    """
    # Calculating min and max for y-axis, considering the std deviation
    min_y = min(lower_bound)
    max_y = max(upper_bound)

    # Add a buffer to min and max y-values for better visualization
    buffer = (max_y - min_y) * 0.1
    min_y -= buffer
    max_y += buffer

    fig = go.Figure()

    # Add traces for mean value and confidence bounds
    fig.add_trace(go.Scatter(
        x=days,
        y=cumulative_loss,
        mode='lines',
        name='Mean Value',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=days,
        y=lower_bound,
        mode='lines',
        name='Lower and Upper Bounds',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=days,
        y=upper_bound,
        mode='lines',
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.3)',
        line=dict(color='blue')
    ))
    if validation_dict is not None:
        fig.add_trace(go.Scatter(
            x=validation_dict['x'],
            y=validation_dict['y'],
            mode='markers',
            marker=dict(
                symbol='x',
                color='green',
            ),
            name='Validation Data',
        ))

    # Optional segmentation
    if segmentation_bool and segment_size:
        colors = ['LightSkyBlue', 'LightGreen', 'LightPink', 'Wheat', 'Lavender', 'Khaki', 'AliceBlue', 'MistyRose']
        start_day = 0
        for i, end_day in enumerate(segment_size):
            fig.add_shape(type="rect",
                          x0=start_day, y0=0, x1=end_day, y1=1,
                          xref="x", yref="paper",
                          fillcolor=colors[i % len(colors)],
                          opacity=0.5,
                          layer="below",
                          line_width=0)
            # Add midpoint annotations for segments
            midpoint = (start_day + end_day) / 2
            fig.add_annotation(
                x=midpoint, y=1, xref="x", yref="paper",
                text=f'{temp_values[i % len(temp_values)]}°C \n {soc_values[i % len(soc_values)] * 100}% SOC',
                showarrow=False,
                font=dict(size=12),
                bgcolor="lightgrey",
                borderpad=4,
                align="left"
            )
            start_day = end_day

    # Update layout
    fig.update_layout(
        title=f'Dynamic Validation with {confidence_level * 100}% confidence level',
        xaxis_title='Days' if validation_type == 'Calendar' else 'FEC',
        yaxis_title='Q_loss,cal (%)' if validation_type == 'Calendar' else 'Q_loss,cyc (%)',
        yaxis=dict(range=[min_y, max_y]),
        showlegend=True
    )
    if save_plot:
        now = datetime.now()
        datetime_str = now.strftime('%Y%m%d_%H%M%S')
        if 'calendar' in save_path:
            filename = os.path.join(save_path, f'validation_calendar_{datetime_str}.html')
        elif 'cycle' in save_path:
            filename = os.path.join(save_path, f'validation_cycle_{datetime_str}.html')
        else:
            raise ValueError('There must be either a "calendar" or a "cycle" in the save_plot path')
        fig.write_html(filename)
    fig.show()


def load_latest_model(model_dir: str, custom_objects: Dict[str, Any] = None) -> Optional[Any]:
    """
    Loads the most recent TensorFlow model from a directory containing
    model files named with a 'model_' prefix. The models are assumed to
    be compressed in ZIP format.

    Args:
        model_dir: A string specifying the directory where model files are stored.
        custom_objects: A dictionary of custom objects needed for TensorFlow
                        model loading.

    Returns:
        The most recent TensorFlow model if found; otherwise, None.

    Raises:
        FileNotFoundError: If no model files with the 'model_' prefix are found
                           in the specified directory.
    """
    model_files = os.listdir(model_dir)
    model_files = sorted([f for f in model_files if f.startswith('model_')], reverse=True)

    if model_files:
        # Path to the most recent model file
        model_pb_al_file = os.path.join(model_dir, model_files[0])
        # Load the most recent model
        model_pb_al = load_tf_model_from_zip(model_pb_al_file, custom_objects=custom_objects)
        print(f"Loaded model from: {model_pb_al_file}")
        return model_pb_al
    else:
        print("No model files found in the directory.")
        return None


def augment_list(original_list: List[float], target_length: int) -> List[float]:
    """
    Augments a list of numbers by adding interpolated points between each pair of original points to reach a specified target length.
    Linear interpolation is used to generate the intermediate points.

    Args:
        original_list (List[float]): The original list of numbers to be augmented.
        target_length (int): The desired length of the augmented list. If the target length is less than or equal to the length of the original list, the original list is returned.

    Returns:
        List[float]: The augmented list of numbers with the specified target length. If the target length is reached by adding interpolated points between the original points, these points are included in the returned list.

    Example:
        >>> original_list = [1.0, 2.0, 3.0]
        >>> target_length = 5
        >>> augment_list(original_list, target_length)
        [1.0, 1.5, 2.0, 2.5, 3.0]
    """
    if len(original_list) >= target_length:
        return original_list  # Return the original list if it's already long enough

    num_points_to_add = target_length - len(original_list)
    intervals = len(original_list) - 1  # Number of intervals between points
    points_per_interval = num_points_to_add // intervals

    # Calculate additional points if points_per_interval does not divide evenly
    extra_points = num_points_to_add % intervals

    augmented_list = []
    for i in range(intervals):
        augmented_list.append(original_list[i])
        # Calculate and insert intermediate points
        for j in range(1, points_per_interval + 1):
            alpha = j / (points_per_interval + 1)
            new_point = original_list[i] * (1 - alpha) + original_list[i + 1] * alpha
            augmented_list.append(new_point)

        # Distribute extra points evenly among the first few intervals
        if extra_points > 0:
            alpha = (points_per_interval + 1) / (points_per_interval + 2)
            new_point = original_list[i] * (1 - alpha) + original_list[i + 1] * alpha
            augmented_list.append(new_point)
            extra_points -= 1

    # Append the last original point
    augmented_list.append(original_list[-1])

    return augmented_list


def plot_feature_distributions(df: pd.DataFrame, type: str, save_plot: bool = False, save_path: str = None) -> None:
    """
    Standardizes the features of the given DataFrame and plots their distributions
    using a violin plot. This function assumes that the DataFrame has a labels
    column which is not included in the features to be standardized and plotted.

    Args:
        df: A pandas DataFrame containing the data to be processed and plotted. It
            must include a labels column and any number of feature columns.
        type (str): Type of the aging. It must be 'calendar' or 'cycle'.
        save_plot (bool, optional): If True, plot will be saved to the specified path. Defaults to False.
        save_path (str, optional): Path to save. Defaults to None.

    Returns:
        None. The function produces a violin plot of the standardized features against
        labels' values.
    """
    # Set font size parameters
    plt.rcParams.update({'font.size': 20})  # Change the number to increase/decrease font size
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 20})
    plt.rcParams.update({'ytick.labelsize': 20})

    # Exclude labels from features and calculate mean and std
    labels_column = df.columns[-1]

    if type.lower() == 'calendar':
        y_name = re.sub("Q_loss", "$Q_{loss, cal}$", labels_column)
    elif type.lower() == 'cycle':
        y_name = re.sub("Q_loss", "$Q_{loss, cyc}$", labels_column)
    else:
        raise ValueError(f"String must be either 'calendar' or 'cycle', but got '{type}'")

    features = df.drop(columns=[labels_column]).columns.tolist()
    mean = df[features].mean()
    std = df[features].std()
    # Replace 0 in std with 1 to avoid division by zero errors
    std_replaced = std.replace(0, 1)
    # Standardize the features
    df_std = (df[features] - mean) / std_replaced
    # Melt the standardized DataFrame for plotting
    df_long = df_std.melt(var_name='Feature', value_name=y_name)
    # Plot using Seaborn's violinplot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Feature', y=y_name, data=df_long)
    # plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.gca().set_xlabel('')
    plt.ylabel(y_name)

    if save_plot and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_static_loss_with_confidence_intervals(x, y_true, y_fitted, lower_bound, upper_bound, name, type: str,
                                               save_plot: bool = False, save_path: str = None, **kwargs):
    """
    Create a plot with upper and lower bounds.

    Parameters
    ----------
    x : array-like
        x-axis data.
    y_true : array-like
        y-axis data for true values.
    y_fitted : array-like
        y-axis data for fitted values.
    lower_bound : array-like
        y-axis data for -2 standard deviations.
    upper_bound : array-like
        y-axis data for +2 standard deviations.
    name : str
        Base name for the saved figure file.
    """
    fontsize = kwargs.get('fontsize', 20)
    x = np.asarray(x, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_fitted = np.asarray(y_fitted, dtype=np.float64)
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)

    name_true = 'True'
    name_fitted = '$\mu$'
    name_pm2sd = '$\mu \pm 2 \sigma$'

    # Create the plot
    plt.figure(figsize=(12, 8))  # Adjusted figure size for clarity

    # Plot true data
    plt.scatter(x, y_true, label=name_true, color='green', marker='o', s=50)  # Increased marker size

    # Plot fitted line
    plt.plot(x, y_fitted, label=name_fitted, color='blue', linewidth=2)  # Increased line width

    # Plot confidence intervals with fill
    plt.plot(x, lower_bound, label=name_pm2sd, color='blue', linestyle='--')
    plt.plot(x, upper_bound, color='blue', linestyle='--')
    plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3)

    # Add labels and title
    if type == 'calendar':
        plt.xlabel('Time (Days)', fontsize=fontsize)  # Set font size
        plt.ylabel('$Q_{loss, cal}$ (%)', fontsize=fontsize)  # Set font size
        plt.title(f'Calendar Model Regression for {name}', fontsize=fontsize)  # Set title font size
    elif type == 'cycle':
        plt.xlabel('FEC', fontsize=fontsize)  # Set font size
        plt.ylabel('$Q_{loss, cyc}$ (%)', fontsize=fontsize)  # Set font size
        plt.title(f'Cyclic Model Regression for {name}', fontsize=fontsize)  # Set title font size

    plt.legend(fontsize=fontsize)  # Set legend font size
    plt.grid(True)  # Added grid for better readability

    # Set tick label font sizes
    plt.tick_params(axis='both', labelsize=fontsize)  # Adjust font size for both axes

    # Layout adjustment
    plt.tight_layout()  # Adjust layout to make room for labels

    # Save the figure
    if save_plot:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')

    # Show the plot
    plt.show()


def process_calendar_data(df_calendar: pd.DataFrame, df_calendar_time: pd.DataFrame) -> pd.DataFrame:
    """
    Processes calendar dataframes by cleaning, normalizing, and merging data.

    Drops NaN columns, transposes the dataframe, filters out rows based on specific keywords in the index,
    normalizes data, merges time data from another dataframe, and renames duplicate columns
    based on temperature and state of charge (SOC) patterns and their mean values.

    Args:
        df_calendar: A pandas DataFrame containing calendar data.
        df_calendar_time: A pandas DataFrame containing time-related data for the calendar.

    Returns:
        A processed pandas DataFrame with cleaned, normalized data and merged time information.

    """
    # Drop NaN columns and transpose
    df_calendar.dropna(axis=1, inplace=True)
    df_calendar = df_calendar.T

    # Filter rows based on keywords
    condition = df_calendar.index.to_series().str.contains('R_i') | df_calendar.index.to_series().str.contains(
        'd_Kapazität') | df_calendar.index.to_series().str.contains(
        'Spalte') | df_calendar.index.to_series().str.contains('Zelle') | df_calendar.index.to_series().str.contains(
        'Start-SoC') | df_calendar.index.to_series().str.contains(
        'Testart') | df_calendar.index.to_series().str.contains(
        'Testpunktnummer') | df_calendar.index.to_series().str.contains(
        'Temperatur') | df_calendar.index.to_series().str.contains('Mittlerer SoC')
    df_calendar = df_calendar[~condition]

    # Set the first row as column headers
    df_calendar.columns = df_calendar.iloc[0]
    df_calendar = df_calendar.drop(df_calendar.index[0])
    df_calendar.reset_index(drop=True, inplace=True)

    # Normalize data
    max_values = df_calendar.max()
    df_calendar = df_calendar / max_values

    df_calendar_time.columns = df_calendar_time.iloc[0]
    df_calendar_time.drop(df_calendar_time.index[0], inplace=True)
    df_calendar_time.reset_index(drop=True, inplace=True)

    # Merge time data
    df_calendar['Storage time / hours'] = df_calendar_time['Storage time / hours']
    df_calendar = rename_duplicate_columns(df_calendar)
    # Rename duplicate columns based on pattern matching and averaging
    df_calendar.columns = [col.replace(",", ".") for col in df_calendar.columns]
    pattern = r"Lagerung_(?P<temperature>-?\d+\.?\d*)°C_(?P<soc>\d+\.?\d*)%"
    groups = {}
    for col in df_calendar.columns:
        match = re.search(pattern, col)
        if match:
            key = (match.group('temperature'), match.group('soc'))  # Group by temperature and SOC
            if key not in groups:
                groups[key] = []
            groups[key].append(col)

    new_columns = {}
    for key, cols in groups.items():
        if len(cols) == 3:
            means = df_calendar[cols].mean().sort_values()
            for rank, col in enumerate(means.index):
                new_col = re.sub(r"\_\d+$", "", col)
                if rank == 0:
                    new_columns[col] = new_col + '_min'
                elif rank == 1:
                    new_columns[col] = new_col + '_mean'
                else:
                    new_columns[col] = new_col + '_max'
        else:
            print(f"Unexpected number of columns for group {key}: {cols}")

    df_calendar.rename(columns=new_columns, inplace=True)

    return df_calendar


def extract_conditions(df: pd.DataFrame, degradation_type: str, complete_data: bool = True) -> Dict[int, List[float]]:
    """Extract conditions from the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing columns starting with 'x_' or 'Lagerung_'
            representing different test point conditions.
        degradation_type (str): Type of degradation.
        complete_data (bool, optional): Whether calendar data is completed. Defaults to True.

    Returns:
        Dict[int, List[float]]: A dictionary mapping an index to a list of conditions.
    """
    conditions = {}
    if degradation_type == 'cycle':
        for i, column_name in enumerate(df.columns[df.columns.str.startswith('x_')]):
            parts = column_name.split('_')
            temp = float(parts[2].replace('°C', ''))
            soc = float(parts[3].replace('%SOC', ''))
            dod = float(parts[4].replace('%DOD', ''))
            charge_rate = float(parts[5].replace('C', ''))
            discharge_rate = float(parts[6].replace('C', ''))
            conditions[i] = [charge_rate, discharge_rate, temp, soc, dod]
    elif degradation_type == 'calendar' and complete_data:
        pattern = r"Lagerung_(?P<temperature>-?\d+\.?\d*)°C_(?P<soc>\d+\.?\d*)%"
        for i, col in enumerate(df.columns[df.columns.str.startswith('Lagerung_') & df.columns.str.endswith('mean')]):
            match = re.match(pattern, col)
            if match:
                temp_soc = match.groupdict()
                temp = float(temp_soc['temperature'])
                soc = float(temp_soc['soc'])
                conditions[i] = [temp, soc]
    elif degradation_type == 'calendar' and complete_data is False:
        pattern = r"TP_(?P<temperature>-?\d+\.?\d*)°C,(?P<soc>\d+\.?\d*)%SOC"
        for i, col in enumerate(df.columns[df.columns.str.startswith('TP_')]):
            match = re.match(pattern, col)
            if match:
                temp_soc = match.groupdict()
                temp = float(temp_soc['temperature'])
                soc = float(temp_soc['soc'])
                conditions[i] = [temp, soc]
    else:
        raise ValueError(f'The type of degradation must be \'cycle\' or \'calendar\', but it is {degradation_type}')
    return conditions


def shuffle_and_split(conditions: Dict[int, List[float]], split_percentages: List[float]) -> Tuple[
    Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    """Shuffles and splits the conditions into train, dev, and test sets.

    Args:
        conditions (Dict[int, List[float]]): The conditions' dictionary.
        split_percentages (List[float]): The percentages to split the conditions
            into train, dev, and test sets.

    Returns:
        Tuple[Dict, Dict, Dict]: Three dictionaries containing the conditions
            for train, dev, and test sets, respectively.
    """
    items = list(conditions.items())
    random.shuffle(items)
    total_items = len(items)
    train_size = round(total_items * split_percentages[0])
    dev_size = round(total_items * split_percentages[1])
    # train_items, dev_items, test_items = (items[:train_size],
    #                                       items[train_size:train_size + dev_size],
    #                                       items[train_size + dev_size:])
    test_num = False
    test_items = {test_num: conditions[test_num]}
    del conditions[test_num]
    train_items = conditions
    dev_items = {}
    return dict(train_items), dict(dev_items), dict(test_items)


def create_df_cycle_dict(df: pd.DataFrame, df_calendar: pd.DataFrame, conditions: Dict, is_inverse: bool,
                         add_rows: bool = False, fec_step_size: float = 25, complete_data: bool = True) \
        -> Dict[Any, DataFrame]:
    """
    Creates a DataFrame df_cycle by subtracting values from df and df_calendar
    based on specified conditions.

    Args:
        df (pd.DataFrame): Input DataFrame with columns starting with 'x_' and 'y_'.
        df_calendar (pd.DataFrame): Input DataFrame with calendar information.
        conditions (dict): A dictionary with indices of 'x_' columns as keys and
                           condition vectors as values.
        is_inverse (bool): If true, the inputs and labels are inverted for virtual time or FEC models.
        add_rows (bool, optional): If True, adds new interpolated rows to df_cycle. Defaults to False.
        fec_step_size (float, optional): Step size for adding new interpolated rows to df_cycle. Defaults to 25.
        complete_data (bool, optional): Whether data is completed or not. The public version of calendar aging
        dataset does not include all cells' data. Defaults to True.

    Returns:
        dict[Any, pd.DataFrame]: The resulting dictionary of DataFrames.
    """
    df_cycle_dict = {}

    for i, indicator in enumerate(df.columns[df.columns.str.startswith('x_')]):
        df_cycle = pd.DataFrame(index=df.index)
        if i in conditions:
            df_cycle[indicator] = df[indicator]
            condition_vector = conditions[i]
            temp, soc = int(condition_vector[2]), int(condition_vector[3])
            if is_inverse:
                comb_columns = [f'y_{indicator.replace("x_", "")}']
                calendar_columns = [f'Lagerung_{temp}°C_{soc}%_mean']
            elif complete_data:
                comb_columns = [f'y_min_{indicator.replace("x_", "")}',
                                f'y_{indicator.replace("x_", "")}',
                                f'y_max_{indicator.replace("x_", "")}']
                calendar_columns = [f'Lagerung_{temp}°C_{soc}%_min',
                                    f'Lagerung_{temp}°C_{soc}%_mean',
                                    f'Lagerung_{temp}°C_{soc}%_max']
            elif complete_data is False:
                comb_columns = [f'y_min_{indicator.replace("x_", "")}',
                                f'y_{indicator.replace("x_", "")}',
                                f'y_max_{indicator.replace("x_", "")}']
                calendar_columns = [f'TP_{temp}°C,{soc}%SOC']
            j = 0
            for comb_col_name in comb_columns:
                for calendar_col_name in calendar_columns:
                    col_name = f"{comb_col_name}_{j}"
                    df_cycle[col_name] = df_calendar[calendar_col_name] - df[comb_col_name]
                    j += 1
            if add_rows:
                df_cycle = df_cycle.apply(pd.to_numeric, errors='coerce')
                step_size = fec_step_size
                max_fec = df_cycle[indicator].max()
                new_range = np.arange(0, max_fec + step_size, step_size)
                temp_df = pd.DataFrame(new_range, columns=[indicator])
                interpolated_df = pd.merge(temp_df, df_cycle, on=indicator, how='outer').sort_values(
                    indicator).set_index(indicator).interpolate()
                df_cycle = interpolated_df.ffill().bfill().reset_index()
            df_cycle_dict[indicator] = df_cycle
    return df_cycle_dict


def preprocess_data(df: pd.DataFrame, conditions: Dict[int, List[float]], df_calendar: pd.DataFrame,
                    degradation_type: str, is_inverse: bool = False, augment: bool = False, num_aug: int = 10,
                    interpolate_rows: bool = False, fec_step_size: float = 25, complete_data: bool = True) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesses the data based on the given conditions and returns input sequences and labels.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        conditions (Dict[int, List[float]]): The conditions' dictionary.
        df_calendar (pd.DataFrame): DataFrame containing calendar data for aging effects.
        degradation_type (str): Type of degradation.
        is_inverse (bool, optional): If true, the inputs and labels are inverted for virtual time or FEC models.
        augment (bool, optional): If True, data will be augmented.
        num_aug (int, optional): Number of augmented data. Default is 10.
        interpolate_rows (bool, optional): If True, adds new interpolated rows to df_cycle. Defaults to False.
        fec_step_size (float, optional): Step size for adding new interpolated rows to df_cycle. Defaults to 25.
        complete_data (bool, optional): Whether data is completed or not. The public version of calendar aging
        dataset does not include all cells' data. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the input sequences and labels, respectively.
    """
    input_sequences, labels = [], []
    if degradation_type == 'cycle':
        df_cycle_dict = create_df_cycle_dict(df, df_calendar, conditions, is_inverse, add_rows=interpolate_rows,
                                             fec_step_size=fec_step_size, complete_data=complete_data)

        for indicator, df_cycle in df_cycle_dict.items():
            for index, row in df_cycle.iterrows():
                fec_value = row[f'{indicator}']
                parts = indicator.split('_')
                temp = float(parts[2].replace('°C', ''))
                soc = float(parts[3].replace('%SOC', ''))
                dod = float(parts[4].replace('%DOD', ''))
                charge_rate = float(parts[5].replace('C', ''))
                discharge_rate = float(parts[6].replace('C', ''))
                condition_vector = [charge_rate, discharge_rate, temp, soc, dod]
                temp, soc = int(condition_vector[2]), int(condition_vector[3])
                if is_inverse:
                    comb_columns = [f'y_{indicator.replace("x_", "")}']
                    calendar_columns = [f'Lagerung_{temp}°C_{soc}%_mean']
                elif complete_data:
                    comb_columns = [f'y_min_{indicator.replace("x_", "")}',
                                    f'y_{indicator.replace("x_", "")}',
                                    f'y_max_{indicator.replace("x_", "")}']
                    calendar_columns = [f'Lagerung_{temp}°C_{soc}%_min',
                                        f'Lagerung_{temp}°C_{soc}%_mean',
                                        f'Lagerung_{temp}°C_{soc}%_max']
                elif complete_data is False:
                    comb_columns = [f'y_min_{indicator.replace("x_", "")}',
                                    f'y_{indicator.replace("x_", "")}',
                                    f'y_max_{indicator.replace("x_", "")}']
                    calendar_columns = [f'TP_{temp}°C,{soc}%SOC']
                cycle_values = []
                j = 0
                for comb_col_name in comb_columns:
                    for calendar_col_name in calendar_columns:
                        col_name = f"{comb_col_name}_{j}"
                        cycle_values.append(df_cycle_dict[indicator].iloc[index][col_name])
                        j += 1
                cycle_values = augment_list(cycle_values, num_aug) if augment else cycle_values
                if not is_inverse:
                    for cycle_age in cycle_values:
                        sequence_step = [fec_value] + condition_vector
                        y = cycle_age * 100
                        input_sequences.append([sequence_step])
                        labels.append([y])
                else:
                    for cycle_age in cycle_values:
                        y = cycle_age * 100
                        if index == 0:
                            prev_fec_value = 0
                        else:
                            prev_fec_value = df_cycle.iloc[index - 1][f'{indicator}']
                        sequence_step = [y, prev_fec_value] + condition_vector
                        input_sequences.append([sequence_step])
                        labels.append([fec_value])
    elif degradation_type == 'calendar' and complete_data:
        for index, row in df_calendar.iterrows():
            for i, indicator in enumerate(df_calendar.columns[df_calendar.columns.str.startswith(
                    'Lagerung_') & df_calendar.columns.str.endswith('mean')]):
                indicator = indicator.replace('mean', '')
                if i in conditions:
                    t_value = row['Storage time / hours']
                    condition_vector = conditions[i]
                    y_min = (1 - row[indicator + 'min']) * 100
                    y_mean = (1 - row[indicator + 'mean']) * 100
                    y_max = (1 - row[indicator + 'max']) * 100
                    calendar_values = [y_min, y_mean, y_max]
                    calendar_values = augment_list(calendar_values, num_aug) if augment else calendar_values
                    if not is_inverse:
                        for y in calendar_values:
                            sequence_step = [t_value] + condition_vector
                            input_sequences.append([sequence_step])
                            labels.append([y])
                    else:
                        for y in calendar_values:
                            sequence_step = [y] + condition_vector
                            input_sequences.append([sequence_step])
                            labels.append([t_value])
    elif degradation_type == 'calendar' and complete_data is False:
        for index, row in df_calendar.iterrows():
            for i, indicator in enumerate(df_calendar.columns[df_calendar.columns.str.startswith('TP_')]):
                if i in conditions:
                    base_value = (1 - row[indicator]) * 100
                    t_value = row['Storage time / hours']
                    condition_vector = conditions[i]

                    # Define initial noise percentage and scaling factor
                    initial_noise_percentage = 0.01  # Initial 1% noise
                    scaling_factor = 0.000005  # Scaling factor for noise increase

                    # Dynamic noise calculation: initial percentage + (t_value * scaling_factor)
                    dynamic_noise_percentage = initial_noise_percentage + (t_value * scaling_factor)
                    noise_value = base_value * dynamic_noise_percentage
                    # Apply dynamic noise to y_min and y_max
                    y_min = base_value - noise_value
                    y_mean = base_value  # y_mean remains unchanged
                    y_max = base_value + noise_value
                    calendar_values = [y_min, y_mean, y_max]
                    calendar_values = augment_list(calendar_values, num_aug) if augment else calendar_values
                    if not is_inverse:
                        for y in calendar_values:
                            sequence_step = [t_value] + condition_vector
                            input_sequences.append([sequence_step])
                            labels.append([y])
                    else:
                        for y in calendar_values:
                            sequence_step = [y] + condition_vector
                            input_sequences.append([sequence_step])
                            labels.append([t_value])
    else:
        raise ValueError(f'The type of degradation must be \'cycle\' or \'calendar\', but it is {degradation_type}')
    return np.array(input_sequences, dtype=object), np.array(labels, dtype=object)[:, -1]


def scale_data(scaler: MinMaxScaler, sequences: np.ndarray, path: str, is_label: bool = False,
               is_inverse: bool = False, val_test_flag: bool = False) -> np.ndarray:
    """Scales the input sequences using MinMaxScaler.

    Args:
        scaler (MinMaxScaler): MinMaxScaler.
        sequences (np.ndarray): The sequences to scale.
        path (str): Path for saving or loading scaler.
        is_label (bool, optional): A flag indicating whether the sequences are labels. Defaults to False.
        is_inverse (bool, optional): If true, the inputs and labels are inverted for virtual time or FEC models.
        val_test_flag (bool, optional): Whether to use validation set or test set. Defaults to False.

    Returns:
        np.ndarray: The scaled sequences.
    """
    if is_label:
        sequences_reshaped = sequences.reshape(-1, 1)
        identifier = 'labels_inverse' if is_inverse else 'labels'
    else:
        sequences_reshaped = sequences.reshape(-1, sequences.shape[2])
        identifier = 'virtual' if is_inverse else 'q_loss'

    if val_test_flag:
        scaler_files = os.listdir(path)
        scaler_files = sorted([f for f in scaler_files if f.startswith(f'scalar_{identifier}')], reverse=True)
        scaler_file = os.path.join(path, scaler_files[0])
        loaded_scaler = joblib.load(scaler_file)
        sequences_scaled = loaded_scaler.transform(sequences_reshaped)
    else:
        sequences_scaled = scaler.fit_transform(sequences_reshaped)
        joblib.dump(scaler, os.path.join(path, f'scalar_{identifier}.pkl'))
    return sequences_scaled.reshape(sequences.shape)


def inverse_scale(sequences_scaled: np.ndarray, path: str, is_label: bool = False,
                  is_inverse: bool = False) -> np.ndarray:
    """
    Inversely scales an array of scaled sequences back to their original scale.

    This function loads the appropriate scaler based on the given parameters
    to inversely transform the scaled sequences. It's useful for converting model
    predictions or any scaled data back to their original units of measurement.

    Args:
        sequences_scaled (np.ndarray): The sequences that have been scaled and need to be inversely transformed.
        path (str): The directory path where the scaler objects are saved.
        is_label (bool, optional): A flag indicating whether the sequences are labels. Defaults to False.
        is_inverse (bool, optional): If true, the inputs and labels are inverted for virtual time or FEC models.
    Returns:
        np.ndarray: The inversely scaled sequences, reshaped to a 1-dimensional array.

    """
    # Determine the scaler identifier based on the input flags
    if is_label:
        identifier = 'labels_inverse' if is_inverse else 'labels'
    else:
        identifier = 'virtual' if is_inverse else 'q_loss'

    # Find the relevant scaler file
    scaler_files = os.listdir(path)
    scaler_file = os.path.join(path,
                               sorted([f for f in scaler_files if f.startswith(f'scalar_{identifier}')], reverse=True)[
                                   0])

    # Load the scaler and inversely transform the sequences
    loaded_scaler = joblib.load(scaler_file)
    inverse_value = loaded_scaler.inverse_transform(sequences_scaled)

    return inverse_value.reshape(-1)[0]


def create_inverse_df(inputs: np.ndarray, labels: np.ndarray, degradation_type: str, is_inverse: bool = False) \
        -> pd.DataFrame:
    """Creates an inverse DataFrame from the inputs and labels.

    Args:
        inputs (np.ndarray): The input sequences.
        labels (np.ndarray): The labels.
        degradation_type (str): Type of degradation.
        is_inverse (bool, optional): If true, the inputs and labels are inverted for virtual time or FEC models.

    Returns:
        pd.DataFrame: A DataFrame containing the inverse transformed data.
    """
    input_sequences_reshaped = inputs.reshape(inputs.shape[0], inputs.shape[2])
    if degradation_type == 'cycle':
        if not is_inverse:
            columns = ['FEC', 'Ch-rate', 'Disch-rate', 'T', 'SOC', 'DOC']
        else:
            columns = ['Q_loss (%)', 'Prev-FEC', 'Ch-rate', 'Disch-rate', 'T', 'SOC', 'DOC']
    elif degradation_type == 'calendar':
        if not is_inverse:
            columns = ['Time', 'T', 'SOC']
        else:
            columns = ['Q_loss (%)', 'T', 'SOC']
    else:
        raise ValueError(f'The type of degradation must be \'cycle\' or \'calendar\', but it is {degradation_type}')

    df_inverse_transformed = pd.DataFrame(input_sequences_reshaped, columns=columns)
    if not is_inverse:
        df_inverse_transformed['Q_loss (%)'] = labels
    else:
        if degradation_type == 'cycle':
            df_inverse_transformed['FEC'] = labels
        elif degradation_type == 'calendar':
            df_inverse_transformed['time'] = labels
    return df_inverse_transformed


def load_tf_model_from_zip(zip_path, delete_after_load=True, custom_objects=None):
    """
    Loads a TensorFlow model from a ZIP archive.

    Parameters:
    - zip_path: Path to the ZIP file containing the TensorFlow model.
    - delete_after_load: Whether to delete the extracted directory after loading. Default is True.
    - custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered during deserialization.

    Returns:
    - The loaded TensorFlow model.
    """
    # Create a temporary directory to extract the ZIP file
    extract_dir = tempfile.mkdtemp()

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted model to temporary directory: {extract_dir}")

    # Assuming there's only one directory in the ZIP file, and the model is inside it
    model_dir = extract_dir
    print(f"Model directory: {model_dir}")
    model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
    print(f"Model loaded from: {model_dir}")

    # Optionally, delete the extracted directory after loading the model
    if delete_after_load:
        shutil.rmtree(extract_dir)
        print(f"Temporary directory deleted: {extract_dir}")

    return model


def zip_and_delete_directory(folder_path, zip_path):
    """
    Compress a folder into a ZIP file and delete the original folder.

    Args:
        folder_path: Path to the folder you want to compress and delete.
        zip_path: The resulting ZIP file path, including the filename (e.g., "path/to/your_file.zip").

    Returns:
        None
    """
    # Compress the folder
    shutil.make_archive(zip_path, 'zip', folder_path)
    print(f"Folder '{folder_path}' has been compressed to '{zip_path}.zip'.")

    # Delete the original folder
    shutil.rmtree(folder_path)
    print(f"Folder '{folder_path}' has been deleted.")


def rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames duplicate columns in the DataFrame by appending a suffix.

    Args:
        df (pd.DataFrame): The DataFrame with potential duplicate column names.

    Returns:
        pd.DataFrame: A DataFrame with renamed duplicate columns.
    """
    new_columns = []
    seen = {}

    for column in df.columns:
        if column in seen:
            seen[column] += 1
            new_column = f"{column}_{seen[column]}"
        else:
            seen[column] = 0
            new_column = column
        new_columns.append(new_column)

    df.columns = new_columns
    return df


def split_data(df: pd.DataFrame, fractions: List = [0.7, 0.2, 0.1]):
    total = sum(fractions)
    assert math.isclose(total, 1.0, rel_tol=1e-6), f"Total is not close to 1.0, got {total}"
    assert all(x >= 0 for x in fractions), "All elements must be greater than zero"

    n = len(df)
    train_end = int(n * fractions[0])
    val_end = int(n * (fractions[0] + fractions[1]))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def normalize(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns to normalize.")
        return

    # Calculate mean and std for each numeric column
    train_mean = train_df[numeric_cols].mean()
    train_std = train_df[numeric_cols].std()

    # Normalize the data
    train_df.loc[:, numeric_cols] = (train_df[numeric_cols] - train_mean) / train_std
    val_df.loc[:, numeric_cols] = (val_df[numeric_cols] - train_mean) / train_std
    test_df.loc[:, numeric_cols] = (test_df[numeric_cols] - train_mean) / train_std
    return train_df, val_df, test_df
