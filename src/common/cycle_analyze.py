from typing import Any, Tuple
import pandas as pd
import numpy as np


def augment_dataframe_with_half_cycles(df: pd.DataFrame, battery_capacity: float = 3, depth_threshold: float = 0,
                                       diff_threshold: float = 1e-3, look_ahead: float = 6)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhances the input DataFrame with half-cycle information based on 'soc' values and specified thresholds.

    This function calculates forward-looking SoC changes, identifies half-cycles, and annotates the DataFrame
    with details about these cycles, including type, depth, and calculated rates.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index and a 'soc' column.
        battery_capacity (float, optional): The battery nominal capacity in Ah, by default 3 Ah.
        depth_threshold (float): Minimum depth change required to consider a half-cycle. Default is 0.
        diff_threshold (float): Minimum difference in SoC to distinguish charging from discharging. Default is 1e-3.
        look_ahead (float): Number of hours to look ahead when evaluating SoC changes. Default is 6.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the enhanced original DataFrame and a new DataFrame
                                           with detailed half-cycle information.
    """
    df = calculate_soc_change(df)
    if look_ahead == 0:
        look_ahead_samples = None
    else:
        look_ahead_samples = get_look_ahead_samples(df, look_ahead)
    half_cycles_df = identify_half_cycles(df, battery_capacity, look_ahead_samples, depth_threshold, diff_threshold)

    # Populate original DataFrame with half-cycle information and calculate rates
    df = populate_half_cycle_info(df, half_cycles_df, battery_capacity)

    # Handle potential NaN in the last soc_change due to the shift operation
    df['soc_change'].fillna(0, inplace=True)

    return df, half_cycles_df


def calculate_soc_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the State of Charge (SoC) change looking forward and updates the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'soc' column and datetime index.

    Returns:
        pd.DataFrame: Updated DataFrame with added 'soc_change', 'Change_Sign', and 'Sign_Change' columns.
    """
    df['soc_change'] = df['soc'].shift(-1) - df['soc']
    df['Change_Sign'] = np.sign(df['soc_change'])
    df['Sign_Change'] = df['Change_Sign'] != df['Change_Sign'].shift(-1)
    return df


def get_look_ahead_samples(df: pd.DataFrame, look_ahead: float) -> int:
    """
    Determines the number of samples to look ahead based on the DataFrame's sampling interval.

    Args:
        df (pd.DataFrame): DataFrame with datetime index.
        look_ahead (float): Number of hours to look ahead.

    Returns:
        int: The number of samples corresponding to the look-ahead period.
    """
    sampling_interval = (df.index[1] - df.index[0]).seconds / 3600
    return int(look_ahead / sampling_interval)


def populate_half_cycle_info(df: pd.DataFrame, half_cycles_df: pd.DataFrame, battery_cappacity: float) -> pd.DataFrame:
    """
    Populates the original DataFrame with half-cycle information including type, depth, and rates.

    Args:
        df (pd.DataFrame): The original DataFrame with datetime index.
        half_cycles_df (pd.DataFrame): DataFrame containing identified half-cycles information.
        battery_cappacity (float): The battery nominal capacity.

    Returns:
        pd.DataFrame: The original DataFrame updated with half-cycle information.
    """
    # Initialize new columns in the original DataFrame for half-cycle information
    df['Half_Cycle_Type'] = 'Idle'
    df['Half_Cycle_Depth'] = 0
    df['Charge_Rate'] = 0
    df['Discharge_Rate'] = 0
    df['Delta_FEC'] = 0
    for _, half_cycle in half_cycles_df.iterrows():
        start_index, end_index = half_cycle['Start_Index'], half_cycle['End_Index']
        # num_steps = (df.index.get_loc(end_index) - df.index.get_loc(start_index)) + 1
        # interpolated_values = np.linspace(0, half_cycle['Depth'], num_steps)
        # df.loc[start_index:end_index, 'Half_Cycle_Type'] = half_cycle['Type']
        df.loc[start_index:end_index, 'Half_Cycle_Depth'] = half_cycle['Depth'] / battery_cappacity
        rate, delta_fec = calculate_rates_and_delta_fec(half_cycle)
        # if half_cycle['Type'] == 'Charging':
        #     df.loc[start_index:end_index, 'Charge_Rate'] = rate
        # else:
        #     df.loc[start_index:end_index, 'Discharge_Rate'] = rate
        df.at[end_index, 'Delta_FEC'] = delta_fec / battery_cappacity
    df['FEC'] = df['Delta_FEC'].cumsum()
    current_profile = calculate_current_profile(df['soc'], battery_capacity=battery_cappacity)
    c_rate = current_profile / battery_cappacity
    df['C_Rate'] = c_rate
    df.loc[df['C_Rate'] > 0, 'Charge_Rate'] = c_rate[c_rate > 0]
    df.loc[df['C_Rate'] < 0, 'Discharge_Rate'] = c_rate[c_rate < 0]
    df.loc[df['C_Rate'] > 0, 'Half_Cycle_Type'] = 'Charging'
    df.loc[df['C_Rate'] < 0, 'Half_Cycle_Type'] = 'Discharging'
    df['Timestamp'] = pd.to_datetime(df.index)
    df['Hours Passed'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600
    df['Days Passed'] = df['Hours Passed'] / 24
    return df


def is_valid_half_cycle(current_soc: float, half_cycle_depth: float, window: pd.DataFrame, depth_threshold: float,
                        diff_threshold: float, current_status: str) -> bool:
    """
    Checks if the identified half-cycle meets the validity criteria based on specified thresholds.

    Args:
        current_soc (float): The current state of charge.
        half_cycle_depth (float): The calculated depth of the potential half-cycle.
        window (pd.DataFrame): The look-ahead window DataFrame containing future 'soc' values.
        depth_threshold (float): Minimum depth to consider a change as a half-cycle.
        diff_threshold (float): Minimum difference in 'soc' to distinguish between charging and discharging.
        current_status (str): The current charging status ('Charging' or 'Discharging').

    Returns:
        bool: True if the half-cycle is considered valid, False otherwise.
    """
    if window is None:
        next_max_soc = 0
        next_min_soc = 1
    else:
        next_max_soc = window['soc'].max()
        next_min_soc = window['soc'].min()

    if (current_status == 'Charging'
            and abs(half_cycle_depth) > depth_threshold
            and current_soc >= next_max_soc):  # There is a SOC value more than current SOC value
        return True
    elif (current_status == 'Discharging'
            and abs(half_cycle_depth) > depth_threshold
            and current_soc <= next_min_soc):  # There is a SOC value more than current SOC value
        return True
    else:
        return False


def calculate_rates_and_delta_fec(half_cycle: pd.Series) -> Tuple[float, float]:
    """
    Calculates charging or discharging rates and delta Full Equivalent Cycle (FEC) for a given half-cycle.

    Args:
        half_cycle (pd.Series): A series containing the start index, end index, depth, and type of half-cycle.

    Returns:
        Tuple[float, float]: The calculated rate (Charge_Rate or Discharge_Rate) and Delta_FEC.
    """
    duration_secs = (half_cycle['End_Index'] - half_cycle['Start_Index']).total_seconds()
    rate = abs(half_cycle['Depth']) * 3600 / duration_secs
    delta_fec = abs(half_cycle['Depth'] / 2)
    return rate, delta_fec


def calculate_half_cycle_depth_and_status(row: pd.Series,
                                          half_cycle_start: pd.Timestamp,
                                          df: pd.DataFrame,
                                          battery_capacity: float,
                                          window: pd.DataFrame,
                                          depth_threshold: float,
                                          diff_threshold: float) -> tuple[None, None] | tuple[Any, str]:
    """
    Calculates the depth and determines the status (Charging/Discharging) of a half-cycle.

    Args:
        row (pd.Series): The current row of the DataFrame being processed.
        half_cycle_start (pd.Timestamp): The start index of the current half-cycle.
        df (pd.DataFrame): The original DataFrame with 'soc' values.
        battery_capacity (float): The battery nominal capacity in Ah.
        window (pd.DataFrame): The DataFrame window to look ahead for max/min SoC values.
        depth_threshold (float): The depth threshold for identifying significant half-cycles.
        diff_threshold (float): The difference threshold for distinguishing between charging and discharging.

    Returns:
        Tuple[float, str]: The depth of the half-cycle and its status ('Charging' or 'Discharging'). Returns
        (None, None) if conditions are not met.
    """
    current_soc = row['soc']
    half_cycle_depth = (current_soc - df.at[half_cycle_start, 'soc']) * battery_capacity
    current_status = 'Charging' if half_cycle_depth > 0 else 'Discharging'

    # Conditions to identify valid half-cycle
    if is_valid_half_cycle(current_soc, half_cycle_depth, window, depth_threshold, diff_threshold, current_status):
        return half_cycle_depth, current_status
    return None, None


def identify_half_cycles(df: pd.DataFrame, battery_capacity: float, look_ahead_samples: int, depth_threshold: float,
                         diff_threshold: float) -> pd.DataFrame:
    """
    Identifies and processes half-cycles based on SoC changes and specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'soc', 'soc_change', and 'Sign_Change' columns.
        battery_capacity (float): The battery nominal capacity in Ah.
        look_ahead_samples (int): Number of samples to consider for look-ahead.
        depth_threshold (float): Minimum depth to consider a change as a half-cycle.
        diff_threshold (float): Minimum difference in 'soc' to distinguish between charging and discharging.

    Returns:
        pd.DataFrame: DataFrame containing identified half-cycles with start/end indices, depth, and type.
    """
    half_cycles = []
    half_cycle_start = df.index[0]
    prev_status = None

    for i in range(len(df) - 1):
        if look_ahead_samples is None:
            window = None
        else:
            window_end = min(i + look_ahead_samples, len(df) - 1)
            window = df.iloc[i + 1:window_end]
        current_idx = df.index[i]
        row = df.loc[current_idx]

        # Logic to identify and process half-cycles
        half_cycle_depth, current_status = calculate_half_cycle_depth_and_status(row,
                                                                                 half_cycle_start,
                                                                                 df,
                                                                                 battery_capacity,
                                                                                 window,
                                                                                 depth_threshold,
                                                                                 diff_threshold)

        if half_cycle_depth is None:  # Conditions not met to record a half-cycle
            continue

        # if current_status == prev_status:
        #     continue  # Skip if the status hasn't changed

        if current_status == 'Charging':
            start_idx = df.index[(df.index < current_idx) &
                                 ((df['soc'] - df.at[current_idx, 'soc']) * battery_capacity <= depth_threshold - half_cycle_depth)][-1]
        else:
            start_idx = df.index[(df.index < current_idx) &
                                 ((df['soc'] - df.at[current_idx, 'soc']) * battery_capacity >= - half_cycle_depth - depth_threshold)][-1]

        # Record the half-cycle
        half_cycles.append({
            'Start_Index': start_idx,
            'End_Index': current_idx,
            'Depth': half_cycle_depth,
            'Type': current_status
        })
        half_cycle_start = current_idx
        prev_status = current_status

    return pd.DataFrame(half_cycles)


def calculate_current_profile(soc_profile, battery_capacity):
    # Calculate change in SoC in fraction form (assuming SoC profile includes the initial SoC)
    delta_soc_fraction = np.diff(np.insert(soc_profile, 0, soc_profile[0]))
    dt = (soc_profile.index[1] - soc_profile.index[0]).seconds / 3600

    # Calculate the charge transferred for each minute in ampere-hours
    charge_transferred = delta_soc_fraction * battery_capacity

    # Convert charge transferred in ampere-hours back to current in amperes
    current_profile = charge_transferred / dt  # Conversion factor from hours to minutes

    return current_profile
