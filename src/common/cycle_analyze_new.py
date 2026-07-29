from typing import Any, Tuple, List, Dict
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from rainflow import *
from typing import Literal, get_args

profile_type_set = Literal[
    'soc_profile',
    'current_profile'
]

def augment_dataframe_with_half_cycles(df: pd.DataFrame, profile_type: profile_type_set, consider_original_current:str, battery_capacity: float = 3, method: str = 'half_cycle',
                                       depth_threshold: float = 0, diff_threshold: float = 1e-3, look_ahead: float = 6)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhances the input DataFrame with half-cycle information based on 'soc'/current values and specified thresholds.

    This function calculates forward-looking 'soc'/current changes, identifies half-cycles, and annotates the DataFrame
    with details about these cycles, including type, depth, and calculated rates.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index and a 'soc'/'I[A]' column.
        profile_type (profile_type_set): Specifies the given type of profile provided. ['soc_profile', 'current_profile']
        battery_capacity (float, optional): The battery nominal capacity in Ah, by default 3 Ah.
        method (str, optional): Method used to analyze the 'soc'/current profile. Defaults to the 'half_cycle' method.
        depth_threshold (float): Minimum depth change required to consider a half-cycle. Default is 0.
        diff_threshold (float): Minimum difference in 'soc'/current to distinguish charging from discharging. Default is 1e-3.
        look_ahead (float): Number of hours to look ahead when evaluating 'soc'/current changes. Default is 6.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the enhanced original DataFrame and a new DataFrame
                                           with detailed half-cycle information.
    """

    options = get_args(profile_type_set)
    assert profile_type in options, f"'{profile_type}' is not in 'profile_type_set'! {options}"

    if profile_type== 'soc_profile':
        df= calculate_soc_change(df)
        if look_ahead == 0:
            look_ahead_samples = None
        else:
            look_ahead_samples = get_look_ahead_samples(df, look_ahead)

        if method == 'half_cycle':
            cycles_df = identify_half_cycles(df, battery_capacity, look_ahead_samples, depth_threshold, diff_threshold)
        elif method == 'rainflow':
            cycles_df = identify_cycles_rainflow(df, battery_capacity=battery_capacity, depth_threshold=depth_threshold) 

        # Populate original DataFrame with half-cycle information and calculate rates
        df = populate_cycle_info(df, cycles_df, battery_capacity, method, profile_type, consider_original_current)

        # Handle potential NaN in the last soc_change due to the shift operation
        df['soc_change'].fillna(0, inplace=True) 

        return df, cycles_df
    elif profile_type== 'current_profile':

        #hier möglichst früh die Berechnung von 'soc' einfügen
        df= calculate_current_signchange(df)    
        if look_ahead == 0:
         look_ahead_samples = None
        else:
            look_ahead_samples = get_look_ahead_samples(df, look_ahead)

        if method == 'half_cycle':
            cycles_df = identify_half_cycles_current(df, battery_capacity, look_ahead_samples, depth_threshold, diff_threshold)
        elif method == 'rainflow':
            cycles_df = identify_cycles_rainflow_current(df, battery_capacity=battery_capacity, depth_threshold=depth_threshold) 

        # Populate original DataFrame with half-cycle information and calculate rates
        df = populate_cycle_info(df, cycles_df, battery_capacity, method, profile_type, consider_original_current)

        # Handle potential NaN in the last current_change due to the shift operation
        df['current_change'].fillna(0, inplace=True)

        return df, cycles_df


def calculate_current_signchange(df: pd.DataFrame) -> pd.DataFrame:     #modifiy and include function in augment_dataframe_with_half_cycles
    """
    Calculates the current signchange looking forward and updates the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a 'I[A]' column and datetime index.

    Returns:
        pd.DataFrame: Updated DataFrame with added 'current_change', 'Change_Sign', and 'Sign_Change' columns.
    """
    df['current_change'] = df['I[A]'].shift(-1) - df['I[A]']
    df['Change_Sign'] = np.sign(df['current_change'])
    df['Sign_Change'] = df['Change_Sign'] != df['Change_Sign'].shift(-1)
    return df
#how do we want the current profile to look like: welche spalten soll es haben, welche Einheit? - Strom in Ampere. Basytec zyklische Strom I[A] Profile als Beispiel nehmen
#update docstrings
#spalten umbenennen


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


def populate_cycle_info(df: pd.DataFrame, half_cycles_df: pd.DataFrame, battery_cappacity: float, 
                             method: str, profile_type: profile_type_set, consider_original_current: str) -> pd.DataFrame:
    """
    Populates the original DataFrame with half-cycle information including type, depth, and rates.

    Args:
        df (pd.DataFrame): The original DataFrame with datetime index.
        half_cycles_df (pd.DataFrame): DataFrame containing identified half-cycles information.
        battery_cappacity (float): The battery nominal capacity.
        method (str): Method used to analyze the SoC profile.
        profile_type (profile_type_set): Specifies the given type of profile provided to determine the 'current_profile' properly. ['soc_profile', 'current_profile']

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
        df.loc[start_index:end_index, 'Half_Cycle_Depth'] = half_cycle['Depth'] / battery_cappacity
        rate, delta_fec = calculate_rates_and_delta_fec(half_cycle, method)                             #Vermutung: muss nicht geändert werden: prüfen
        if method == 'half_cycle':
            df.at[end_index, 'Delta_FEC'] = delta_fec / battery_cappacity
        elif method == 'rainflow':
            df.at[end_index, 'Delta_FEC'] = half_cycle['Delta_FEC']

    df['FEC'] = df['Delta_FEC'].cumsum()
    if consider_original_current is None:
        if profile_type== 'soc_profile':
            current_profile= calculate_current_profile(df['soc'], battery_capacity=battery_cappacity)  
        elif profile_type== 'current_profile':
            current_profile= df['I[A]']
        #ergänze if Bedingung je nach Source:
        #Wenn soc als source muss calculate_current_profile durchgefürht werden, wenn current profile als source gilt Currte_profile gleich spalte im input df 
    else:
        current_profile=df[consider_original_current]
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


def is_valid_half_cycle(current_soc: float, half_cycle_depth: float, window: pd.DataFrame, depth_threshold: float,                  # prüfe, wird nicht mehr gebraucht
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


def calculate_rates_and_delta_fec(half_cycle: pd.Series, method: str) -> Tuple[float, float]:
    """
    Calculates charging or discharging rates and delta Full Equivalent Cycle (FEC) for a given half-cycle.

    Args:
        half_cycle (pd.Series): A series containing the start index, end index, depth, and type of half-cycle.
        method (str): Method used to analyze the SoC or current profile.

    Returns:
        Tuple[float, float]: The calculated rate (Charge_Rate or Discharge_Rate) and Delta_FEC.
    """
    duration_secs = (half_cycle['End_Index'] - half_cycle['Start_Index']).total_seconds()
    rate = abs(half_cycle['Depth']) * 3600 / duration_secs
    delta_fec = abs(half_cycle['Depth'] / 2)

    return rate, delta_fec


def calculate_half_cycle_depth_and_status(row: pd.Series,                                       # prüfe: wird nicht mehr gebraucht, 
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
    data = df['soc'].to_numpy()
    peaks, _ = find_peaks(data, height=0)
    valleys, _ = find_peaks(-data, height=-1)
    extreme_points = np.sort(np.concatenate((peaks, valleys)))
    extreme_values = data[extreme_points]
    # Calculate cycle depths
    cycle_depths = []
    for i in range(len(extreme_points) - 1):
        # Calculate the absolute difference between successive points
        depth = (extreme_values[i + 1] - extreme_values[i]) * battery_capacity
        cycle_depths.append(depth)

    # Pair these depths with their corresponding indices or intervals
    cycle_info = list(zip(extreme_points[:-1], extreme_points[1:], cycle_depths))

    idx_start = 0

    for i_start, i_end, depth in cycle_info:
        if abs(depth) <= depth_threshold and idx_start == 0:
            idx_start = i_start
            continue
        if abs(depth) <= depth_threshold:
            continue

        cycle_depth = ((df.iloc[i_end]['soc'] - df.iloc[(idx_start if idx_start != 0 else i_start)]['soc'])
                       * battery_capacity)

        status = _determine_status(depth, depth_threshold)

        half_cycles.append({
            'Start_Index': df.index[(idx_start if idx_start != 0 else i_start)],
            'End_Index': df.index[i_end],
            'Depth': cycle_depth,
            'Type': status
        })
        idx_start = 0    

    return pd.DataFrame(half_cycles)


def identify_half_cycles_current(df: pd.DataFrame, battery_capacity: float, look_ahead_samples: int, depth_threshold: float,
                         diff_threshold: float) -> pd.DataFrame:
    """
    Identifies and processes half-cycles based on Current changes and specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'I[A]', 'current_change', and 'Sign_Change' columns.
        battery_capacity (float): The battery nominal capacity in Ah.
        look_ahead_samples (int): Number of samples to consider for look-ahead.
        depth_threshold (float): Minimum depth to consider a change as a half-cycle.
        diff_threshold (float): Minimum difference in 'I[A]' to distinguish between charging and discharging.

    Returns:
        pd.DataFrame: DataFrame containing identified half-cycles with start/end indices, depth, and type.
    """
    #ermittle half cycles basierend auf Stromsignal
    half_cycles = []
    half_cycle_start = df.index[0]
    data = df['I[A]'].to_numpy()
    peaks, _ = find_peaks(data, height=0)                           #works also for I[A] signals?
    valleys, _ = find_peaks(-data, height=-1)
    extreme_points = np.sort(np.concatenate((peaks, valleys)))
    extreme_values = data[extreme_points]
    # Calculate cycle depths
    cycle_depths = []
    for i in range(len(extreme_points) - 1):
        # Calculate the absolute difference between successive points
        depth = (extreme_values[i + 1] - extreme_values[i]) * battery_capacity
        cycle_depths.append(depth)

    # Pair these depths with their corresponding indices or intervals
    cycle_info = list(zip(extreme_points[:-1], extreme_points[1:], cycle_depths))

    idx_start = 0

    for i_start, i_end, depth in cycle_info:
        if abs(depth) <= depth_threshold and idx_start == 0:
            idx_start = i_start
            continue
        if abs(depth) <= depth_threshold:
            continue

        cycle_depth = ((df.iloc[i_end]['I[A]'] - df.iloc[(idx_start if idx_start != 0 else i_start)]['I[A]'])
                       * battery_capacity)

        status = _determine_status(depth, depth_threshold)

        half_cycles.append({
            'Start_Index': df.index[(idx_start if idx_start != 0 else i_start)],
            'End_Index': df.index[i_end],
            'Depth': cycle_depth,
            'Type': status
        })
        idx_start = 0    

    return pd.DataFrame(half_cycles)


def identify_cycles_rainflow(df: pd.DataFrame, battery_capacity: float, depth_threshold: float) -> pd.DataFrame:
    """
    Identifies and processes rainflow based on SoC changes and specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'soc', 'soc_change', and 'Sign_Change' columns.
        battery_capacity (float): The battery nominal capacity in Ah.
        depth_threshold (float): Minimum depth to consider a change as a half-cycle.

    Returns:
        pd.DataFrame: DataFrame containing identified half-cycles with start/end indices, depth, and type.
    """
    cycles = []

    for rng, mean, count, i_start, i_end in extract_cycles(df['soc'].tolist()):
        depth = (df.iloc[i_end]['soc'] - df.iloc[i_start]['soc']) * battery_capacity

        status = _determine_status(depth, depth_threshold)

        # Record the cycle
        cycles.append({
            'Start_Index': df.index[i_start],
            'End_Index': df.index[i_end],
            'Depth': depth,
            'Type': status,
            'Delta_FEC': count * rng
        })

    cycles = _merge_overlapping_cycles(cycles)

    return pd.DataFrame(cycles)


def identify_cycles_rainflow_current(df: pd.DataFrame, battery_capacity: float, depth_threshold: float) -> pd.DataFrame:
    """
    Identifies and processes rainflow based on Current changes and specified thresholds.

    Args:
        df (pd.DataFrame): DataFrame with 'I[A]', 'current_change', and 'Sign_Change' columns.
        battery_capacity (float): The battery nominal capacity in Ah.
        depth_threshold (float): Minimum depth to consider a change as a half-cycle.

    Returns:
        pd.DataFrame: DataFrame containing identified half-cycles with start/end indices, depth, and type.
    """
    cycles = []

    for rng, mean, count, i_start, i_end in extract_cycles(df['I[A]'].tolist()):
        depth = (df.iloc[i_end]['I[A]'] - df.iloc[i_start]['I[A]']) * battery_capacity

        status = _determine_status(depth, depth_threshold)

        # Record the cycle
        cycles.append({
            'Start_Index': df.index[i_start],
            'End_Index': df.index[i_end],
            'Depth': depth,
            'Type': status,
            'Delta_FEC': count * rng
        })

    cycles = _merge_overlapping_cycles(cycles)

    return pd.DataFrame(cycles)


def _merge_overlapping_cycles(cycles: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Merges overlapping cycles in a list of cycle dictionaries based on their start and end indices.

    Each cycle dictionary should have at least the keys 'Start_Index', 'End_Index', 'Depth', and 'Delta_FEC'.
    Overlapping cycles are merged by averaging their 'Depth' if both depths are positive, or averaging them
    as negative values if the original depth is negative. The 'Delta_FEC' values are summed.

    Args:
        cycles (List[Dict[str, float]]): A list of dictionaries, where each dictionary represents a cycle
            with keys 'Start_Index', 'End_Index', 'Depth', and 'Delta_FEC'.

    Returns:
        List[Dict[str, float]]: A list of dictionaries representing the merged cycles.

    Example:
        input_cycles = [
            {'Start_Index': 0, 'End_Index': 2, 'Depth': 10.0, 'Delta_FEC': 0.5},
            {'Start_Index': 1, 'End_Index': 3, 'Depth': 15.0, 'Delta_FEC': 0.7}
        ]
        merged_cycles = merge_overlapping_cycles(input_cycles)
        # Output: [{'Start_Index': 0, 'End_Index': 3, 'Depth': 12.5, 'Delta_FEC': 1.2}]
    """
    if not cycles:
        return []

    # Sort cycles by start index to ensure overlaps are handled consecutively
    cycles.sort(key=lambda x: x['Start_Index'])
    merged_cycles = [cycles[0]]  # Start with the first cycle as the initial merged cycle

    for cycle in cycles[1:]:  # Start from the second element
        last_merged = merged_cycles[-1]

        # Check if there is an overlap
        if (last_merged['Start_Index'] <= cycle['Start_Index'] <= last_merged['End_Index']) and (
                last_merged['Start_Index'] <= cycle['End_Index'] <= last_merged['End_Index']):
            # Average the 'Depth' and 'Delta_FEC' values
            # if last_merged['Depth'] > 0:
            #     last_merged['Depth'] = (last_merged['Depth'] + cycle['Depth']) / 2
            # else:
            #     last_merged['Depth'] = (last_merged['Depth'] - cycle['Depth']) / 2
            last_merged['Depth'] = last_merged['Depth'] + cycle['Depth']
            last_merged['Delta_FEC'] = last_merged['Delta_FEC'] + cycle['Delta_FEC']
        else:
            # If not overlapping, simply add the new cycle to the list
            merged_cycles.append(cycle)

    return merged_cycles


def _determine_status(depth: float, depth_threshold: float) -> str:
    """
    Determines the status based on the given depth relative to a threshold.

    Args:
        depth (float): The current depth or level measurement.
        depth_threshold (float): The threshold value for determining status.

    Returns:
        str: Returns 'Charging' if depth is greater than the threshold,
             'Discharging' if depth is less than the negative of the threshold,
             otherwise 'Idle'.
    """
    if depth > depth_threshold:
        status = 'Charging'
    elif depth < -depth_threshold:
        status = 'Discharging'
    else:
        status = 'Idle'

    return status


def calculate_current_profile(soc_profile: pd.Series, battery_capacity: float) -> List:
    """
    Calculates the current profile based on changes in the State of Charge (SoC) and battery capacity.

    This function assumes that the input SoC profile includes the initial SoC and that the timestamps
    are evenly spaced.

    Args:
        soc_profile (pd.Series): A pandas Series where the index is datetime and values are SoC percentages.
        battery_capacity (float): The total capacity of the battery in ampere-hours (Ah).

    Returns:
        List: A pandas Series representing the current profile in amperes, indexed by the same timestamps.
    """
    # Calculate change in SoC in fraction form (assuming SoC profile includes the initial SoC)
    delta_soc_fraction = np.diff(np.insert(soc_profile.values, 0, soc_profile.iloc[0]))

    # Calculate the time difference in hours between measurements
    dt = (soc_profile.index[1] - soc_profile.index[0]).total_seconds() / 3600

    # Calculate the charge transferred for each interval in ampere-hours
    charge_transferred = delta_soc_fraction * battery_capacity

    # Convert charge transferred in ampere-hours back to current in amperes
    current_profile = charge_transferred / dt  # Conversion factor from hours to the interval duration

    return current_profile
