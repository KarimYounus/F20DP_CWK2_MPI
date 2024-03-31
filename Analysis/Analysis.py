import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tabulate as tb
from scipy.signal import savgol_filter

pd.set_option('display.max_columns', None)

PARA_COLS = ["Upper", "Core Count", "chunking"]
SEQ_COLS = ["Upper"]
DATASETS = (15000, 30000, 100000)
CORE_COUNTS = [2, 4, 8, 16, 32, 64, 92, 128, 160, 192]
SCHEDULING_STRATEGIES = ["equal", "dynamic"]
DYNAMIC_COL = 'mediumslateblue'
EQUAL_COL = 'teal'
WIN_SIZE = 7
POLY_ORDER = 2
path = "Benchmarks"


def main():
    # Import CSV files
    seq_df, para_df = import_csv(path)

    # # COMPUTE UNIQUE MEAN EXECUTION TIME
    para_mean = compute_unique_mean_execution_time(para_df, PARA_COLS)
    seq_mean = compute_unique_mean_execution_time(seq_df, SEQ_COLS)
    # # COMPUTE SPEEDUP
    seq_mean_rt = seq_mean['Mean Execution Time'].unique()
    para_processed = compute_speedup(para_mean, seq_mean_rt)
    # # COMPUTE EFFICIENCY
    para_processed['Efficiency'] = para_processed['Speedup'] / para_processed['Core Count']

    # plot_main_graph_speedup(para_processed)
    plot_main_graph_runtime(para_processed)
    # plot_main_graph_efficiency(para_processed)
    table = generate_performance_table(para_df)
    print(table)


def plot_main_graph_speedup(processed_df):
    """
    Plot the speedup vs chunk size against the core count for each dataset
    :param processed_df:
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))  # Share the y-axis among all subplots

    for i, upper in enumerate(DATASETS):
        # Filter data for each subplot
        ds = select_combinations(processed_df, upper=upper)

        # Plot each chunk srtat
        for strat in SCHEDULING_STRATEGIES:
            subset = ds[ds['chunking'] == strat]
            if strat == 'dynamic':
                colour = DYNAMIC_COL
            else:
                colour = EQUAL_COL
            axes[i].plot(subset['Core Count'], subset['Speedup'], label=f'Chunking = {strat}', color=colour)
            # Smooth the data using Savitzky-Golay filter
            window_size = WIN_SIZE
            poly_order = POLY_ORDER
            smoothed_speedup = savgol_filter(subset['Speedup'], window_size, poly_order)

            # Plot the smoothed curve with a faded line
            axes[i].plot(subset['Core Count'], smoothed_speedup, color=colour, alpha=0.7, linestyle='dotted')

        # Add Ideal speedup line
        axes[i].plot(CORE_COUNTS, CORE_COUNTS, label='Ideal Speedup', linestyle='--', color='gray')

        # Customizing the subplot
        axes[i].set_title(f'DS{i + 1}', fontsize=16)
        axes[i].set_xlabel('Core Count')
        axes[i].grid(True)
        axes[i].set_ylabel('Speedup')
        axes[i].legend()

    plt.suptitle('Parallel vs Sequential Speedup across Core Count\n', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def plot_main_graph_runtime(processed_df):
    """
    Plot the runtime vs core count for each dataset
    :param processed_df:
    :return:
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

    for i, upper in enumerate(DATASETS):
        # Filter data for each subplot
        ds = select_combinations(processed_df, upper=upper)

        for strat in SCHEDULING_STRATEGIES:
            subset = ds[ds['chunking'] == strat]
            if strat == 'dynamic':
                colour = DYNAMIC_COL
            else:
                colour = EQUAL_COL
            axes[i].plot(subset['Core Count'], np.log10(subset['Mean Execution Time']), label=f'Chunking = {strat}', color=colour)

            # Smooth the data using Savitzky-Golay filter
            window_size = WIN_SIZE
            poly_order = POLY_ORDER
            smoothed_runtime = savgol_filter(np.log10(subset['Mean Execution Time']), window_size, poly_order)

            # Plot the smoothed curve with a faded line
            axes[i].plot(subset['Core Count'], smoothed_runtime, color=colour, alpha=0.7, linestyle='dotted')

        # Customizing the subplot
        axes[i].set_title(f'DS{i + 1}', fontsize=16)
        axes[i].set_xlabel('Core Count', fontsize=12)
        axes[i].grid(True)
        axes[i].set_ylabel('Mean Execution Time (Log10 of s)', fontsize=12)
        axes[i].legend()

    plt.suptitle('Parallel Mean Execution Time across Core Count\n', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


def plot_main_graph_efficiency(processed_df):
    """
    Plot the efficiency vs core count for each dataset
    :param processed_df:
    :return:
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18), sharey=True)

    for i, upper in enumerate(DATASETS):
        # Filter data for each subplot
        ds = select_combinations(processed_df, upper=upper)

        # Plot each chunk size
        for strat in SCHEDULING_STRATEGIES:
            subset = ds[ds['chunking'] == strat]
            if strat == 'dynamic':
                colour = DYNAMIC_COL
            else:
                colour = EQUAL_COL
            axes[i].plot(subset['Core Count'], subset['Efficiency'], label=f'Chunking = {strat}', color=colour)

            # Smooth the data using Savitzky-Golay filter
            window_size = WIN_SIZE
            poly_order = POLY_ORDER
            smoothed_speedup = savgol_filter(subset['Efficiency'], window_size, poly_order)

            # Plot the smoothed curve with a faded line
            axes[i].plot(subset['Core Count'], smoothed_speedup, color=colour, alpha=0.7, linestyle='dotted')

        # Customizing the subplot
        axes[i].set_title(f'DS{i + 1}', fontsize=16)
        axes[i].set_xlabel('Core Count')
        axes[i].grid(True)
        axes[i].set_ylabel('Efficiency')
        axes[i].legend()

    plt.suptitle('Parallel Efficiency across Core Count\n', fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig


# def plot_amd_vs_intel_runtime(amd, intel):
#     """
#     Plot the speedup vs core count for each dataset
#     :param processed_df:
#     :return:
#     """
#     plt.figure(figsize=(10, 6))
#
#     for i, gcd in enumerate(GCD_VERSION):
#         # Filter data for each subplot
#         ds_amd = select_combinations(amd, gcd_version=gcd)
#         ds_intel = select_combinations(intel, gcd_version=gcd)
#
#         plt.plot(ds_amd['Upper'], ds_amd['Mean Execution Time'], label=f'AMD, GCD = {gcd}')
#         plt.plot(ds_intel['Upper'], ds_intel['Mean Execution Time'], label=f'Intel, GCD = {gcd}')
#
#     plt.title('Mean Execution Time vs. Upper Bound')
#     plt.xlabel('Upper Bound')
#     plt.grid(True)
#     plt.ylabel('Mean Execution Time (s)')
#     plt.legend()
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.show()


def plot_speedup_vs_core_count(processed_df):
    """
    Plot the speedup vs core count for each dataset
    :param processed_df:
    :return:
    """
    ds1 = select_combinations(processed_df, upper=15000, chunking='dynamic')
    ds2 = select_combinations(processed_df, upper=30000, chunking='dynamic')
    ds3 = select_combinations(processed_df, upper=100000, chunking='dynamic')

    plt.figure(figsize=(10, 6))
    plt.plot(ds3['Core Count'], ds3['Speedup'], label='DS3')
    plt.plot(ds2['Core Count'], ds2['Speedup'], label='DS2')
    plt.plot(ds1['Core Count'], ds1['Speedup'], label='DS1')
    plt.legend()
    plt.xlabel('Core Count')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup vs Core Count')
    plt.plot()


def plot_scheduling_strategy(processed_df):
    """
    Plot the speedup vs scheduling strategy
    :param processed_df:
    :return:
    """

    dynamic = select_combinations(processed_df, chunking='dynamic', chunk_size=10, core_count=32)
    static = select_combinations(processed_df, chunking='equal', chunk_size=10, core_count=32)

    plt.figure(figsize=(10, 6))
    for i in range(len(dynamic)):
        plt.bar(dynamic['Scheduling Strategy'], dynamic['Speedup'], label='Dynamic')
    plt.bar(static['Scheduling Strategy'], static['Speedup'], label='Equal')
    plt.xlabel('Scheduling Strategy')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup vs Scheduling Strategy')
    plt.show()


def generate_performance_table(df):
    """
    Generates a performance summary table for each combination of core count and dataset size.

    :param df: DataFrame containing the columns 'Core Count', 'Dataset Size', and 'Execution Time'.
    :return: A DataFrame with the median, mean, and differences between mean and extreme values for each group.

    """

    df_sorted = df.sort_values(by=['Core Count', 'Upper', 'chunking'])

    # Group the data by 'Core Count' and 'Dataset Size' and calculate the required statistics
    performance_stats = df_sorted.groupby(['Core Count', 'Upper', 'chunking']).agg(
        First_Runtime=('Execution Time', lambda x: x.iloc[0] if len(x) > 0 else None),
        Second_Runtime=('Execution Time', lambda x: x.iloc[1] if len(x) > 1 else None),
        Third_Runtime=('Execution Time', lambda x: x.iloc[2] if len(x) > 2 else None),
        Median_Runtime=('Execution Time', 'median'),
        Mean_Runtime=('Execution Time', 'mean'),
        Min_Runtime=('Execution Time', 'min'),
        Max_Runtime=('Execution Time', 'max'),
        # Max_Speedup=('Speedup', 'max'),
    ).reset_index()

    # Calculate the differences between the mean and extreme values
    performance_stats['Mean-Min Difference'] = performance_stats['Mean_Runtime'] - performance_stats['Min_Runtime']
    performance_stats['Mean-Max Difference'] = performance_stats['Max_Runtime'] - performance_stats['Mean_Runtime']

    # Convert time to a suitable unit, e.g., milliseconds, for readability
    # Round the values for better readability
    performance_stats = performance_stats.round(3)

    # Sort the DataFrame by 'Core Count'
    performance_stats = performance_stats.sort_values(by=['Upper', 'Core Count'])

    table_string = tb.tabulate(performance_stats, headers='keys', tablefmt='github', showindex=False)

    return table_string


def compute_speedup(df, seq_exec_time):
    """
    Compute the speedup for each parallel entry, comparing the sequential execution time to the parallel execution time.

    :param df: A pandas DataFrame containing the parallel mean execution times.
    :param seq_exec_time: A dictionary with 'Upper' bounds as keys and sequential execution times as values.
    :return: DataFrame with the 'Speedup' column updated.
    """
    # Ensure seq_exec_time is a dictionary for direct mapping: {15000: time1, 30000: time2, 100000: time3}
    for i in range(len(seq_exec_time)):
        upper_bound = 0
        match i:
            case 0:
                upper_bound = 15000
            case 1:
                upper_bound = 30000
            case 2:
                upper_bound = 100000

        mask = df['Upper'] == upper_bound
        df.loc[mask, 'Speedup'] = seq_exec_time[i] / df.loc[mask, 'Mean Execution Time']

    return df


def compute_unique_mean_execution_time(df, group_by_columns):
    """
    Computes the mean execution time for each unique combination of specified parameters in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing execution times and other related parameters.
    - group_by_columns (list): A list of column names to define unique combinations for which to compute the mean execution time.

    Returns:
    - pd.DataFrame: A DataFrame with each row representing a unique combination of parameters and its corresponding mean execution time.
    """
    # Group by the specified columns and calculate the mean execution time
    unique_means_df = df.groupby(group_by_columns)['Execution Time'].agg('mean').reset_index()

    # Optionally, rename the aggregated column for clarity
    unique_means_df.rename(columns={'Execution Time': 'Mean Execution Time'}, inplace=True)

    return unique_means_df


def select_combinations(df, upper=None, core_count=None, chunking=None) -> pd.DataFrame:
    """
    Selects data combinations based on given criteria from the DataFrame.

    :param upper: The upper bound (optional).
    :param DataFrame containing the benchmark data.
    :param core_count: The number of cores (optional).
    :param chunking: The scheduling strategy ('static', 'dynamic', 'guided') (optional).
    :param chunk_size: The size of the chunks (optional).
    :param gcd_version: The version of the GCD algorithm, default is 'Euclid'.

    :return: A Filtered DataFrame based on the given criteria.
    """

    # Filter based on other criteria
    if upper is not None:
        filtered_df = df[df['Upper'] == upper]
    if core_count is not None:
        filtered_df = df[df['Core Count'] == core_count]
    if chunking is not None:
        filtered_df = df[df['chunking'] == chunking]

    return filtered_df


def import_csv(csv_path: str):
    def clean_column_names(df):
        # Strip whitespace and remove quotation marks from column names
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\'', '')
        return df

    def clean_strings(df):
        # Strip whitespace and remove quotation marks from strings
        df = df.map(lambda x: x.strip().replace('"', '').replace('\'', '') if isinstance(x, str) else x)
        return df

    files = os.listdir(csv_path)
    raw_seq_data = []
    raw_para_data = []

    seq_pattern = r"seq.csv"
    para_pattern = r"benchmark_(\d+).csv"
    dynamic_pattern = r"benchmark_(\d+)_dynamic.csv"

    for file in files:
        if file.endswith(".csv"):
            if re.match(seq_pattern, file):
                raw_seq_data.append(pd.read_csv(os.path.join(csv_path, file), dtype={'Scheduling Strategy': str}))
                # print("Added Seq")
            elif re.match(para_pattern, file):
                df = pd.read_csv(os.path.join(csv_path, file))
                df['chunking'] = 'equal'  # Add the 'chunking' column with value 'equal'
                raw_para_data.append(df)

                # print("Added Para")
            elif re.match(dynamic_pattern, file):
                df = pd.read_csv(os.path.join(csv_path, file))
                df['chunking'] = 'dynamic'  # Add the 'chunking' column with value 'dynamic'
                raw_para_data.append(df)
                # print("Added Dynamic")
            else:
                print(f"Unknown file format: {file}")

    seq_df = clean_column_names(clean_strings(pd.concat(raw_seq_data, ignore_index=True)))
    para_df = clean_column_names(clean_strings(pd.concat(raw_para_data, ignore_index=True)))

    return seq_df, para_df


main()
