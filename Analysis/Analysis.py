import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import tabulate as tb

pd.set_option('display.max_columns', None)

COLUMNS = columns_to_group = ["Filename", "Lower", "Upper", "Core Count",
                              "GCD Version", "Scheduling Strategy", "Chunk Size"]
DATASETS = (15000, 30000, 100000)
CORE_COUNTS = [2, 4, 8, 16, 32, 64]
SCHEDULING_STRATEGIES = ["static", "dynamic", "guided"]
CHUNK_SIZES = [1, 10, 20, 40, 60, 80, 100, 120, 140, 160]
GCD_VERSION = ["Euclid", "Binary"]
path = "Benchmarks"


def main():
    # Import CSV files
    seq_df, para_df = import_csv(path, False)
    amd_df, intel_df = import_csv(path + "/Seq", True)
    euclid_seq_df = seq_df.loc[seq_df['GCD Version'] == 'Euclid']

    # COMPUTE UNIQUE MEAN EXECUTION TIME
    para_mean = compute_unique_mean_execution_time(para_df, COLUMNS)
    seq_mean = compute_unique_mean_execution_time(euclid_seq_df, COLUMNS)
    amd_mean = compute_unique_mean_execution_time(amd_df, COLUMNS)
    intel_mean = compute_unique_mean_execution_time(intel_df, COLUMNS)
    # COMPUTE SPEEDUP
    seq_mean_rt = seq_mean['Mean Execution Time'].unique()
    para_processed = compute_speedup(para_mean, seq_mean_rt)
    # COMPUTE EFFICIENCY
    para_processed['Efficiency'] = para_processed['Speedup'] / para_processed['Core Count']

    # plot_speedup_vs_chunk_size(para_processed)

    table = generate_performance_table(para_processed)
    print(table)
    # PLOT


def plot_speedup_vs_chunk_size(processed_df):
    """
    Plot the speedup vs chunk size for each dataset
    :param processed_df:
    :return:
    """
    ds1 = select_combinations(processed_df, upper=100000, scheduling_strategy='static', core_count=4)
    ds2 = select_combinations(processed_df, upper=100000, scheduling_strategy='guided', core_count=4)
    ds3 = select_combinations(processed_df, upper=100000, scheduling_strategy='dynamic', core_count=4)

    plt.figure(figsize=(10, 6))
    plt.plot(ds3['Chunk Size'], ds3['Speedup'], label='Dynamic')
    plt.plot(ds2['Chunk Size'], ds2['Speedup'], label='Guided')
    plt.plot(ds1['Chunk Size'], ds1['Speedup'], label='Static')
    plt.legend()
    plt.xlabel('Chunk Size')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup vs Chunk Size for Scheduling Strategies, Core Count = 4, DS3')
    plt.show()


def plot_main_graph_speedup(processed_df):
    """
    Plot the speedup vs chunk size against the core count for each dataset
    :param processed_df:
    :return:
    """

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))  # Share the y-axis among all subplots

    for i, upper in enumerate(DATASETS):
        # Filter data for each subplot
        ds = select_combinations(processed_df, upper=upper, scheduling_strategy='dynamic')

        # Plot each chunk size
        for chunk_size in CHUNK_SIZES:
            subset = ds[ds['Chunk Size'] == chunk_size]
            axes[i].plot(subset['Core Count'], subset['Speedup'], label=f'Chunk size = {chunk_size}')

            # Annotate the best-performing chunk size at incremental core counts
            for core_count in CORE_COUNTS:
                # Filter to the current core count
                core_subset = ds[ds['Core Count'] == core_count]
                # Find the best chunk size for the current core count
                if not core_subset.empty:
                    best_chunk_size = core_subset.loc[core_subset['Speedup'].idxmax()]['Chunk Size']
                    best_speedup = core_subset['Speedup'].max()
                    # Annotate the best chunk size
                    axes[i].annotate(f'{best_chunk_size}',
                                     xy=(core_count, best_speedup),
                                     xytext=(3, 3),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     fontsize=10,
                                     bbox=dict(boxstyle='round,pad=0.01', fc='white', alpha=0.3))

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
        ds = select_combinations(processed_df, upper=upper, scheduling_strategy='dynamic')

        # Plot each chunk size
        for chunk_size in CHUNK_SIZES:
            subset = ds[ds['Chunk Size'] == chunk_size]
            axes[i].plot(subset['Core Count'], subset['Mean Execution Time'], label=f'Chunk size = {chunk_size}')

            # Annotate the best-performing chunk size at incremental core counts
            for core_count in CORE_COUNTS:
                # Filter to the current core count
                core_subset = ds[ds['Core Count'] == core_count]
                # Find the best chunk size for the current core count
                if not core_subset.empty:
                    best_chunk_size = core_subset.loc[core_subset['Mean Execution Time'].idxmax()]['Chunk Size']
                    best_runtime = core_subset['Mean Execution Time'].max()
                    # Annotate the best chunk size
                    axes[i].annotate(f'{best_chunk_size}',
                                     xy=(core_count, best_runtime),
                                     xytext=(3, 3),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     fontsize=10,
                                     bbox=dict(boxstyle='round,pad=0.01', fc='white', alpha=0.3))

        # Customizing the subplot
        axes[i].set_title(f'DS{i + 1}', fontsize=16)
        axes[i].set_xlabel('Core Count', fontsize=12)
        axes[i].grid(True)
        axes[i].set_ylabel('Mean Execution Time (s)', fontsize=12)
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
        ds = select_combinations(processed_df, upper=upper, scheduling_strategy='dynamic')

        # Plot each chunk size
        for chunk_size in CHUNK_SIZES:
            subset = ds[ds['Chunk Size'] == chunk_size]
            axes[i].plot(subset['Core Count'], subset['Efficiency'], label=f'Chunk size = {chunk_size}')

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


def plot_main_graph_sched(processed_df):
    """
    Plot the speedup vs chunk size for each scheduling strategy
    :param processed_df:
    :return:
    """

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)  # Share the y-axis among all subplots

    for i, sched in enumerate(SCHEDULING_STRATEGIES):
        # Filter data for each subplot
        ds = select_combinations(processed_df, upper=100000, scheduling_strategy=sched)

        # Plot each chunk size
        for chunk_size in CHUNK_SIZES:
            subset = ds[ds['Chunk Size'] == chunk_size]
            axes[i].plot(subset['Core Count'], subset['Speedup'], label=f'Chunk size = {chunk_size}')

        # Customizing the subplot
        axes[i].set_title(f'Scheduling Strategy: {sched}')
        axes[i].set_xlabel('Core Count')
        axes[i].grid(True)
        if i == 0:
            axes[i].set_ylabel('Speedup')
        axes[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    return fig


def plot_amd_vs_intel_runtime(amd, intel):
    """
    Plot the speedup vs core count for each dataset
    :param processed_df:
    :return:
    """
    plt.figure(figsize=(10, 6))

    for i, gcd in enumerate(GCD_VERSION):
        # Filter data for each subplot
        ds_amd = select_combinations(amd, gcd_version=gcd)
        ds_intel = select_combinations(intel, gcd_version=gcd)

        plt.plot(ds_amd['Upper'], ds_amd['Mean Execution Time'], label=f'AMD, GCD = {gcd}')
        plt.plot(ds_intel['Upper'], ds_intel['Mean Execution Time'], label=f'Intel, GCD = {gcd}')

    plt.title('Mean Execution Time vs. Upper Bound')
    plt.xlabel('Upper Bound')
    plt.grid(True)
    plt.ylabel('Mean Execution Time (s)')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_speedup_vs_core_count(processed_df):
    """
    Plot the speedup vs core count for each dataset
    :param processed_df:
    :return:
    """
    ds1 = select_combinations(processed_df, upper=15000, scheduling_strategy='dynamic', chunk_size=80)
    ds2 = select_combinations(processed_df, upper=30000, scheduling_strategy='dynamic', chunk_size=80)
    ds3 = select_combinations(processed_df, upper=100000, scheduling_strategy='dynamic', chunk_size=80)

    plt.figure(figsize=(10, 6))
    plt.plot(ds3['Core Count'], ds3['Speedup'], label='DS3')
    plt.plot(ds2['Core Count'], ds2['Speedup'], label='DS2')
    plt.plot(ds1['Core Count'], ds1['Speedup'], label='DS1')
    plt.legend()
    plt.xlabel('Core Count')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup vs Core Count')
    plt.show()


def plot_scheduling_strategy(processed_df):
    """
    Plot the speedup vs scheduling strategy
    :param processed_df:
    :return:
    """

    dynamic = select_combinations(processed_df, scheduling_strategy='dynamic', chunk_size=10, core_count=32)
    static = select_combinations(processed_df, scheduling_strategy='static', chunk_size=10, core_count=32)
    guided = select_combinations(processed_df, scheduling_strategy='guided', chunk_size=10, core_count=32)

    plt.figure(figsize=(10, 6))
    for i in range(len(dynamic)):
        plt.bar(dynamic['Scheduling Strategy'], dynamic['Speedup'], label='Dynamic')
    plt.bar(static['Scheduling Strategy'], static['Speedup'], label='Static')
    plt.bar(guided['Scheduling Strategy'], guided['Speedup'], label='Guided')
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

    df_sorted = df.sort_values(by=['Core Count', 'Upper', 'Mean Execution Time'])

    # Group the data by 'Core Count' and 'Dataset Size' and calculate the required statistics
    performance_stats = df_sorted.groupby(['Core Count', 'Upper']).agg(
        # First_Runtime=('Execution Time', lambda x: x.iloc[0] if len(x) > 0 else None),
        # Second_Runtime=('Execution Time', lambda x: x.iloc[1] if len(x) > 1 else None),
        # Third_Runtime=('Execution Time', lambda x: x.iloc[2] if len(x) > 2 else None),
        # Median_Runtime=('Execution Time', 'median'),
        # Mean_Runtime=('Execution Time', 'mean'),
        # Min_Runtime=('Execution Time', 'min'),
        # Max_Runtime=('Execution Time', 'max'),
        Max_Speedup=('Speedup', 'max'),
    ).reset_index()

    # Calculate the differences between the mean and extreme values
    # performance_stats['Mean-Min Difference'] = performance_stats['Mean_Runtime'] - performance_stats['Min_Runtime']
    # performance_stats['Mean-Max Difference'] = performance_stats['Max_Runtime'] - performance_stats['Mean_Runtime']

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


def select_combinations(df, upper=None, core_count=None, scheduling_strategy=None, chunk_size=None,
                        gcd_version='Euclid') -> pd.DataFrame:
    """
    Selects data combinations based on given criteria from the DataFrame.

    :param upper: The upper bound (optional).
    :param DataFrame containing the benchmark data.
    :param core_count: The number of cores (optional).
    :param scheduling_strategy: The scheduling strategy ('static', 'dynamic', 'guided') (optional).
    :param chunk_size: The size of the chunks (optional).
    :param gcd_version: The version of the GCD algorithm, default is 'Euclid'.

    :return: A Filtered DataFrame based on the given criteria.
    """
    # Filter based on GCD Version
    filtered_df = df[df['GCD Version'] == gcd_version]

    # Filter based on other criteria
    if upper is not None:
        filtered_df = filtered_df[filtered_df['Upper'] == upper]
    if core_count is not None:
        filtered_df = filtered_df[filtered_df['Core Count'] == core_count]
    if scheduling_strategy is not None:
        filtered_df = filtered_df[filtered_df['Scheduling Strategy'] == scheduling_strategy]
    if chunk_size is not None:
        filtered_df = filtered_df[filtered_df['Chunk Size'] == chunk_size]

    return filtered_df


def import_csv(csv_path: str, seq=False):
    def clean_column_names(df):
        # Strip whitespace and remove quotation marks from column names
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\'', '')
        return df

    def clean_strings(df):
        # Strip whitespace and remove quotation marks from strings
        df = df.map(lambda x: x.strip().replace('"', '').replace('\'', '') if isinstance(x, str) else x)
        return df

    files = os.listdir(csv_path)
    seq_df = []
    para_df = []

    seq_pattern = ""
    para_pattern = ""

    # regex
    seq_pattern = r"seq_benchmark_ds(\d+)_gcd(\d+).csv"
    para_pattern = r"para_benchmark_ds(\d+).csv"

    if seq:
        seq_pattern = r"seq_amd.csv"
        para_pattern = r"seq_intel.csv"

    for file in files:
        if file.endswith(".csv"):
            if re.match(seq_pattern, file):
                seq_df.append(pd.read_csv(os.path.join(csv_path, file), dtype={'Scheduling Strategy': str}))
                # print("Added Seq")
            elif re.match(para_pattern, file):
                para_df.append(pd.read_csv(os.path.join(csv_path, file)))
                # print("Added Para")
            else:
                print(f"Unknown file format: {file}")

    return clean_strings(clean_column_names(pd.concat(seq_df, ignore_index=True))), clean_strings(clean_column_names(
        pd.concat(para_df, ignore_index=True)))


main()
