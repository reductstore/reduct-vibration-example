"""Plot benchmark results"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

CSV_FILE_PATH = "benchmark_results.csv"


def prepare_csv(csv_file_path: str):
    """Prepare a CSV file for writing."""
    columns = ["Database", "Frequency (Hz)", "Write Time (s)", "Read Time (s)"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file_path, index=False)


def write_to_csv(
    csv_file_path: str,
    database: str,
    frequency: int,
    write_time: float,
    read_time: float,
):
    """Write benchmark results to the CSV file."""
    df = pd.DataFrame(
        {
            "Database": [database],
            "Frequency (Hz)": [frequency],
            "Write Time (s)": [write_time],
            "Read Time (s)": [read_time],
        }
    )
    df.to_csv(csv_file_path, mode="a", header=False, index=False)


def read_benchmark_results(csv_file_path: str) -> pd.DataFrame:
    """Read benchmark results from a CSV file."""
    return pd.read_csv(csv_file_path)


def plot_benchmark_results(df: pd.DataFrame, path: Path = Path(".")):
    """Plot benchmark results using Seaborn's catplot with log scale."""
    df_melted = pd.melt(
        df,
        id_vars=["Database", "Frequency (Hz)"],
        value_vars=["Write Time (s)", "Read Time (s)"],
        var_name="Metric",
        value_name="Time (s)",
    )

    g = sns.catplot(
        data=df_melted,
        x="Frequency (Hz)",
        y="Time (s)",
        hue="Database",
        col="Metric",
        kind="point",
        markers=["o", "s"],
        linestyles=["-", "--"],
        height=5,
        aspect=1.5,
        capsize=0.2,
        palette="Set2",
        legend_out=False,
    )

    for ax in g.axes.flat:
        ax.set_yscale("log")

    g.set_axis_labels("Frequency (Hz)", "Time (s)")
    g.set_xticklabels(rotation=45)
    g.set_titles("{col_name}")
    g.despine(left=True)

    plt.tight_layout()
    plt.savefig(path / "benchmark_results.png")
    plt.show()


if __name__ == "__main__":
    data = read_benchmark_results(CSV_FILE_PATH)
    plot_benchmark_results(data)
