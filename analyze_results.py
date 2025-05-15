import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and visualize attack results")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results_logs",
        help="Directory containing result logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="all",
        choices=["success_rate", "perturbation", "time", "transferability", "all"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--separate_methods",
        action="store_true",
        help="Generate separate graphs for each attack method",
    )
    return parser.parse_args()


def collect_all_results(log_dir):
    """Collect all CSV results into a single DataFrame"""
    csv_files = glob.glob(os.path.join(log_dir, "summary_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return None

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def plot_success_rates(df, output_dir):
    """Plot success rates for all models and attack methods"""
    plt.figure(figsize=(14, 10))

    # Convert success_rate to numeric and to percentage
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce") * 100

    # Create the grouped bar plot
    ax = sns.barplot(x="model_name", y="success_rate", hue="attack_method", data=df)

    plt.title("Attack Success Rate by Model and Method", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter())

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.legend(title="Attack Method", loc="upper right")
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "success_rates.png")
    plt.savefig(output_path, dpi=300)
    print(f"Success rate plot saved to {output_path}")
    return output_path


def plot_method_specific_success_rates(df, output_dir):
    """Plot success rates for each attack method separately"""
    # Get unique attack methods
    attack_methods = df["attack_method"].unique()
    output_paths = []

    for method in attack_methods:
        # Filter data for this method
        method_df = df[df["attack_method"] == method]

        plt.figure(figsize=(12, 8))

        # Convert success_rate to numeric and to percentage
        method_df["success_rate"] = pd.to_numeric(
            method_df["success_rate"], errors="coerce"
        )

        # Create the bar plot
        ax = sns.barplot(
            x="model_name", y="success_rate", data=method_df, palette="viridis"
        )

        plt.title(f"{method} Attack Success Rate by Model", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Success Rate (%)", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter())

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3)

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f"success_rates_{method}.png")
        plt.savefig(output_path, dpi=300)
        output_paths.append(output_path)
        print(f"{method} success rate plot saved to {output_path}")

    return output_paths


def plot_perturbations(df, output_dir):
    """Plot L2 and Linf perturbation metrics"""
    # Convert to numeric if they are strings
    df["avg_perturbation_l2"] = pd.to_numeric(
        df["avg_perturbation_l2"], errors="coerce"
    )
    df["avg_perturbation_linf"] = pd.to_numeric(
        df["avg_perturbation_linf"], errors="coerce"
    )

    # L2 norm plot
    plt.figure(figsize=(14, 10))
    ax1 = sns.barplot(
        x="model_name", y="avg_perturbation_l2", hue="attack_method", data=df
    )
    plt.title("Average L2 Perturbation by Model and Attack Method", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("L2 Norm", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.4f", padding=3)

    plt.legend(title="Attack Method")
    plt.tight_layout()

    # Save L2 plot
    l2_output_path = os.path.join(output_dir, "l2_perturbations.png")
    plt.savefig(l2_output_path, dpi=300)

    # Linf norm plot
    plt.figure(figsize=(14, 10))
    ax2 = sns.barplot(
        x="model_name", y="avg_perturbation_linf", hue="attack_method", data=df
    )
    plt.title("Average L∞ Perturbation by Model and Attack Method", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("L∞ Norm", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.4f", padding=3)

    plt.legend(title="Attack Method")
    plt.tight_layout()

    # Save Linf plot
    linf_output_path = os.path.join(output_dir, "linf_perturbations.png")
    plt.savefig(linf_output_path, dpi=300)

    print(f"Perturbation plots saved to {l2_output_path} and {linf_output_path}")
    return l2_output_path, linf_output_path


def plot_method_specific_perturbations(df, output_dir):
    """Plot perturbation metrics for each attack method separately"""
    # Get unique attack methods
    attack_methods = df["attack_method"].unique()
    l2_output_paths = []
    linf_output_paths = []

    for method in attack_methods:
        # Filter data for this method
        method_df = df[df["attack_method"] == method]

        # Convert to numeric
        method_df["avg_perturbation_l2"] = pd.to_numeric(
            method_df["avg_perturbation_l2"], errors="coerce"
        )
        method_df["avg_perturbation_linf"] = pd.to_numeric(
            method_df["avg_perturbation_linf"], errors="coerce"
        )

        # L2 norm plot
        plt.figure(figsize=(12, 8))
        ax1 = sns.barplot(
            x="model_name", y="avg_perturbation_l2", data=method_df, palette="cool"
        )
        plt.title(f"{method} - Average L2 Perturbation by Model", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("L2 Norm", fontsize=14)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.4f", padding=3)

        plt.tight_layout()

        # Save L2 plot
        l2_path = os.path.join(output_dir, f"l2_perturbations_{method}.png")
        plt.savefig(l2_path, dpi=300)
        l2_output_paths.append(l2_path)

        # Linf norm plot
        plt.figure(figsize=(12, 8))
        ax2 = sns.barplot(
            x="model_name", y="avg_perturbation_linf", data=method_df, palette="cool"
        )
        plt.title(f"{method} - Average L∞ Perturbation by Model", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("L∞ Norm", fontsize=14)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt="%.4f", padding=3)

        plt.tight_layout()

        # Save Linf plot
        linf_path = os.path.join(output_dir, f"linf_perturbations_{method}.png")
        plt.savefig(linf_path, dpi=300)
        linf_output_paths.append(linf_path)

        print(f"{method} perturbation plots saved to {l2_path} and {linf_path}")

    return l2_output_paths, linf_output_paths


def plot_execution_time(df, output_dir):
    """Plot execution time for different attacks and models"""
    plt.figure(figsize=(14, 10))

    # Convert to numeric if they are strings
    df["execution_time"] = pd.to_numeric(df["execution_time"], errors="coerce")

    # Create the grouped bar plot
    ax = sns.barplot(x="model_name", y="execution_time", hue="attack_method", data=df)

    plt.title("Execution Time by Model and Attack Method", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)

    plt.legend(title="Attack Method")
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "execution_times.png")
    plt.savefig(output_path, dpi=300)
    print(f"Execution time plot saved to {output_path}")
    return output_path


def plot_method_specific_execution_time(df, output_dir):
    """Plot execution time for each attack method separately"""
    # Get unique attack methods
    attack_methods = df["attack_method"].unique()
    output_paths = []

    for method in attack_methods:
        # Filter data for this method
        method_df = df[df["attack_method"] == method]

        # Convert to numeric
        method_df["execution_time"] = pd.to_numeric(
            method_df["execution_time"], errors="coerce"
        )

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x="model_name", y="execution_time", data=method_df, palette="mako"
        )

        plt.title(f"{method} - Execution Time by Model", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Time (seconds)", fontsize=14)
        plt.xticks(rotation=45, ha="right")

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3)

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f"execution_times_{method}.png")
        plt.savefig(output_path, dpi=300)
        output_paths.append(output_path)
        print(f"{method} execution time plot saved to {output_path}")

    return output_paths


def generate_summary_table(df, output_dir):
    """Generate a summary table of all results"""
    # Group by attack method and model name
    summary = (
        df.groupby(["attack_method", "model_name"])
        .agg(
            {
                "success_rate": "mean",
                "avg_perturbation_l2": "mean",
                "avg_perturbation_linf": "mean",
                "execution_time": "mean",
                "total_images": "sum",
                "success_count": "sum",
            }
        )
        .reset_index()
    )

    # Convert success_rate to percentage
    summary["success_rate"] = pd.to_numeric(summary["success_rate"], errors="coerce")

    # Sort by success rate descending
    summary = summary.sort_values("success_rate", ascending=False)

    # Format the table
    formatted_summary = summary.copy()
    formatted_summary["success_rate"] = formatted_summary["success_rate"].map(
        "{:.2f}%".format
    )
    formatted_summary["avg_perturbation_l2"] = formatted_summary[
        "avg_perturbation_l2"
    ].map("{:.6f}".format)
    formatted_summary["avg_perturbation_linf"] = formatted_summary[
        "avg_perturbation_linf"
    ].map("{:.6f}".format)
    formatted_summary["execution_time"] = formatted_summary["execution_time"].map(
        "{:.2f} s".format
    )

    # Save to CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    formatted_summary.to_csv(csv_path, index=False)

    # Also create a LaTeX table for research papers
    latex_path = os.path.join(output_dir, "summary_table.tex")
    try:
        with open(latex_path, "w") as f:
            f.write(
                formatted_summary.to_latex(
                    index=False,
                    float_format="%.2f",
                    caption="Summary of Attack Results",
                )
            )
        print(f"Summary tables saved to {csv_path} and {latex_path}")
    except Exception as e:
        print(f"Error generating LaTeX table: {e}")
        print(f"Summary table saved to {csv_path}")

    # Generate method-specific summary tables
    method_tables = {}
    for method in df["attack_method"].unique():
        method_df = summary[summary["attack_method"] == method]
        method_csv = os.path.join(output_dir, f"summary_table_{method}.csv")
        method_df.to_csv(method_csv, index=False)
        method_tables[method] = method_csv
        print(f"{method} summary table saved to {method_csv}")

    return csv_path, latex_path, method_tables


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all results
    df = collect_all_results(args.log_dir)

    if df is None or len(df) == 0:
        print("No data found. Make sure you've run attacks that generate logs.")
        return

    # Generate combined plots
    if args.plot_type in ["success_rate", "all"]:
        plot_success_rates(df, args.output_dir)

    if args.plot_type in ["perturbation", "all"]:
        plot_perturbations(df, args.output_dir)

    if args.plot_type in ["time", "all"]:
        plot_execution_time(df, args.output_dir)

    # Generate method-specific plots if requested or if "all" is specified
    if args.separate_methods or args.plot_type == "all":
        if args.plot_type in ["success_rate", "all"]:
            plot_method_specific_success_rates(df, args.output_dir)

        if args.plot_type in ["perturbation", "all"]:
            plot_method_specific_perturbations(df, args.output_dir)

        if args.plot_type in ["time", "all"]:
            plot_method_specific_execution_time(df, args.output_dir)

    # Always generate the summary table
    generate_summary_table(df, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
