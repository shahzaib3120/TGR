import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze adversarial transferability results"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="evaluation_logs",
        help="Directory containing evaluation logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="transferability_results",
        help="Directory to save analysis results",
    )
    return parser.parse_args()


def load_evaluation_data(log_dir):
    """Load evaluation data from the CSV log file"""
    csv_path = os.path.join(log_dir, "transfer_evaluation.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Evaluation log file {csv_path} not found.")
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return None


def plot_transferability_heatmap(df, output_dir):
    """Create a heatmap showing attack transferability between models"""
    # Ensure numeric values
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")

    # Create pivot table for the heatmap
    pivot_df = df.pivot_table(
        index="source_model",
        columns="target_model",
        values="success_rate",
        aggfunc="mean",
    )

    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="YlOrRd",
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Success Rate (%)"},
    )

    plt.title("Adversarial Example Transferability Between Models", fontsize=16)
    plt.xlabel("Target Model", fontsize=14)
    plt.ylabel("Source Model", fontsize=14)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "transferability_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Transferability heatmap saved to {output_path}")

    return output_path


def plot_attack_method_comparison(df, output_dir):
    """Compare different attack methods' transferability"""
    # Ensure numeric values
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")

    # Group by attack method and calculate mean success rate
    attack_df = (
        df.groupby(["attack_method", "source_model", "target_model"])["success_rate"]
        .mean()
        .reset_index()
    )

    # Plot barplot comparing attack methods
    plt.figure(figsize=(16, 10))
    sns.barplot(
        x="attack_method", y="success_rate", hue="source_model", data=attack_df
    )

    plt.title("Attack Method Transferability Comparison", fontsize=16)
    plt.xlabel("Attack Method", fontsize=14)
    plt.ylabel("Average Success Rate (%)", fontsize=14)
    plt.xticks(rotation=45, ha="right")

    plt.legend(title="Source Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "attack_method_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Attack method comparison saved to {output_path}")

    return output_path


def plot_per_attack_heatmaps(df, output_dir):
    """Create separate heatmaps for each attack method"""
    # Ensure numeric values
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")

    # Create a directory for attack-specific heatmaps
    attack_dir = os.path.join(output_dir, "per_attack_heatmaps")
    os.makedirs(attack_dir, exist_ok=True)

    # Get unique attack methods
    attack_methods = df["attack_method"].unique()

    output_paths = []

    # Generate a separate heatmap for each attack method
    for attack in attack_methods:
        # Filter data for this attack method
        attack_df = df[df["attack_method"] == attack]

        # Create pivot table for the heatmap
        pivot_df = attack_df.pivot_table(
            index="source_model",
            columns="target_model",
            values="success_rate",
            aggfunc="mean",
        )

        # Plot the heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            pivot_df,
            annot=True,
            cmap="YlOrRd",
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "Success Rate (%)"},
        )

        plt.title(f"Adversarial Example Transferability - {attack} Attack", fontsize=16)
        plt.xlabel("Target Model", fontsize=14)
        plt.ylabel("Source Model", fontsize=14)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(attack_dir, f"transferability_heatmap_{attack}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Heatmap for {attack} attack saved to {output_path}")
        output_paths.append(output_path)

    return output_paths


def generate_transferability_table(df, output_dir):
    """Generate a detailed table of transferability results"""
    # Ensure numeric values
    df["success_rate"] = pd.to_numeric(df["success_rate"], errors="coerce")
    df["top1_accuracy"] = pd.to_numeric(df["top1_accuracy"], errors="coerce")
    df["top5_accuracy"] = pd.to_numeric(df["top5_accuracy"], errors="coerce")

    # Create a pivot table for source->target transferability
    transfer_table = df.pivot_table(
        index=["source_model", "attack_method"],
        columns="target_model",
        values="success_rate",
        aggfunc="mean",
    ).reset_index()

    # Sort by source model and attack method
    transfer_table = transfer_table.sort_values(["source_model", "attack_method"])

    # Save to CSV
    csv_path = os.path.join(output_dir, "transferability_table.csv")
    transfer_table.to_csv(csv_path, index=False)

    # Also create a LaTeX table for research papers
    try:
        latex_path = os.path.join(output_dir, "transferability_table.tex")
        with open(latex_path, "w") as f:
            f.write(
                transfer_table.style.format(precision=1).to_latex(
                    caption="Transferability of Adversarial Examples Between Models (Success Rate %)",
                    label="tab:transferability",
                )
            )
        print(f"LaTeX table saved to {latex_path}")
    except:
        print(
            "Could not generate LaTeX table. Make sure pandas version supports style.to_latex()"
        )

    print(f"Transferability table saved to {csv_path}")
    return csv_path


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation data
    df = load_evaluation_data(args.log_dir)

    if df is None:
        print("No evaluation data found.")
        return

    print(f"Loaded {len(df)} evaluation records.")

    # Generate visualizations
    plot_transferability_heatmap(df, args.output_dir)  # Overall heatmap
    plot_per_attack_heatmaps(df, args.output_dir)  # Per-attack heatmaps
    plot_attack_method_comparison(df, args.output_dir)
    generate_transferability_table(df, args.output_dir)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
