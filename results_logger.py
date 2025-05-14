import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


class ResultsLogger:
    def __init__(self, log_dir="results_logs"):
        """Initialize the results logger with a directory for saving logs."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log paths
        self.summary_path = os.path.join(log_dir, f"summary_{self.timestamp}.csv")
        self.details_path = os.path.join(log_dir, f"details_{self.timestamp}")
        os.makedirs(self.details_path, exist_ok=True)

        # Initialize summary CSV with headers
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "attack_method",
                    "model_name",
                    "epsilon",
                    "steps",
                    "total_images",
                    "success_count",
                    "success_rate",
                    "avg_perturbation_l2",
                    "avg_perturbation_linf",
                    "execution_time",
                    "timestamp",
                ]
            )

    def log_batch_results(
        self,
        attack_method,
        model_name,
        batch_idx,
        images,
        labels,
        predictions,
        adv_images,
        adv_predictions,
        perturbations,
        execution_time,
    ):
        """Log detailed results for a batch"""
        # Save batch results as JSON
        batch_results = {
            "attack_method": attack_method,
            "model_name": model_name,
            "batch_idx": batch_idx,
            "execution_time": execution_time,
            "images": [],
        }

        # For each image in the batch
        for i in range(len(images)):
            orig_pred = (
                predictions[i].item()
                if hasattr(predictions[i], "item")
                else int(predictions[i])
            )
            adv_pred = (
                adv_predictions[i].item()
                if hasattr(adv_predictions[i], "item")
                else int(adv_predictions[i])
            )
            true_label = (
                labels[i].item() if hasattr(labels[i], "item") else int(labels[i])
            )

            l2_norm = float(
                np.linalg.norm(perturbations[i].cpu().numpy().flatten(), ord=2)
            )
            linf_norm = float(np.max(np.abs(perturbations[i].cpu().numpy())))

            is_success = orig_pred == true_label and adv_pred != true_label

            batch_results["images"].append(
                {
                    "index": i,
                    "true_label": true_label,
                    "original_prediction": orig_pred,
                    "adversarial_prediction": adv_pred,
                    "success": is_success,
                    "l2_norm": l2_norm,
                    "linf_norm": linf_norm,
                }
            )

        # Save the batch results
        batch_file = os.path.join(
            self.details_path, f"{attack_method}_{model_name}_batch_{batch_idx}.json"
        )
        with open(batch_file, "w") as f:
            json.dump(batch_results, f, indent=2)

        return batch_results

    def log_summary_results(
        self,
        attack_method,
        model_name,
        epsilon,
        steps,
        total_images,
        success_count,
        avg_l2,
        avg_linf,
        execution_time,
    ):
        """Log summary results to the CSV file"""
        success_rate = success_count / total_images if total_images > 0 else 0

        with open(self.summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    attack_method,
                    model_name,
                    epsilon,
                    steps,
                    total_images,
                    success_count,
                    f"{success_rate:.4f}",
                    f"{avg_l2:.6f}",
                    f"{avg_linf:.6f}",
                    f"{execution_time:.2f}",
                    self.timestamp,
                ]
            )

    def get_summary_dataframe(self):
        """Return the summary results as a pandas DataFrame"""
        if os.path.exists(self.summary_path):
            return pd.read_csv(self.summary_path)
        return pd.DataFrame()

    def export_plots(self, output_dir="plots"):
        """Generate and save plots based on the collected data"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Load summary data
            df = self.get_summary_dataframe()

            if len(df) > 0:
                # Plot success rates by attack method and model
                plt.figure(figsize=(12, 8))
                sns.barplot(
                    x="model_name", y="success_rate", hue="attack_method", data=df
                )
                plt.title("Attack Success Rate by Model and Method")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"success_rates_{self.timestamp}.png")
                )

                # Plot average L2 perturbation
                plt.figure(figsize=(12, 8))
                sns.barplot(
                    x="model_name",
                    y="avg_perturbation_l2",
                    hue="attack_method",
                    data=df,
                )
                plt.title("Average L2 Perturbation by Model and Method")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"l2_perturbation_{self.timestamp}.png")
                )

                # Plot execution time
                plt.figure(figsize=(12, 8))
                sns.barplot(
                    x="model_name", y="execution_time", hue="attack_method", data=df
                )
                plt.title("Execution Time by Model and Method")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"execution_time_{self.timestamp}.png")
                )

            return True
        except Exception as e:
            print(f"Error generating plots: {e}")
            return False

    def log_evaluation(
        self,
        source_model,
        target_model,
        attack_method,
        success_rate,
        top1,
        top5,
        total_images,
    ):
        """Log evaluation results in a simple CSV format for compatibility with evaluate.py"""
        eval_log_path = os.path.join(self.log_dir, "transfer_evaluation.csv")
        file_exists = os.path.exists(eval_log_path)
        with open(eval_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "source_model",
                        "target_model",
                        "attack_method",
                        "success_rate",
                        "top1_accuracy",
                        "top5_accuracy",
                        "total_images",
                        "timestamp",
                    ]
                )
            writer.writerow(
                [
                    source_model,
                    target_model,
                    attack_method,
                    f"{success_rate:.2f}",
                    f"{top1:.2f}",
                    f"{top5:.2f}",
                    total_images,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
        print(f"Evaluation results logged to {eval_log_path}")
