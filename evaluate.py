import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AdvDataset
from model import get_model
from utils import BASE_ADV_PATH, AverageMeter, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the results logger if available, otherwise create a simple CSV logger
try:
    from results_logger import ResultsLogger
except ImportError:
    print("ResultsLogger not found, creating a simple CSV logger.")

    class ResultsLogger:
        def __init__(self, log_dir="evaluation_logs"):
            self.log_dir = log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.evaluation_log = os.path.join(log_dir, "transfer_evaluation.csv")

            # Create file with headers if it doesn't exist
            if not os.path.exists(self.evaluation_log):
                with open(self.evaluation_log, "w") as f:
                    f.write(
                        "source_model,target_model,attack_method,success_rate,top1_accuracy,top5_accuracy,total_images,timestamp\n"
                    )

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
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.evaluation_log, "a") as f:
                f.write(
                    f"{source_model},{target_model},{attack_method},{success_rate},{top1},{top5},{total_images},{timestamp}\n"
                )
            print(f"Evaluation results logged to {self.evaluation_log}")


def arg_parse():
    parser = argparse.ArgumentParser(description="Evaluate adversarial examples")
    parser.add_argument(
        "--adv_path", type=str, default="", help="the path of adversarial examples."
    )
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        metavar="N",
        help="input batch size for reference (default: 16)",
    )
    parser.add_argument(
        "--model_name", type=str, default="", help="Model to evaluate against"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="evaluation_logs",
        help="Directory to save evaluation logs",
    )
    args = parser.parse_args()
    args.adv_path = os.path.join(BASE_ADV_PATH, args.adv_path)
    return args


def extract_source_info(adv_path):
    """Extract source model and attack method from the adversarial path"""
    path = Path(adv_path)
    dirname = path.name

    # Expected format: model_{source_model}-method_{attack_method}
    parts = dirname.split("-")

    source_model = "unknown"
    attack_method = "unknown"

    if len(parts) >= 1 and parts[0].startswith("model_"):
        source_model = parts[0].replace("model_", "")

    if len(parts) >= 2 and parts[1].startswith("method_"):
        attack_method = parts[1].replace("method_", "")

    return source_model, attack_method


if __name__ == "__main__":
    args = arg_parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Initialize logger
    logger = ResultsLogger(log_dir=args.log_dir)

    # Extract source model and attack method from path
    source_model, attack_method = extract_source_info(args.adv_path)

    print(
        f"Evaluating adversarial examples from {source_model} using {attack_method} attack"
    )
    print(f"Target model for evaluation: {args.model_name}")

    # Start timing
    total_start_time = time.time()

    # Loading dataset
    dataset = AdvDataset(args.model_name, args.adv_path)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    total_images = len(dataset)
    print(f"Total images: {total_images}")

    # Loading model
    model = get_model(args.model_name)
    model.to(device)
    model.eval()

    # main
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    prediction = []
    gts = []
    with torch.no_grad():
        end = time.time()
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Evaluating")):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader)}")

            batch_x = batch_data[0].to(device)
            batch_y = batch_data[1].to(device)
            batch_name = batch_data[3] if len(batch_data) > 3 else batch_data[2]

            output = model(batch_x)
            acc1, acc5 = accuracy(output.detach(), batch_y, topk=(1, 5))
            top1.update(acc1.item(), batch_x.size(0))
            top5.update(acc5.item(), batch_x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = output.detach().topk(1, 1, True, True)
            pred = pred.t()
            prediction.extend(list(torch.squeeze(pred.cpu()).numpy()))
            gts.extend(list(batch_y.cpu().numpy()))

    # Calculate success rate and misclassification
    success_count = 0
    df = pd.DataFrame()
    df["path"] = dataset.paths[: len(prediction)]
    df["pre"] = prediction
    df["gt"] = gts

    for i in range(len(df["pre"])):
        if df["pre"][i] != df["gt"][i]:
            success_count += 1

    success_rate = success_count / total_images * 100

    # Total evaluation time
    total_time = time.time() - total_start_time

    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Source Model: {source_model}")
    print(f"Attack Method: {attack_method}")
    print(f"Target Model: {args.model_name}")
    print(f"Total Images: {total_images}")
    print(f"Attack Success Rate: {success_rate:.2f}%")
    print(f"Top-1 Accuracy: {top1.avg:.2f}%")
    print(f"Top-5 Accuracy: {top5.avg:.2f}%")
    print(f"Total Evaluation Time: {total_time:.2f} seconds")

    # Log the evaluation results
    logger.log_evaluation(
        source_model=source_model,
        target_model=args.model_name,
        attack_method=attack_method,
        success_rate=success_rate,
        top1=top1.avg,
        top5=top5.avg,
        total_images=total_images,
    )

    # Save detailed results
    results_dir = os.path.join(args.log_dir, "detailed_results")
    os.makedirs(results_dir, exist_ok=True)

    detailed_file = os.path.join(
        results_dir, f"{source_model}_{attack_method}_to_{args.model_name}.csv"
    )
    df.to_csv(detailed_file, index=False)
    print(f"Detailed prediction results saved to {detailed_file}")

    # Generate a simple visualized confusion matrix
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        # Create a simple binary confusion matrix: correct vs. incorrect classification
        y_true = [1 if gt == pred else 0 for gt, pred in zip(df["gt"], df["pre"])]
        y_pred = [0] * len(y_true)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # <-- Specify labels

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        labels = [
            "Misclassified\n(Attack Success)",
            "Correctly Classified\n(Attack Failed)",
        ]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".0f")
        plt.title(
            f"Attack Success: {source_model} → {args.model_name}\nvia {attack_method}"
        )

        # Save the plot
        cm_file = os.path.join(
            results_dir, f"{source_model}_{attack_method}_to_{args.model_name}_cm.png"
        )
        plt.tight_layout()
        plt.savefig(cm_file, dpi=300)
        print(f"Confusion matrix saved to {cm_file}")
    except ImportError:
        print(
            "Matplotlib or scikit-learn not found, skipping confusion matrix visualization"
        )
