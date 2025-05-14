import argparse
import os
import pickle as pkl
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import methods
from dataset import AdvDataset
from results_logger import ResultsLogger
from utils import BASE_ADV_PATH, ROOT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arg_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--attack", type=str, default="", help="the name of specific attack method"
    )
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for reference (default: 16)",
    )
    parser.add_argument("--model_name", type=str, default="", help="")
    parser.add_argument("--filename_prefix", type=str, default="", help="")
    parser.add_argument(
        "--epsilon", type=float, default=16 / 255, help="Perturbation magnitude"
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of attack steps")
    parser.add_argument(
        "--log_dir", type=str, default="results_logs", help="Directory to save logs"
    )
    args = parser.parse_args()
    args.opt_path = os.path.join(
        BASE_ADV_PATH, f"model_{args.model_name}-method_{args.attack}"
    )
    if not os.path.exists(args.opt_path):
        os.makedirs(args.opt_path)
    return args


if __name__ == "__main__":
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Initialize results logger
    logger = ResultsLogger(log_dir=args.log_dir)

    # loading dataset
    dataset = AdvDataset(
        args.model_name,
        os.path.join(ROOT_PATH, "clean_resized_images"),
        load_percentage=0.1,
    )
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(args.attack, args.model_name)

    # Attack
    attack_method = getattr(methods, args.attack)(
        args.model_name, steps=args.steps, epsilon=args.epsilon
    )

    # For tracking statistics
    total_images = 0
    success_count = 0
    all_l2_norms = []
    all_linf_norms = []
    all_loss_info = {}

    # Record total execution time
    total_start_time = time.time()

    # Main
    for batch_idx, batch_data in enumerate(
        tqdm(data_loader, desc="Processing batches")
    ):
        if batch_idx % 100 == 0:
            print("Running batch_idx", batch_idx)

        batch_x = batch_data[0].to(device)
        batch_y = batch_data[1].to(device)
        batch_name = batch_data[3]

        # Get original predictions
        with torch.no_grad():
            orig_outputs = attack_method.model(batch_x)
            _, orig_predictions = torch.max(orig_outputs, 1)

        # Start batch timing
        batch_start_time = time.time()

        # Generate adversarial examples
        adv_inps, loss_info = attack_method(batch_x, batch_y)

        # End batch timing
        batch_execution_time = time.time() - batch_start_time

        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = attack_method.model(adv_inps)
            _, adv_predictions = torch.max(adv_outputs, 1)

        # Calculate perturbations
        perturbations = attack_method._return_perts(batch_x, adv_inps)

        # Track success rate and norms
        batch_success = (orig_predictions == batch_y) & (adv_predictions != batch_y)
        batch_success_count = batch_success.sum().item()
        success_count += batch_success_count
        total_images += len(batch_x)

        # Calculate norms
        for i in range(len(batch_x)):
            pert = perturbations[i].cpu().numpy()
            l2_norm = np.linalg.norm(pert.flatten(), ord=2)
            linf_norm = np.max(np.abs(pert))
            all_l2_norms.append(l2_norm)
            all_linf_norms.append(linf_norm)

        # Log batch results
        logger.log_batch_results(
            args.attack,
            args.model_name,
            batch_idx,
            batch_x.cpu(),
            batch_y.cpu(),
            orig_predictions.cpu(),
            adv_inps.cpu(),
            adv_predictions.cpu(),
            perturbations.cpu(),
            batch_execution_time,
        )

        # Save adversarial examples
        attack_method._save_images(adv_inps, batch_name, args.opt_path)

        if loss_info is not None:
            all_loss_info[batch_name] = loss_info

    # Record total time
    total_execution_time = time.time() - total_start_time

    # Calculate averages
    avg_l2 = np.mean(all_l2_norms) if all_l2_norms else 0
    avg_linf = np.mean(all_linf_norms) if all_linf_norms else 0
    success_rate = success_count / total_images if total_images > 0 else 0

    # Log summary results
    logger.log_summary_results(
        args.attack,
        args.model_name,
        args.epsilon,
        args.steps,
        total_images,
        success_count,
        avg_l2,
        avg_linf,
        total_execution_time,
    )

    # Print summary
    print("\nAttack Results Summary:")
    print(f"Model: {args.model_name}, Attack: {args.attack}")
    print(f"Total Images: {total_images}")
    print(f"Success Count: {success_count}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Avg L2 Perturbation: {avg_l2:.6f}")
    print(f"Avg L∞ Perturbation: {avg_linf:.6f}")
    print(f"Total Execution Time: {total_execution_time:.2f} seconds")

    # Generate plots
    try:
        logger.export_plots()
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")

    # Save loss info if available
    if loss_info is not None:
        with open(os.path.join(args.opt_path, "loss_info.json"), "wb") as opt:
            pkl.dump(all_loss_info, opt)
