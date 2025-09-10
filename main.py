#!/usr/bin/env python3
import argparse
import numpy as np
import io
import plotext as plt
from algorithms.mahalanobis import run_scenario  
from algorithms.mahalanobis_for_image import run_scenario  as run_scenario_image
from algorithms.mahalanobis_thresh import compute_thresholds as get_thresholds_m
from algorithms.knn import run_scenario_knn
from algorithms.knn_thresh import compute_knn_thresholds_raw
from algorithms.tsne import run_scenario_tsne
from algorithms.tsne_thresh import compute_tsne_perplexity_thresholds
from algorithms.kde import run_scenario_kde
from algorithms.kde_thresh import compute_kde_thresholds_txt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OOD detection / trajectory prediction with configurable parameters."
    )
    parser.add_argument("--decision-value", "-c", type=float, nargs="+", default=0.4,
                        help="Numeric decision value (interpreted according to --criterion-mode)")
    parser.add_argument("--criterion-mode", choices=["contamination", "threshold"],
                        default="contamination",
                        help="Interpret --decision-value as contamination level or as a threshold")
    parser.add_argument("--algorithm", "-a",
                        choices=["mahalanobis", "knn", "kde", "tsne"],
                        default="mahalanobis",
                        help="Which OOD algorithm to use")
    parser.add_argument("--knn-neighbors", "-k", type=int, default=5,
                        help="Number of neighbors k for KNN")
    parser.add_argument("--history-length", "-H", type=int, default=8,
                        help="History length H")
    parser.add_argument("--prediction-length", "-N", type=int, default=12,
                        help="Prediction length N")
    parser.add_argument("--error-radius", "-R", type=float, nargs="+",
                        default=[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3],
                        help="Acceptable error radii R_i (12 values)")
    parser.add_argument("--confidence-levels", "-α", type=float, nargs="+", default=[0.4],
                        help="List of confidence levels α to evaluate")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step dt")
    parser.add_argument("--perplexity", "-P", type=float, default=30.0,
                        help="Target perplexity for t‑SNE / KDE")
    parser.add_argument("--bandwidth", "-b", type=float, default=1.0,
                        help="Bandwidth (σ) for the Gaussian KDE")
    parser.add_argument("--model", type=str, default="linear",
                        help="Prediction model name")
    parser.add_argument("--cp-type", choices=["adaptive", "split"],
                        default="adaptive",
                        help="Conformal predictor type")
    parser.add_argument("--n-pedestrians", type=int, default=280,
                        help="Number of pedestrians to simulate")
    parser.add_argument("--test-dirpath", type=str, default="lobby3/test",
                        help="Directory containing test files")
    parser.add_argument("--train-dirpath", type=str,
                        default="./algorithms/train_dataset/lobby3",
                        help="Directory containing training files")
    parser.add_argument("--t-begin", type=int, default=40,
                        help="Start timestep for evaluation")
    parser.add_argument("--t-end", type=int, default=200,
                        help="End timestep for evaluation")
    parser.add_argument("--mode", type=str, default="feature",
                        help="Mode for evaluation")
    return parser.parse_args()

def main(args):
    # print configuration 
    print("Configuration:")
    print(f"  algorithm:         {args.algorithm}")
    if args.algorithm == "knn":
        print(f"    knn neighbors:   {args.knn_neighbors}")
    print(f"  model:             {args.model}")
    print(f"  cp_type:           {args.cp_type}")
    print(f"  n_pedestrians:     {args.n_pedestrians}")
    print(f"  test_dirpath:      {args.test_dirpath}")
    print(f"  train_dirpath:     {args.train_dirpath}")
    print(f"  history length H:  {args.history_length}")
    print(f"  prediction length N:{args.prediction_length}")
    print(f"  error radii R:     {args.error_radius}")
    print(f"  confidence levels α: {np.ones(len(args.confidence_levels))-args.confidence_levels}")
    print(f"  dt:                {args.dt}")
    print(f"  decision-value β:  {args.decision_value}  ({args.criterion_mode})")
    print(f"  t_begin:           {args.t_begin}")
    print(f"  t_end:             {args.t_end}")
    print(f"  vector mode:       {args.mode}")
    print()

    # prepare container for results per confidence level (store means and stds)
    results = {
        α: {
            "x": [],
            "id_mean": [], "id_std": [],
            "ood_mean": [], "ood_std": []
        }
        for α in args.confidence_levels
    }

    # dispatch based on algorithm choice
    if args.algorithm == "mahalanobis":
        thresholds, mu, cov_inv = get_thresholds_m(
            train_dir=args.train_dirpath,
            contamination_list=args.decision_value,
            window_size=args.history_length + args.prediction_length,
            dt_factor=args.dt,
            mode=args.mode
        )
        contamination = (thresholds if args.criterion_mode == "contamination"
                         else {β: β for β in args.decision_value})
        for β in contamination:
            for α in args.confidence_levels:
                mc = 1.0 - α
                id_mean, id_std, ood_mean, ood_std = run_scenario(
                    model=args.model,
                    cp_type=args.cp_type,
                    miscoverage_level=mc,
                    n_pedestrians=args.n_pedestrians,
                    test_dirpath=args.test_dirpath,
                    map_size=[10, 5, -10, -15],
                    threshold=contamination[β],
                    mu=mu,
                    cov_inv=cov_inv,
                    history_length=args.history_length,
                    prediction_length=args.prediction_length,
                    t_begin=args.t_begin,
                    t_end=args.t_end,
                    r_star=args.error_radius,
                    mode=args.mode
                )
                """  _, _, _, _ = run_scenario_image(
                    model=args.model,
                    cp_type=args.cp_type,
                    miscoverage_level=mc,
                    n_pedestrians=args.n_pedestrians,
                    test_dirpath=args.test_dirpath,
                    map_size=[10, 5, -10, -15],
                    threshold=contamination[β],
                    mu=mu,
                    cov_inv=cov_inv,
                    history_length=args.history_length,
                    prediction_length=args.prediction_length,
                    t_begin=args.t_begin,
                    t_end=args.t_end,
                    r_star=args.error_radius,
                    mode=args.mode
                ) """
                results[α]["x"].append(β)
                results[α]["id_mean"].append(id_mean)
                results[α]["id_std"].append(id_std)
                results[α]["ood_mean"].append(ood_mean)
                results[α]["ood_std"].append(ood_std)

    elif args.algorithm == "knn":
        thresholds, neigh, k = compute_knn_thresholds_raw(
            train_dir=args.train_dirpath,
            contamination_list=args.decision_value,
            k=args.knn_neighbors,
            history_length=args.history_length,
            prediction_length=args.prediction_length,
            mode=args.mode
        )
        contamination = (thresholds if args.criterion_mode == "contamination"
                         else {β: β for β in args.decision_value})
        for β in contamination:
            for α in args.confidence_levels:
                mc = 1.0 - α
                id_mean, id_std, ood_mean, ood_std = run_scenario_knn(
                    model=args.model,
                    cp_type=args.cp_type,
                    miscoverage_level=mc,
                    n_pedestrians=args.n_pedestrians,
                    test_dirpath=args.test_dirpath,
                    map_size=[10, 5, -10, -15],
                    thresholds=contamination[β],
                    neigh=neigh,
                    k=k,
                    history_length=args.history_length,
                    prediction_length=args.prediction_length,
                    t_begin=args.t_begin,
                    t_end=args.t_end,
                    r_star=args.error_radius,
                    mode=args.mode
                )
 
                results[α]["x"].append(β)
                results[α]["id_mean"].append(id_mean)
                results[α]["id_std"].append(id_std)
                results[α]["ood_mean"].append(ood_mean)
                results[α]["ood_std"].append(ood_std)

    elif args.algorithm == "kde":
        thresholds, kde_model = compute_kde_thresholds_txt(
            train_dir=args.train_dirpath,
            contamination_list=args.decision_value,
            bandwidth=args.bandwidth,
            history_length=args.history_length,
            prediction_length=args.prediction_length,
            mode=args.mode
        )
        contamination = (thresholds if args.criterion_mode == "contamination"
                         else {β: β for β in args.decision_value})
        for β in contamination:
            for α in args.confidence_levels:
                mc = 1.0 - α
                id_mean, id_std, ood_mean, ood_std = run_scenario_kde(
                    model=args.model,
                    cp_type=args.cp_type,
                    miscoverage_level=mc,
                    n_pedestrians=args.n_pedestrians,
                    test_dirpath=args.test_dirpath,
                    map_size=[10, 5, -10, -15],
                    thresholds=thresholds,
                    kde=kde_model,
                    contamination=β,
                    history_length=args.history_length,
                    prediction_length=args.prediction_length,
                    r_star=args.error_radius,
                    t_begin=args.t_begin,
                    t_end=args.t_end,
                    mode=args.mode
                )

                results[α]["x"].append(β)
                results[α]["id_mean"].append(id_mean)
                results[α]["id_std"].append(id_std)
                results[α]["ood_mean"].append(ood_mean)
                results[α]["ood_std"].append(ood_std)

    elif args.algorithm == "tsne":
        thresholds, sigmas, X_train = compute_tsne_perplexity_thresholds(
            train_dir=args.train_dirpath,
            contamination_list=args.decision_value,
            perplexity=args.perplexity,
            history_length=args.history_length,
            prediction_length=args.prediction_length,
            dt_factor=args.dt,
            mode=args.mode
        )
        contamination = (thresholds if args.criterion_mode == "contamination"
                         else {β: β for β in args.decision_value})
        for β in contamination:
            for α in args.confidence_levels:
                mc = 1.0 - α
                id_mean, id_std, ood_mean, ood_std = run_scenario_tsne(
                    model=args.model,
                    cp_type=args.cp_type,
                    miscoverage_level=mc,
                    n_pedestrians=args.n_pedestrians,
                    test_dirpath=args.test_dirpath,
                    map_size=[10, 5, -10, -15],
                    thresholds=contamination[β],
                    X_train=X_train,
                    perplexity=args.perplexity,
                    history_length=args.history_length,
                    prediction_length=args.prediction_length,
                    r_star=args.error_radius,
                    t_begin=args.t_begin,
                    t_end=args.t_end,
                    mode=args.mode
                )
      
                results[α]["x"].append(β)
                results[α]["id_mean"].append(id_mean)
                results[α]["id_std"].append(id_std)
                results[α]["ood_mean"].append(ood_mean)
                results[α]["ood_std"].append(ood_std)

    else:
        print(f"Algorithm “{args.algorithm}” not yet implemented.")
        return

    # bar chart grouped by confidence level α
    all_vals = []
    for α, data in results.items():
        all_vals += data["id_mean"] + data["ood_mean"]
    y_min, y_max = min(all_vals), max(all_vals)
    margin      = (y_max - y_min) * 0.05   # 5% padding

    plt.clear_figure()
    plt.canvas_color('default')
    plt.axes_color('default')
    plt.ticks_color('grey')
    plt.grid(True)
    plt.ylim(y_min - margin, y_max + margin)
    plt.title(f"IID vs OOD Means ({args.algorithm})")
    plt.xlabel("confidence level α")
    plt.ylabel("mean error")

    alphas   = sorted(results.keys())
    betas    = results[alphas[0]]["x"]
    positions = list(range(len(alphas)))
    xticks    = [f"{1-α:.2f}" for α in alphas]
    if len(positions) > 1:
        gaps = [j - i for i, j in zip(positions, positions[1:])]
        spacing = min(gaps)
    else:
        spacing = 1.0   # arbitrary if only one α
    bars_per_group = 2 * len(betas)
    group_width    = 0.8*spacing
    bar_width      = group_width / bars_per_group
    palette = list(range(4, 256, 4))
    for i, α in enumerate(alphas):
        center      = positions[i]
        group_start = center - group_width/2

        for j, β in enumerate(betas):
            idx    = results[α]["x"].index(β)
            id_val = results[α]["id_mean"][idx]
            oo_val = results[α]["ood_mean"][idx]

            x_iid = group_start + 2*j*bar_width
            x_ood = x_iid + bar_width

            # both IID and OOD for this β use the same odd‑number colour
            color_code = palette[j]  

            label_i = f"IID contamination level β={β:.2f}" if i == 0 else None
            label_o = f"OOD contamination level β={β:.2f}" if i == 0 else None
            # TODO: modify code for raw thresholds cases too.
            plt.bar(
                [x_iid], [id_val],
                width=bar_width*0.9,
                label=label_i,
                color=color_code
            )
            plt.bar(
                [x_ood], [oo_val],
                width=bar_width*0.9,
                label=label_o,
                color=color_code+2
            )

    plt.xticks(positions, xticks)
    plt.show()

    # 2) now we print out the results in a readable format
    print()
    for α in alphas:
        for β, μ_i, σ_i, μ_o, σ_o in zip(
                results[α]["x"],
                results[α]["id_mean"], results[α]["id_std"],
                results[α]["ood_mean"], results[α]["ood_std"]
        ):
            print(
                f"α={1-α:.2f} contamination level={β:.2f} → "
                f"IID μ={μ_i:.4f} σ={σ_i:.4f} | "
                f"OOD μ={μ_o:.4f} σ={σ_o:.4f}"
            )
if __name__ == "__main__":
    main(parse_args())
