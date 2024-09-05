import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from typing import List, Dict, Any
from lmexp.utils.steering_settings import SteeringSettings
from lmexp.utils.behaviors import ANALYSIS_PATH, HUMAN_NAMES, get_results_dir, get_analysis_dir, ALL_BEHAVIORS
from lmexp.utils.helpers import set_plotting_settings

set_plotting_settings()

print("Script started. Imported necessary modules.")

def get_data(layer: int, multiplier: float, settings: SteeringSettings) -> Dict[str, Any]:
    print(f"Getting data for layer {layer}, multiplier {multiplier}, behavior {settings.behavior}")
    directory = get_results_dir(settings.behavior)
    if settings.type == "open_ended":
        directory = os.path.join(directory, "open_ended_scores")
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for filter {settings}", filenames)
    if len(filenames) == 0:
        print(f"[WARN] no filenames found for filter {settings}")
        return {}
    with open(filenames[0], "r") as f:
        return json.load(f)

def get_avg_key_prob(results: Dict[str, Any], key: str) -> float:
    print(f"Calculating average key probability for key: {key}")
    if not results:
        print("[WARN] Empty results, returning 0.0")
        return 0.0
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        if denom == 0:
            print(f"[WARN] Denominator is zero for result: {result}")
            continue
        if "A" in matching_value:
            match_key_prob_sum += result["a_prob"] / denom
        elif "B" in matching_value:
            match_key_prob_sum += result["b_prob"] / denom
    avg_prob = match_key_prob_sum / len(results) if results else 0.0
    print(f"Average probability: {avg_prob}")
    return avg_prob

def plot_effect_on_behaviors(
    layer: int, multipliers: List[float], behaviors: List[str], settings: SteeringSettings, title: str = None   
):
    print(f"Plotting effect on behaviors for layer {layer}, multipliers {multipliers}, behaviors {behaviors}")
    plt.clf()
    plt.figure(figsize=(10, 6))  # Increased figure size
    save_to = os.path.join(
        ANALYSIS_PATH,
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    print(f"Plot will be saved to: {save_to}")
    all_results = []
    for behavior in behaviors:
        print(f"Processing behavior: {behavior}")
        results = []
        for mult in multipliers:
            print(f"  Processing multiplier: {mult}")
            settings.behavior = behavior
            data = get_data(layer, mult, settings)
            print(f"  Retrieved data for layer {layer}, multiplier {mult}: {data}")
            if settings.type == "open_ended":
                avg_score = get_avg_score(data)
                results.append(avg_score)
            elif settings.type == "ab":
                avg_key_prob = get_avg_key_prob(data, "answer_matching_behavior")
                results.append(avg_key_prob * 100)
            else:
                avg_key_prob = get_avg_key_prob(data, "correct")
                results.append(avg_key_prob * 100)
            print(f"  Calculated result: {results[-1]}")
        all_results.append(results)
    
    print("Plotting results...")
    for idx, behavior in enumerate(behaviors):
        plt.plot(
            multipliers,
            all_results[idx],
            marker="o",
            linestyle="solid",
            markersize=10,
            linewidth=3,
            label=HUMAN_NAMES[behavior],
        )
    plt.xticks(ticks=multipliers, labels=multipliers)
    if title is not None:
        plt.title(title)
    plt.xlabel("Steering vector multiplier")
    ylabel = "p(answer matching behavior) (%)"
    if settings.type == "open_ended":
        ylabel = "Mean behavioral score (/10)"
    elif settings.type == "mmlu":
        ylabel = "p(correct answer to A/B question)"
    elif settings.type == "truthful_qa":
        ylabel = "p(correct answer to A/B question)"
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)  # Added grid lines
    plt.tight_layout()
    plt.savefig(save_to, format="png")
    plt.savefig(save_to.replace("png", "svg"), format="svg")
    print(f"Plots saved as PNG and SVG")
    
    print("Saving data to text file...")
    with open(save_to.replace(".png", ".txt"), "w") as f:
        for mult in multipliers:
            f.write(f"{mult}\t")
            for idx, behavior in enumerate(behaviors):
                f.write(f"{all_results[idx][multipliers.index(mult)]}\t")
            f.write("\n")
    print("Data saved to text file")

def plot_layer_sweeps(
    layers: List[int], behaviors: List[str], settings: SteeringSettings, title: str = None
):
    print(f"Plotting layer sweeps for layers {layers}, behaviors {behaviors}")
    plt.clf()
    plt.figure(figsize=(10, 6))  # Increased figure size
    all_results = []
    save_to = os.path.join(
        ANALYSIS_PATH,
        f"LAYER_SWEEPS_{settings.make_result_save_suffix()}.png",
    )
    print(f"Layer sweep plot will be saved to: {save_to}")
    for behavior in behaviors:
        if "coordinate" in behavior:
            print(f"Skipping coordinate behavior: {behavior}")
            continue
        print(f"Processing behavior: {behavior}")
        settings.behavior = behavior
        pos_per_layer = []
        neg_per_layer = []
        for layer in sorted(layers):
            print(f"  Processing layer: {layer}")
            base_res = get_avg_key_prob(get_data(layer, 0, settings), "answer_matching_behavior")
            pos_res = get_avg_key_prob(get_data(layer, 1, settings), "answer_matching_behavior") - base_res
            neg_res = get_avg_key_prob(get_data(layer, -1, settings), "answer_matching_behavior") - base_res
            pos_per_layer.append(pos_res)
            neg_per_layer.append(neg_res)
            print(f"  Results - Base: {base_res}, Positive: {pos_res}, Negative: {neg_res}")
        all_results.append((pos_per_layer, neg_per_layer))
        plt.plot(
            sorted(layers),
            pos_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#377eb8",
        )
        plt.plot(
            sorted(layers),
            neg_per_layer,
            linestyle="solid",
            linewidth=2,
            color="#ff7f00",
        )

    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#377eb8",
        label="Positive steering",
    )
    plt.plot(
        [],
        [],
        linestyle="solid",
        linewidth=2,
        color="#ff7f00",
        label="Negative steering",
    )

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))
    plt.xlabel("Layer")
    plt.ylabel("$\Delta$ p(answer matching behavior)")
    if not title:
        plt.title(f"Per-layer CAA effect: {settings.model_name}")  # Use settings.model_name here
    else:
        plt.title(title)
    plt.xticks(ticks=sorted(layers)[::5], labels=sorted(layers)[::5])
    plt.legend()
    plt.grid(True)  # Added grid lines
    plt.tight_layout()
    plt.ylim([-0.01, 0.01])  # Adjust y-axis limits to show small differences
    plt.savefig(save_to, format="png")
    print(f"Layer sweep plot saved")

def steering_settings_from_args(args, behavior: str) -> SteeringSettings:
    print(f"Creating SteeringSettings for behavior: {behavior}")
    steering_settings = SteeringSettings()
    steering_settings.type = args.type
    steering_settings.behavior = behavior
    steering_settings.override_vector = args.override_vector
    steering_settings.override_vector_model = args.override_vector_model
    steering_settings.use_base_model = args.use_base_model
    steering_settings.model_size = args.model_size
    if len(args.override_weights) > 0:
        steering_settings.override_model_weights_path = args.override_weights[0]
    print(f"SteeringSettings created: {steering_settings}")
    return steering_settings

if __name__ == "__main__":
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--title", type=str, required=False, default=None)
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--type",
        type=str,
        default="ab",
        choices=["ab", "open_ended", "truthful_qa", "mmlu"],
    )
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--override_weights", type=str, nargs="+", default=[])
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")

    steering_settings = steering_settings_from_args(args, args.behaviors[0])

    if len(args.override_weights) > 0:
        print("Override weights provided. Plotting finetuning openended comparison...")
        plot_finetuning_openended_comparison(steering_settings, args.override_weights[0], args.override_weights[1], args.multipliers, args.layers[0])
        print("Finetuning openended comparison plotted. Exiting.")
        exit(0)

    if steering_settings.type == "ab":
        print("Plotting layer sweeps...")
        plot_layer_sweeps(args.layers, args.behaviors, steering_settings, args.title)
        print("Layer sweeps plotted.")

    if len(args.layers) == 1 and steering_settings.type != "truthful_qa":
        print("Plotting effect on behaviors...")
        plot_effect_on_behaviors(args.layers[0], args.multipliers, args.behaviors, steering_settings, args.title)
        print("Effect on behaviors plotted.")

    for behavior in args.behaviors:
        print(f"Processing behavior: {behavior}")
        steering_settings = steering_settings_from_args(args, behavior)
        if steering_settings.type == "ab":
            if len(args.layers) > 1 and 1 in args.multipliers and -1 in args.multipliers:
                print("Plotting AB data per layer...")
                plot_ab_data_per_layer(
                    args.layers, [1, -1], steering_settings
                )
                print("AB data per layer plotted.")
            if len(args.layers) == 1:
                print("Plotting AB results for layer...")
                plot_ab_results_for_layer(args.layers[0], args.multipliers, steering_settings)
                print("AB results for layer plotted.")
        elif steering_settings.type == "open_ended":
            for layer in args.layers:
                print(f"Plotting open-ended results for layer {layer}...")
                plot_open_ended_results(layer, args.multipliers, steering_settings)
                print(f"Open-ended results for layer {layer} plotted.")
        elif steering_settings.type == "truthful_qa" or steering_settings.type == "mmlu":
            for layer in args.layers:
                print(f"Plotting TQA/MMLU results for layer {layer}...")
                plot_tqa_mmlu_results_for_layer(layer, args.multipliers, steering_settings)
                print(f"TQA/MMLU results for layer {layer} plotted.")

    print("Script execution completed.")