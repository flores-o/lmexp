import matplotlib.pyplot as plt
import json
from typing import Dict, Any, List
import os
from collections import defaultdict
import matplotlib.cm as cm
import numpy as np
from lmexp.utils.steering_settings import SteeringSettings
from lmexp.utils.behaviors import ANALYSIS_PATH, HUMAN_NAMES, get_results_dir, get_analysis_dir, ALL_BEHAVIORS
from lmexp.utils.helpers import set_plotting_settings

set_plotting_settings()

def get_data(
    layer: int,
    multiplier: float,
    settings: SteeringSettings,
) -> Dict[str, Any]:
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
        return []
    with open(filenames[0], "r") as f:
        return json.load(f)

def get_avg_score(results: Dict[str, Any]) -> float:
    score_sum = 0.0
    tot = 0
    for result in results:
        try:
            score_sum += float(result["score"])
            tot += 1
        except:
            print(f"[WARN] Skipping invalid score: {result}")
    if tot == 0:
        print(f"[WARN] No valid scores found in results")
        return 0.0
    return score_sum / tot

def get_avg_key_prob(results: Dict[str, Any], key: str) -> float:
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        if "A" in matching_value:
            match_key_prob_sum += result["a_prob"] / denom
        elif "B" in matching_value:
            match_key_prob_sum += result["b_prob"] / denom
    return match_key_prob_sum / len(results)

def plot_ab_results_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    system_prompt_options = [
        ("pos", f"Positive system prompt"),
        ("neg", f"Negative system prompt"),
        (None, f"No system prompt"),
    ]
    settings.system_prompt = None
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    plt.clf()
    plt.figure(figsize=(3.5, 3.5))
    all_results = {}
    for system_prompt, label in system_prompt_options:
        settings.system_prompt = system_prompt
        try:
            res_list = []
            for multiplier in multipliers:
                results = get_data(layer, multiplier, settings)
                avg_key_prob = get_avg_key_prob(results, "answer_matching_behavior")
                res_list.append((multiplier, avg_key_prob))
            res_list.sort(key=lambda x: x[0])
            plt.plot(
                [x[0] for x in res_list],
                [x[1] for x in res_list],
                label=label,
                marker="o",
                linestyle="solid",
                markersize=10,
                linewidth=3,
            )
            all_results[system_prompt] = res_list
        except:
            print(f"[WARN] Missing data for system_prompt={system_prompt} for layer={layer}")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Multiplier")
    plt.ylabel("p(answer matching behavior)")
    plt.xticks(ticks=multipliers, labels=multipliers)
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"{HUMAN_NAMES[settings.behavior]} - {settings.get_formatted_model_name()}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_to, format="png")
    # Save data in all_results used for plotting as .txt and .tex
    with open(save_to.replace(".png", ".txt"), "w") as f, open(save_to.replace(".png", ".tex"), "w") as f_tex:
        for system_prompt, res_list in all_results.items():
            for multiplier, score in res_list:
                f.write(f"{system_prompt}\t{multiplier}\t{score}\n")
        if len(all_results) == 3:
            try:
                none_results = dict(all_results[None])[-1], dict(all_results[None])[0], dict(all_results[None])[1]
                pos_results = dict(all_results["pos"])[-1], dict(all_results["pos"])[0], dict(all_results["pos"])[1]
                neg_results = dict(all_results["neg"])[-1], dict(all_results["neg"])[0], dict(all_results["neg"])[1]
                f_tex.write(f"{HUMAN_NAMES[settings.behavior]} & {none_results[0]:.2f} & {none_results[1]:.2f} & {none_results[2]:.2f} & {pos_results[0]:.2f} & {pos_results[1]:.2f} & {pos_results[2]:.2f} & {neg_results[0]:.2f} & {neg_results[1]:.2f} & {neg_results[2]:.2f}")
            except KeyError:
                pass

def plot_tqa_mmlu_results_for_layer(
    layer: int, multipliers: List[float], settings: SteeringSettings
):
    save_to = os.path.join(
        get_analysis_dir(settings.behavior),
        f"{settings.make_result_save_suffix(layer=layer)}.png",
    )
    res_per_category = defaultdict(list)
    for multiplier in multipliers:
        results = get_data(layer, multiplier, settings)
        categories = set([item["category"] for item in results])
        for category in categories:
            category_results = [
                item for item in results if item["category"] == category
            ]
            avg_key_prob = get_avg_key_prob(category_results, "correct")
            res_per_category[category].append((multiplier, avg_key_prob))

    plt.figure(figsize=(10, 5))
    for idx, (category, res_list) in enumerate(sorted(res_per_category.items(), key=lambda x: x[0])):
        x = [idx] * len(res_list)
        y = [score for _, score in res_list]
        colors = cm.rainbow(np.linspace(0, 1, len(res_list)))
        plt.scatter(x, y, color=colors, s=80)

    for idx, multiplier in enumerate(multipliers):
        plt.scatter([], [], color=cm.rainbow(np.linspace(0, 1, len(multipliers)))[idx], label=f"Multiplier {multiplier}")
    plt.legend(loc="upper left")

    plt.xticks(range(len(res_per_category)), sorted(res_per_category.keys()), rotation=45, ha="right")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.xlabel("Category")
    plt.ylabel("Probability of correct answer")
    if (settings.override_vector is None) and (settings.override_vector_model is None) and (settings.override_model_weights_path is None):
        plt.title(f"Effect of {HUMAN_NAMES[settings.behavior]} CAA on {settings.get_formatted_model_name()} performance")
    plt.tight_layout()
    plt.savefig(save_to, format="png")

    # Save data used for plotting
    with open(save_to.replace(".png", ".txt"), "w") as f, open(save_to.replace(".png", ".tex"), "w") as f_tex:
        for category in sorted(res_per_category.keys()):
            res_list = res_per_category[category]
            res_dict = dict(res_list)
            try:
                no_steering_res = res_dict[0]
                positive_steering_res = res_dict[1]
                negative_steering_res = res_dict[-1]
                f_tex.write(f"{category.capitalize()} & {positive_steering_res:.2f} & {negative_steering_res:.2f} & {no_steering_res:.2f} \\\ \n")
            except KeyError:
                pass
            for multiplier, score in res_list:
                f.write(f"{category}\t{multiplier}\t{score}\n")
