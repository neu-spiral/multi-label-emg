"""
Using the model that was trained during the experiment,
compare performance during feedback and no-feedback blocks.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from multi_label_emg.data import load_data_dict
from multi_label_emg.train import plot_confusion_matrix
from multi_label_emg.utils import PROJECT_ROOT, canonical_coords, confusion_matrix
from scripts.run_experiment_1 import subjects


def get_stats(cm):
    single_acc = np.diag(cm)[:6].mean()
    double_acc = np.diag(cm)[6:].mean()
    overall_acc = np.diag(cm).mean()

    coords, coords_str = canonical_coords()
    dir_accs = []
    mod_accs = []
    exact_accs = []
    for i, (dir, mod) in enumerate(coords):
        row_i = cm[i]
        same_dir_cols = [j for j, (d, m) in enumerate(coords) if d == dir]
        same_mod_cols = [j for j, (d, m) in enumerate(coords) if m == mod]
        dir_acc = row_i[same_dir_cols].sum()
        mod_acc = row_i[same_mod_cols].sum()
        exact_acc = row_i[i]
        dir_accs.append(dir_acc)
        mod_accs.append(mod_acc)
        exact_accs.append(exact_acc)
    dir_acc = np.mean(dir_accs)
    mod_acc = np.mean(mod_accs)
    return {
        "single_acc": round(single_acc, 3),
        "double_acc": round(double_acc, 3),
        "overall_acc": round(overall_acc, 3),
        "dir_acc": round(dir_acc, 3),
        "mod_acc": round(mod_acc, 3),
        "exact_acc": round(exact_acc, 3),
    }


@dataclass
class Result:
    single_acc: float
    double_acc: float
    overall_acc: float
    dir_acc: float
    mod_acc: float
    exact_acc: float
    confusion_matrix: np.ndarray


def summarize_one_condition(results_list, output_dir: Path, name: str):
    single_acc_avg = np.nanmean([r.single_acc for r in results_list])
    single_acc_std = np.nanstd([r.single_acc for r in results_list])
    double_acc_avg = np.nanmean([r.double_acc for r in results_list])
    double_acc_std = np.nanstd([r.double_acc for r in results_list])
    overall_acc_avg = np.nanmean([r.overall_acc for r in results_list])
    overall_acc_std = np.nanstd([r.overall_acc for r in results_list])
    dir_acc_avg = np.nanmean([r.dir_acc for r in results_list])
    dir_acc_std = np.nanstd([r.dir_acc for r in results_list])
    mod_acc_avg = np.nanmean([r.mod_acc for r in results_list])
    mod_acc_std = np.nanstd([r.mod_acc for r in results_list])
    exact_acc_avg = np.nanmean([r.exact_acc for r in results_list])
    exact_acc_std = np.nanstd([r.exact_acc for r in results_list])

    cm_avg = np.nanmean([r.confusion_matrix for r in results_list], axis=0)

    print(f"Singles acc \t {single_acc_avg:.3f} +/- {single_acc_std:.3f}".expandtabs(20))
    print(f"Doubles acc \t {double_acc_avg:.3f} +/- {double_acc_std:.3f}".expandtabs(20))
    print(f"Overall acc \t {overall_acc_avg:.3f} +/- {overall_acc_std:.3f}".expandtabs(20))
    print(f"Dir acc \t {dir_acc_avg:.3f} +/- {dir_acc_std:.3f}".expandtabs(20))
    print(f"Mod acc \t {mod_acc_avg:.3f} +/- {mod_acc_std:.3f}".expandtabs(20))
    print(f"Exact acc \t {exact_acc_avg:.3f} +/- {exact_acc_std:.3f}".expandtabs(20))

    fig = plot_confusion_matrix(cm_avg)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(output_dir / f"confusion_matrix.{name}.png", width=1000, height=1000, scale=2)
    fig.write_html(output_dir / f"confusion_matrix.{name}.html", include_plotlyjs="cdn")


def make_boxplots(nofeedback_results, feedback_results, output_dir):
    fig = go.Figure()
    # 2 names (no feedback, feedback)
    # 3 x axis values (singles, doubles, overall)

    # Add a trace describing performance of "nofeeback" condition
    x_nofeedback = []
    y_nofeedback = []
    # add the singles acc
    y_nofeedback.extend([r.single_acc for r in nofeedback_results])
    x_nofeedback.extend(["Single Acc"] * len(nofeedback_results))
    # add the doubles acc
    y_nofeedback.extend([r.double_acc for r in nofeedback_results])
    x_nofeedback.extend(["Double Acc"] * len(nofeedback_results))
    # add the overall acc
    y_nofeedback.extend([r.overall_acc for r in nofeedback_results])
    x_nofeedback.extend(["Overall Acc"] * len(nofeedback_results))
    # add the trace
    fig.add_trace(
        go.Box(
            x=x_nofeedback,
            y=y_nofeedback,
            name="No Feedback",
            boxmean=True,
            pointpos=0,
            boxpoints="all",
            marker_color="#08529c",
        )
    )

    # Add a trace describing performance of "feeback" condition
    x_feedback = []
    y_feedback = []
    # add the singles acc
    y_feedback.extend([r.single_acc for r in feedback_results])
    x_feedback.extend(["Single Acc"] * len(feedback_results))
    # add the doubles acc
    y_feedback.extend([r.double_acc for r in feedback_results])
    x_feedback.extend(["Double Acc"] * len(feedback_results))
    # add the overall acc
    y_feedback.extend([r.overall_acc for r in feedback_results])
    x_feedback.extend(["Overall Acc"] * len(feedback_results))
    # add the trace
    fig.add_trace(
        go.Box(
            x=x_feedback,
            y=y_feedback,
            name="Feedback",
            boxmean=True,
            pointpos=0,
            boxpoints="all",
            marker_color="#FF851B",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        template="simple_white",
        yaxis_title="Accuracy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=20)),
        boxmode="group",
        boxgroupgap=0.05,
        boxgap=0.1,
        yaxis_range=[0, 1],
    )
    fig.write_image(output_dir / "boxplots.png", width=1000, height=1000, scale=2)
    fig.write_html(output_dir / "boxplots.html", include_plotlyjs="cdn")


def compare_gamepad_to_prompts(
    gamepad_dir_labels,
    gamepad_mod_labels,
    visual_dir_labels,
    visual_mod_labels,
    subj_name,
    figs_dir,
):
    y_true_2d = np.stack([visual_dir_labels.argmax(-1), visual_mod_labels.argmax(-1)], axis=-1)
    y_pred_2d = np.stack([gamepad_dir_labels.argmax(-1), gamepad_mod_labels.argmax(-1)], axis=-1)

    cm = confusion_matrix(y_true_2d, y_pred_2d)
    fig_cm = plot_confusion_matrix(cm)
    filename = f"GamePad-vs-VisualCues-{subj_name}"
    fig_cm.update_layout(title=filename)
    fig_cm.write_image(figs_dir / f"{filename}-cm.png", width=1000, height=1000, scale=2)
    fig_cm.write_html(figs_dir / f"{filename}-cm.html", include_plotlyjs="cdn")


def main(output_dir: Path):
    nofeedback_results = []
    feedback_results = []
    data_dict = load_data_dict()

    for subject in tqdm(subjects, desc="subjects"):
        data = data_dict[subject]

        # For calibration block - compare gamepad to visual cues
        compare_gamepad_to_prompts(
            data["Calibration_dir_labels"],  # gamepad
            data["Calibration_mod_labels"],  # gamepad
            data["Calibration_visual_dir_labels"],  # visual cue
            data["Calibration_visual_mod_labels"],  # visual cue
            subject,
            output_dir,
        )

        # Load the no-feedback part
        blocks = [
            "HoldPulse1_NoFeedBack",
            "SimultaneousPulse1_NoFeedBack",
            "HoldPulse2_NoFeedBack",
            "SimultaneousPulse2_NoFeedBack",
            "HoldPulse3_NoFeedBack",
            "SimultaneousPulse3_NoFeedBack",
        ]

        # Pool data from all blocks
        for block in blocks:
            # load block
            dir_probs = data[f"{block}_dir_probs"]
            mod_probs = data[f"{block}_mod_probs"]
            dir_labels = data[f"{block}_dir_labels"]
            mod_labels = data[f"{block}_mod_labels"]

            # convert to 2d predictions and make conf mat
            y_pred_2d = np.stack([dir_probs.argmax(-1), mod_probs.argmax(-1)], axis=1)
            y_true_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=1)
            conf_mat = confusion_matrix(y_true_2d, y_pred_2d)
            stats = get_stats(conf_mat)
            nofeedback_results.append(
                Result(
                    single_acc=stats["single_acc"],
                    double_acc=stats["double_acc"],
                    overall_acc=stats["overall_acc"],
                    dir_acc=stats["dir_acc"],
                    mod_acc=stats["mod_acc"],
                    exact_acc=stats["exact_acc"],
                    confusion_matrix=conf_mat,
                )
            )

        # Load the feedback part
        blocks = [
            "HoldPulse1_WithFeedBack",
            "SimultaneousPulse1_WithFeedBack",
            "HoldPulse2_WithFeedBack",
            "SimultaneousPulse2_WithFeedBack",
            "HoldPulse3_WithFeedBack",
            "SimultaneousPulse3_WithFeedBack",
        ]

        # Pool data from all blocks
        for block in blocks:
            # load block
            dir_probs = data[f"{block}_dir_probs"]
            mod_probs = data[f"{block}_mod_probs"]
            dir_labels = data[f"{block}_dir_labels"]
            mod_labels = data[f"{block}_mod_labels"]

            # convert to 2d predictions and make conf mat
            y_pred_2d = np.stack([dir_probs.argmax(-1), mod_probs.argmax(-1)], axis=1)
            y_true_2d = np.stack([dir_labels.argmax(-1), mod_labels.argmax(-1)], axis=1)
            conf_mat = confusion_matrix(y_true_2d, y_pred_2d)
            stats = get_stats(conf_mat)
            feedback_results.append(
                Result(
                    single_acc=stats["single_acc"],
                    double_acc=stats["double_acc"],
                    overall_acc=stats["overall_acc"],
                    dir_acc=stats["dir_acc"],
                    mod_acc=stats["mod_acc"],
                    exact_acc=stats["exact_acc"],
                    confusion_matrix=conf_mat,
                )
            )

    # Summarize results from each experimental condition
    print("No Feedback results:")
    summarize_one_condition(nofeedback_results, output_dir, "nofeedback")
    print()

    print("Feedback results:")
    summarize_one_condition(feedback_results, output_dir, "feedback")

    make_boxplots(nofeedback_results, feedback_results, output_dir)


if __name__ == "__main__":
    output_dir = PROJECT_ROOT.parent / "results" / "figures" / "feedback_effect"
    output_dir.mkdir(exist_ok=True, parents=True)

    main(output_dir)
