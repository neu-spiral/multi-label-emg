from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from run_experiment_1 import settings

from multi_label_emg.train import get_name

colors = px.colors.qualitative.Plotly


BOXPLOT_WIDTH = 16 * 100
BOXPLOT_HEIGHT = 9 * 100
SCALE = 2
FONT_SIZE = 18


def get_color(name):
    colorscale = px.colors.qualitative.Plotly
    prefixes = {
        "baseline": colorscale[0],
        "lower_bound": colorscale[7],
        "upper_bound": colorscale[8],
    }
    for prefix, color in prefixes.items():
        if name.startswith(prefix):
            return color
    raise ValueError(f"Unknown name: {name}")


def boxplots(df, output_dir):
    fig_singles = go.Figure()
    fig_doubles = go.Figure()
    fig_overall = go.Figure()

    # For each choice of model arch and clf alg, show the results of lower bound, upper bound, and augmented
    # First, collect the results
    lower_bound = {"x": [], "single_accs": [], "double_accs": [], "overall_accs": []}
    upper_bound = {"x": [], "single_accs": [], "double_accs": [], "overall_accs": []}
    augmented = {"x": [], "single_accs": [], "double_accs": [], "overall_accs": []}
    for group_details, group in df.groupby(["parallel_model_type", "clf_name"]):
        print("-" * 80)
        print(f"Group details: {group_details}")
        print(f"Group shape: {group.shape}")
        print()

        # Get results from "upper bound" model
        upper_bound_group = group[
            (group["doubles_method"] == "none")
            & (group["singles_method"] == "none")
            & (group["include_doubles_in_train"])
        ]
        N = len(upper_bound_group)

        # The x-axis coords for this group are shared for lower, upper, and aug models
        # (They just describe the parallel model and clf alg)
        x = [", ".join(group_details)] * N

        upper_bound["x"].extend(x)
        upper_bound["single_accs"].extend(upper_bound_group["singles_acc"])
        upper_bound["double_accs"].extend(upper_bound_group["doubles_acc"])
        upper_bound["overall_accs"].extend(upper_bound_group["overall_acc"])

        # Get results from "lower bound" model
        # NOTE - the "SerialControl" model will not have a lower bound (it must see all classes at train time)
        lower_bound_group = group[
            (group["doubles_method"] == "none")
            & (group["singles_method"] == "none")
            & (~group["include_doubles_in_train"])
        ]
        lower_bound["x"].extend(x)
        lower_bound["single_accs"].extend(lower_bound_group["singles_acc"])
        lower_bound["double_accs"].extend(lower_bound_group["doubles_acc"])
        lower_bound["overall_accs"].extend(lower_bound_group["overall_acc"])

        # Get results from augmented model
        augmented_group = group[
            (group["doubles_method"] == "all")
            & (group["singles_method"] == "none")
            & (~group["include_doubles_in_train"])
        ]
        augmented["x"].extend(x)
        augmented["single_accs"].extend(augmented_group["singles_acc"])
        augmented["double_accs"].extend(augmented_group["doubles_acc"])
        augmented["overall_accs"].extend(augmented_group["overall_acc"])

    # Plot the data
    # Add info for the lower bound models
    shared_kw = dict(pointpos=0, boxmean=True, boxpoints="all", jitter=0.8)
    lower_kw = dict(name="Lower Bound", marker_color=get_color("lower_bound"), **shared_kw)
    fig_singles.add_trace(go.Box(x=lower_bound["x"], y=lower_bound["single_accs"], **lower_kw))
    fig_doubles.add_trace(go.Box(x=lower_bound["x"], y=lower_bound["double_accs"], **lower_kw))
    fig_overall.add_trace(go.Box(x=lower_bound["x"], y=lower_bound["overall_accs"], **lower_kw))

    # Add info for upper bound models
    upper_kw = dict(name="Upper Bound", marker_color=get_color("upper_bound"), **shared_kw)
    fig_singles.add_trace(go.Box(x=upper_bound["x"], y=upper_bound["single_accs"], **upper_kw))
    fig_doubles.add_trace(go.Box(x=upper_bound["x"], y=upper_bound["double_accs"], **upper_kw))
    fig_overall.add_trace(go.Box(x=upper_bound["x"], y=upper_bound["overall_accs"], **upper_kw))

    # Add info for the augmented models
    aug_kw = dict(name="Synthetic", marker_color=get_color("baseline"), **shared_kw)
    fig_singles.add_trace(go.Box(x=augmented["x"], y=augmented["single_accs"], **aug_kw))
    fig_doubles.add_trace(go.Box(x=augmented["x"], y=augmented["double_accs"], **aug_kw))
    fig_overall.add_trace(go.Box(x=augmented["x"], y=augmented["overall_accs"], **aug_kw))

    # Adjust plot layout
    layout_kw = dict(
        template="simple_white",
        height=BOXPLOT_HEIGHT,
        width=BOXPLOT_WIDTH,
        yaxis_title="Balanced Accuracy",
        xaxis_title="Model Architecture, Classifier Algorithm",
        boxmode="group",
        title_x=0.5,
        title_y=0.9,
        # boxgroupgap=0.0,
        # boxgap=0.3,
        yaxis_range=[0, 1],
        font_size=FONT_SIZE,
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=1),
    )
    fig_singles.update_layout(title="Single Gestures", **layout_kw)
    fig_doubles.update_layout(title="Double Gestures", **layout_kw)
    fig_overall.update_layout(title="Overall", **layout_kw)

    # Add line for random change level
    hline_kw = dict(
        line_dash="dash",
        line_color="gray",
        opacity=0.8,
        line_width=1,
        annotation_text="Chance Level",
        annotation_position="bottom right",
    )
    fig_singles.add_hline(y=1 / 15, **hline_kw)
    fig_doubles.add_hline(y=1 / 15, **hline_kw)
    fig_overall.add_hline(y=1 / 15, **hline_kw)

    fig_singles.write_html(output_dir / "singles.html", include_plotlyjs="cdn")
    fig_doubles.write_html(output_dir / "doubles.html", include_plotlyjs="cdn")
    fig_overall.write_html(output_dir / "overall.html", include_plotlyjs="cdn")

    fig_singles.write_image(output_dir / "singles.png", scale=2)
    fig_doubles.write_image(output_dir / "doubles.png", scale=2)
    fig_overall.write_image(output_dir / "overall.png", scale=2)


def main(results_dir: Path, output_dir: Path):
    records = []
    missing = []

    for S in settings:
        name = get_name(
            subject=S.subject,
            seed=S.seed,
            parallel_model_type=S.parallel_model_type,
            clf_name=S.clf_name,
            doubles_method=S.doubles_method,
            fraction_doubles_per_class=S.fraction_doubles_per_class,
            singles_method=S.singles_method,
            rel_fraction_singles_per_class=S.rel_fraction_singles_per_class,
            include_doubles_in_train=S.include_doubles_in_train,
            feature_combine_type=S.feature_combine_type,
        )
        name += ".npy"
        if not (results_dir / name).exists():
            missing.append(results_dir / name)
            continue

        record = asdict(S)
        record["cm"] = np.load(results_dir / name)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df["singles_acc"] = df["cm"].apply(lambda cm: np.diag(cm)[:6].mean())
    df["doubles_acc"] = df["cm"].apply(lambda cm: np.diag(cm)[6:].mean())
    df["overall_acc"] = df["cm"].apply(lambda cm: np.diag(cm).mean())

    # Use different terminology for parallel model structure
    df["parallel_model_type"] = df["parallel_model_type"].replace(
        {"ParallelA": "Parallel", "ParallelB": "Hierarchical", "SerialControl": "Serial"}
    )
    df["clf_name"] = df["clf_name"].replace({"mlp": "MLP", "rf": "RF", "logr": "LogR"})

    try:
        boxplots(df, output_dir)
    finally:
        if missing:
            missing_file = output_dir / "missing.txt"
            print("*" * 80)
            print(f"MISSING: {len(missing)} RUNS!")
            print(f"Files printed to: {missing_file}")
            print("*" * 80)
            with open(missing_file, "w") as f:
                for m in missing:
                    print(m, file=f)


if __name__ == "__main__":
    from multi_label_emg.utils import RESULTS_DIR

    experiments_dir = RESULTS_DIR / "experiments"
    output_dir = RESULTS_DIR / "figures" / "experiment_1"
    output_dir.mkdir(exist_ok=True, parents=True)
    main(experiments_dir, output_dir)
