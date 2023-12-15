from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from run_experiment_1 import settings as settings1
from run_experiment_2 import settings as settings2

from multi_label_emg.train import get_name

BOXPLOT_WIDTH = 16 * 100
BOXPLOT_HEIGHT = 9 * 100
SCALE = 2
FONT_SIZE = 18


def get_color(name):
    colorscale = px.colors.qualitative.Plotly
    prefixes = {
        "baseline": colorscale[0],
        "subset_uniform": colorscale[1],
        "subset_near_mean": colorscale[2],
        "subset_spaced_quantiles": colorscale[3],
        "subsetInput_uniform": colorscale[4],
        "subsetInput_near_mean": colorscale[5],
        "subsetInput_spaced_quantiles": colorscale[6],
        "lower_bound": colorscale[7],
        "upper_bound": colorscale[8],
    }
    for prefix, color in prefixes.items():
        if name.startswith(prefix):
            return color
    raise ValueError(f"Unknown name: {name}")


def get_marker_size(frac_doubles):
    sizes = np.arange(7) * 4
    fracs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    i = fracs.index(frac_doubles)
    return sizes[i]


def boxplots(df, output_dir):
    fig_singles = go.Figure()
    fig_doubles = go.Figure()
    fig_overall = go.Figure()

    x_axis_values = []
    x_axis_text = []

    hline_kw = dict(line_dash="dash", opacity=1.0, line_width=2)

    # For each choice of model arch and clf alg, show the results of lower bound, upper bound, and augmented
    baseline_df = df[(df["doubles_method"] == "all")]

    # Add boxplot for baseline
    box_kw = dict(pointpos=0, boxmean=True, boxpoints="all", jitter=0.8)
    N = len(baseline_df)
    name = "baseline"
    x = [name] * N
    x_axis_values.append(name)
    x_axis_text.append("Baseline")
    kw = dict(name=name, marker_color=get_color(name), showlegend=True, **box_kw)
    fig_singles.add_trace(go.Box(x=x, y=baseline_df["singles_acc"], **kw))
    fig_doubles.add_trace(go.Box(x=x, y=baseline_df["doubles_acc"], **kw))
    fig_overall.add_trace(go.Box(x=x, y=baseline_df["overall_acc"], **kw))
    # Add h lines for mean
    kw = dict(line_color=get_color("baseline"), annotation_position="top right", **hline_kw)
    fig_singles.add_hline(y=np.mean(baseline_df["singles_acc"]), annotation_text="Baseline", **kw)
    fig_doubles.add_hline(y=np.mean(baseline_df["doubles_acc"]), annotation_text="Baseline", **kw)
    fig_overall.add_hline(y=np.mean(baseline_df["overall_acc"]), annotation_text="Baseline", **kw)

    # Add box for lower bound
    lower_bound_df = df[
        (df["doubles_method"] == "none") & (df["singles_method"] == "none") & (~df["include_doubles_in_train"])
    ]
    N = len(lower_bound_df)
    name = "lower_bound"
    x = [name] * N
    x_axis_values.append(name)
    x_axis_text.append("Lower Bound")
    kw = dict(name=name, marker_color=get_color(name), showlegend=True, **box_kw)
    fig_singles.add_trace(go.Box(x=x, y=lower_bound_df["singles_acc"], **kw))
    fig_doubles.add_trace(go.Box(x=x, y=lower_bound_df["doubles_acc"], **kw))
    fig_overall.add_trace(go.Box(x=x, y=lower_bound_df["overall_acc"], **kw))
    # Add h lines
    kw = dict(line_color=get_color("lower_bound"), annotation_position="bottom right", **hline_kw)
    fig_singles.add_hline(y=np.mean(lower_bound_df["singles_acc"]), annotation_text="Lower Bound", **kw)
    fig_doubles.add_hline(y=np.mean(lower_bound_df["doubles_acc"]), annotation_text="Lower Bound", **kw)
    fig_overall.add_hline(y=np.mean(lower_bound_df["overall_acc"]), annotation_text="Lower Bound", **kw)

    # Add box for upper bound
    upper_bound_df = df[
        (df["doubles_method"] == "none") & (df["singles_method"] == "none") & (df["include_doubles_in_train"])
    ]
    N = len(upper_bound_df)
    name = "upper_bound"
    x = [name] * N
    x_axis_values.append(name)
    x_axis_text.append("Upper Bound")
    kw = dict(name=name, marker_color=get_color(name), showlegend=True, **box_kw)
    fig_singles.add_trace(go.Box(x=x, y=upper_bound_df["singles_acc"], **kw))
    fig_doubles.add_trace(go.Box(x=x, y=upper_bound_df["doubles_acc"], **kw))
    fig_overall.add_trace(go.Box(x=x, y=upper_bound_df["overall_acc"], **kw))
    # Add h lines
    kw = dict(line_color=get_color("upper_bound"), annotation_position="top right", **hline_kw)
    fig_singles.add_hline(y=np.mean(upper_bound_df["singles_acc"]), annotation_text="Upper Bound", **kw)
    fig_doubles.add_hline(y=np.mean(upper_bound_df["doubles_acc"]), annotation_text="Upper Bound", **kw)
    fig_overall.add_hline(y=np.mean(upper_bound_df["overall_acc"]), annotation_text="Upper Bound", **kw)

    other_aug = df[(df["doubles_method"] != "all") & (df["doubles_method"] != "none")]
    groups_seen = {}
    for group_details, group in other_aug.groupby(["doubles_method", "printable_frac"]):
        showlegend = True
        if group_details[0] in groups_seen:
            showlegend = False
        else:
            groups_seen[group_details[0]] = True
        print("-" * 80)
        print(f"Group details: {group_details}")
        print(f"Group shape: {group.shape}")
        print()

        N = len(group)
        hidden_name = ", ".join(map(str, group_details))
        x = [hidden_name] * N
        x_axis_values.append(hidden_name)
        x_axis_text.append(f"f={group_details[1]}")
        kw = dict(name=group_details[0], marker_color=get_color(group_details[0]), showlegend=showlegend, **box_kw)
        fig_singles.add_trace(go.Box(x=x, y=group["singles_acc"], **kw))
        fig_doubles.add_trace(go.Box(x=x, y=group["doubles_acc"], **kw))
        fig_overall.add_trace(go.Box(x=x, y=group["overall_acc"], **kw))

    # Adjust plot layout
    layout_kw = dict(
        template="simple_white",
        height=BOXPLOT_HEIGHT,
        width=BOXPLOT_WIDTH,
        title_x=0.5,
        title_y=0.85,
        xaxis=dict(
            title="Method (f = Fraction Doubles per Class)",
            tickmode="array",
            tickvals=x_axis_values,
            ticktext=x_axis_text,
        ),
        yaxis_title="Balanced Acc",
        boxgap=0.1,
        boxgroupgap=0,
        font_size=FONT_SIZE,
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.9, xanchor="right", x=1.0),
    )
    fig_singles.update_layout(title="Singles Acc", **layout_kw)
    fig_doubles.update_layout(title="Doubles Acc", **layout_kw)
    fig_overall.update_layout(title="Overall Acc", **layout_kw)

    # Save output
    fig_singles.write_html(output_dir / "singles.html", include_plotlyjs="cdn")
    fig_doubles.write_html(output_dir / "doubles.html", include_plotlyjs="cdn")
    fig_overall.write_html(output_dir / "overall.html", include_plotlyjs="cdn")

    fig_singles.write_image(output_dir / "singles.png", scale=SCALE)
    fig_doubles.write_image(output_dir / "doubles.png", scale=SCALE)
    fig_overall.write_image(output_dir / "overall.png", scale=SCALE)


def main(results_dir: Path, output_dir: Path):
    records = []
    missing = []

    # Get the baseline settings from experiment 1
    lower_bounds = []
    upper_bounds = []
    aug_without_subset = []
    for S in settings1:
        if S.parallel_model_type != "ParallelA" or S.clf_name != "mlp":
            continue
        if S.doubles_method == "none" and S.singles_method == "none":
            if not S.include_doubles_in_train:
                lower_bounds.append(S)
            else:
                upper_bounds.append(S)
        if S.doubles_method == "all":
            aug_without_subset.append(S)

    # Combine these with all settings from experiment 2
    settings = settings2 + lower_bounds + upper_bounds + aug_without_subset

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

    # Simplify some text names for the dataframe
    replacements = {
        0.0316: 0.001,
        0.0707: 0.005,
        0.1: 0.01,
        0.2236: 0.05,
        0.3162: 0.1,
        0.5: 0.25,
        0.7071: 0.5,
    }

    def get_printable_frac(method, frac):
        if method.startswith("subsetInput"):
            return replacements[frac]
        else:
            return frac

    df["printable_frac"] = df.apply(
        lambda row: get_printable_frac(row["doubles_method"], row["fraction_doubles_per_class"]), axis=1
    )

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

    output_dir = RESULTS_DIR / "figures" / "experiment_2"
    output_dir.mkdir(exist_ok=True, parents=True)
    experiments_dir = RESULTS_DIR / "experiments"
    main(experiments_dir, output_dir)
