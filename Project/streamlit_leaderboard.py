import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AutoML Leaderboard", layout="wide")
st.title("AutoML Leaderboard")

RUNS_ROOT = Path("runs")
REPORTS_ROOT = Path("reports")
VISION_LEADERBOARD = REPORTS_ROOT / "leaderboard_vision.csv"
VISION_DATASET_NAME = "Vision (images)"
AUDIO_LEADERBOARD = REPORTS_ROOT / "leaderboard_audio.csv"
AUDIO_DATASET_NAME = "Audio (speech)"
NLP_LEADERBOARD = REPORTS_ROOT / "leaderboard_nlp.csv"
NLP_DATASET_NAME = "Text (NLP)"


def available_datasets() -> list[str]:
    options: list[str] = []
    if RUNS_ROOT.exists():
        for child in sorted(RUNS_ROOT.iterdir()):
            if not child.is_dir():
                continue
            if (child / "reports" / "leaderboard.csv").exists():
                options.append(child.name)
    return options


@st.cache_data(show_spinner=False)
def load_dataset_registry() -> pd.DataFrame:
    registry_path = REPORTS_ROOT / "dataset_registry.json"
    if registry_path.exists():
        try:
            data = json.loads(registry_path.read_text())
            return pd.DataFrame(data)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_framework_registry() -> pd.DataFrame:
    registry_path = REPORTS_ROOT / "framework_registry.json"
    if registry_path.exists():
        try:
            data = json.loads(registry_path.read_text())
            return pd.DataFrame(data)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def prepare_metric_data(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = frame[["framework", metric]].dropna()
    return subset.sort_values(by=metric, ascending=False)


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_timeseries_metrics(path: Path) -> dict[str, object]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def render_leaderboard(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True)
    metric_cols = [col for col in ["f1_macro", "accuracy", "roc_auc_ovr", "avg_precision_ovr"] if col in df.columns]
    if not metric_cols:
        return

    st.markdown("### Metric Visuals")
    tabs = st.tabs(metric_cols)
    for metric, tab in zip(metric_cols, tabs):
        with tab:
            chart_df = prepare_metric_data(df, metric)
            if chart_df.empty:
                st.info("No data available for this metric.")
                continue

            chart_type = st.radio(
                "Chart type",
                ("Bar", "Line", "Pie"),
                horizontal=True,
                key=f"{metric}_chart",
            )

            chart_df = chart_df.sort_values(by=metric, ascending=False)

            if chart_type == "Pie":
                chart = (
                    alt.Chart(chart_df)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta(field=metric, type="quantitative"),
                        color=alt.Color("framework", type="nominal"),
                        tooltip=["framework", alt.Tooltip(metric, format=".4f")],
                    )
                )
            elif chart_type == "Line":
                chart = (
                    alt.Chart(chart_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("framework", type="nominal", sort=None),
                        y=alt.Y(metric, type="quantitative"),
                        tooltip=["framework", alt.Tooltip(metric, format=".4f")],
                    )
                )
            else:
                chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("framework", type="nominal", sort=None),
                        y=alt.Y(metric, type="quantitative"),
                        tooltip=["framework", alt.Tooltip(metric, format=".4f")],
                    )
                )

            st.altair_chart(chart, use_container_width=True)


def render_metric_summary(df: pd.DataFrame) -> None:
    metric_cols = [col for col in ["f1_macro", "accuracy", "roc_auc_ovr", "avg_precision_ovr"] if col in df.columns]
    if not metric_cols:
        return

    st.markdown("### Metric Summary (mean ± std)")

    group_cols = ["framework"]
    if "dataset" in df.columns:
        group_cols.insert(0, "dataset")

    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    grouped = grouped.round(4)
    grouped = grouped.fillna(0.0)
    grouped.columns = ["_".join(filter(None, col)).strip("_") for col in grouped.columns]
    st.dataframe(grouped, use_container_width=True)

    if "dataset" in df.columns:
        dataset_summary = df.groupby("dataset")[metric_cols].agg(["mean", "std"]).round(4).fillna(0.0)
        dataset_summary.columns = ["_".join(filter(None, col)).strip("_") for col in dataset_summary.columns]
        with st.expander("Dataset-level summary"):
            st.dataframe(dataset_summary, use_container_width=True)


dataset_options = available_datasets()
aggregated_path = REPORTS_ROOT / "leaderboard_multi.csv"
vision_df = load_optional_csv(VISION_LEADERBOARD)
audio_df = load_optional_csv(AUDIO_LEADERBOARD)
nlp_df = load_optional_csv(NLP_LEADERBOARD)
if aggregated_path.exists():
    dataset_options.append("Combined (all datasets)")
if not vision_df.empty and VISION_DATASET_NAME not in dataset_options:
    dataset_options.append(VISION_DATASET_NAME)
if not audio_df.empty and AUDIO_DATASET_NAME not in dataset_options:
    dataset_options.append(AUDIO_DATASET_NAME)
if not nlp_df.empty and NLP_DATASET_NAME not in dataset_options:
    dataset_options.append(NLP_DATASET_NAME)

    if not dataset_options:
        fallback = REPORTS_ROOT / "leaderboard.csv"
        if fallback.exists():
            df = pd.read_csv(fallback)
            st.info("Showing fallback leaderboard (single dataset).")
            render_leaderboard(df)
        else:
            st.warning("No leaderboard yet. Run `python scripts/run_all.py` first.")
        st.stop()

selected = st.sidebar.selectbox("Dataset", dataset_options)

if selected == "Combined (all datasets)":
    df = pd.read_csv(aggregated_path)
    if not vision_df.empty:
        if "dataset" not in vision_df.columns:
            vision_df = vision_df.copy()
            vision_df["dataset"] = "vision"
        df = pd.concat([df, vision_df], ignore_index=True)
    if not audio_df.empty:
        if "dataset" not in audio_df.columns:
            audio_df = audio_df.copy()
            audio_df["dataset"] = "audio"
        df = pd.concat([df, audio_df], ignore_index=True)
    if not nlp_df.empty:
        if "dataset" not in nlp_df.columns:
            nlp_df = nlp_df.copy()
            nlp_df["dataset"] = "nlp"
        df = pd.concat([df, nlp_df], ignore_index=True)
    datasets_in_df = sorted(df["dataset"].unique()) if "dataset" in df.columns else []
    if datasets_in_df:
        chosen = st.sidebar.multiselect("Filter datasets", datasets_in_df, default=datasets_in_df)
        if chosen:
            df = df[df["dataset"].isin(chosen)]
    render_leaderboard(df)
    render_metric_summary(df)
elif selected == VISION_DATASET_NAME:
    if vision_df.empty:
        st.error("Vision leaderboard missing. Run the vision training script first.")
        st.stop()
    render_leaderboard(vision_df)
    render_metric_summary(vision_df)
elif selected == AUDIO_DATASET_NAME:
    if audio_df.empty:
        st.error("Audio leaderboard missing. Run the audio training script first.")
        st.stop()
    render_leaderboard(audio_df)
    render_metric_summary(audio_df)
elif selected == NLP_DATASET_NAME:
    if nlp_df.empty:
        st.error("NLP leaderboard missing. Run the NLP training script first.")
        st.stop()
    render_leaderboard(nlp_df)
    render_metric_summary(nlp_df)
else:
    leaderboard_path = RUNS_ROOT / selected / "reports" / "leaderboard.csv"
    if not leaderboard_path.exists():
        st.error(f"No leaderboard for dataset {selected}.")
        st.stop()
    df = pd.read_csv(leaderboard_path)
    render_leaderboard(df)
    render_metric_summary(df)

registry_df = load_dataset_registry()
if not registry_df.empty:
    st.subheader("Dataset Snapshot")
    if selected == "Combined (all datasets)":
        st.dataframe(registry_df, use_container_width=True)
    else:
        subset = registry_df[registry_df["dataset"] == selected]
        st.dataframe(subset, use_container_width=True)

framework_df = load_framework_registry()
if not framework_df.empty:
    st.subheader("Framework Coverage")
    st.dataframe(framework_df, use_container_width=True)

vision_df = load_optional_csv(REPORTS_ROOT / "leaderboard_vision.csv")
if not vision_df.empty:
    st.subheader("Vision Leaderboard")
    st.dataframe(vision_df, use_container_width=True)

timeseries_metrics = load_timeseries_metrics(REPORTS_ROOT / "timeseries_metrics.json")
if timeseries_metrics:
    st.subheader("Time-Series Baseline")
    cols = st.columns(len(timeseries_metrics))
    for (key, value), col in zip(timeseries_metrics.items(), cols):
        with col:
            display_value = value if isinstance(value, (int, float)) else str(value)
            st.metric(label=key, value=display_value)

anomaly_df = load_optional_csv(REPORTS_ROOT / "top_anomalies.csv")
if not anomaly_df.empty:
    st.subheader("Top Anomalies")
    st.dataframe(anomaly_df.head(20), use_container_width=True)
