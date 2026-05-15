"""
--------------------------------------------------------------------------------
  StudentScore AI — Student Exam Performance Predictor
--------------------------------------------------------------------------------
  Author  : StudentScore AI Project
  Purpose : Generate synthetic student data, perform EDA, visualise patterns,
            train a Random Forest regression model, and provide an interactive
            console-based score predictor.

  Usage:
      python student_performance.py                # full pipeline + console UI
      python student_performance.py --skip-plots   # skip visualisation step
--------------------------------------------------------------------------------
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── constants 
RANDOM_SEED   = 42
N_STUDENTS    = 1200
TEST_SIZE     = 0.20
MODEL_PATH    = "rf_model.pkl"
DATA_PATH     = "student_data.csv"

FEATURES = [
    "StudyHoursPerDay",
    "AttendancePercentage",
    "SleepHours",
    "SocialMediaHours",
    "PreviousExamScore",
    "ParticipationInActivities",
    "InternetUsageHours",
]
TARGET = "FinalExamScore"

# Visual theme
BG, ACCENT, TEAL, CORAL = "#0d0f1a", "#7c6af7", "#43d9a2", "#f07167"
TEXT, GRID              = "#e8e6f0", "#1e2138"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "font.family": "DejaVu Sans",
})


#1. DATA GENERATION 

def generate_dataset(n: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Synthesise a realistic student performance dataset.

    The final exam score is constructed as a weighted linear combination of
    study habits and lifestyle features, plus Gaussian noise, so that the
    resulting dataset exhibits real-world-plausible correlations.

    Returns
    -------
    pd.DataFrame  — shape (n, 9)
    """
    rng = np.random.default_rng(seed)

    def clamp(arr, lo, hi):
        return np.clip(arr, lo, hi)

    study_hours   = clamp(rng.normal(4.0, 1.8, n),  0,  12)
    attendance    = clamp(rng.normal(75,  15,  n),  30, 100)
    sleep_hours   = clamp(rng.normal(7.0, 1.2, n),   3,  10)
    social_media  = clamp(rng.normal(3.0, 1.5, n),   0,  10)
    prev_score    = clamp(rng.normal(65,  15,  n),  20, 100)
    participation = rng.binomial(1, 0.55, n)
    internet      = clamp(rng.normal(5.0, 2.0, n),   0,  14)

    noise = rng.normal(0, 5, n)
    final = (
        35
        + 4.20 * study_hours
        + 0.25 * attendance
        + 1.50 * sleep_hours
        - 2.00 * social_media
        + 0.22 * prev_score
        + 3.50 * participation
        - 0.80 * internet
        + noise
    )
    final = clamp(final, 20, 100)

    df = pd.DataFrame({
        "StudentID":                [f"STU{1000 + i}" for i in range(n)],
        "StudyHoursPerDay":         study_hours.round(2),
        "AttendancePercentage":     attendance.round(2),
        "SleepHours":               sleep_hours.round(2),
        "SocialMediaHours":         social_media.round(2),
        "PreviousExamScore":        prev_score.round(2),
        "ParticipationInActivities": participation,
        "InternetUsageHours":       internet.round(2),
        "FinalExamScore":           final.round(2),
    })

    # Inject ~2 % missing values in three columns to simulate real data
    for col in ["StudyHoursPerDay", "SleepHours", "SocialMediaHours"]:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


# 2. PREPROCESSING 

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      • Median-impute missing values in numeric columns
      • Drop duplicate StudentIDs (none expected in synthetic data)
      • Reset the index

    Returns
    -------
    pd.DataFrame  — cleaned copy
    """
    df = df.copy()

    # Median imputation for columns that may contain NaN
    impute_cols = ["StudyHoursPerDay", "SleepHours", "SocialMediaHours"]
    for col in impute_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [impute] '{col}': {n_missing} NaN → median ({median_val:.2f})")

    df.drop_duplicates(subset="StudentID", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# 3. EXPLORATORY DATA ANALYSIS      

def run_eda(df: pd.DataFrame) -> None:
    """Print key descriptive statistics to stdout."""
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"\n  Shape         : {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print("\n── Descriptive Statistics ──────────────────────────────────")
    print(df[FEATURES + [TARGET]].describe().round(2).to_string())

    print("\n── Correlation with FinalExamScore ─────────────────────────")
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print(corr.round(4).to_string())

    part_grp = df.groupby("ParticipationInActivities")[TARGET].mean()
    print(f"\n── Activity Participation Impact ───────────────────────────")
    print(f"  Non-participants avg score : {part_grp[0]:.2f}")
    print(f"  Participants avg score     : {part_grp[1]:.2f}")
    print(f"  Difference                 : {part_grp[1] - part_grp[0]:+.2f} pts")


# 4. VISUALISATIONS   

def _save(fig, name: str) -> None:
    fig.savefig(name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {name}")


def plot_scatter(df: pd.DataFrame) -> None:
    """Scatter: Study Hours vs Final Exam Score, coloured by Attendance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        df["StudyHoursPerDay"], df["FinalExamScore"],
        c=df["AttendancePercentage"], cmap="plasma",
        alpha=0.65, s=28, linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Attendance %", color=TEXT, fontsize=11)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

    # Trend line
    z = np.polyfit(df["StudyHoursPerDay"], df["FinalExamScore"], 1)
    xs = np.linspace(0, 12, 200)
    ax.plot(xs, np.poly1d(z)(xs), color=CORAL, linewidth=2.5,
            linestyle="--", label=f"Trend  (slope ≈ {z[0]:.2f})")

    ax.set_title("Study Hours vs Final Exam Score", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Study Hours Per Day", fontsize=12)
    ax.set_ylabel("Final Exam Score", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save(fig, "fig1_scatter.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    """Lower-triangle correlation heatmap for all numeric features."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
        annot=True, fmt=".2f", linewidths=0.5, linecolor=BG,
        annot_kws={"size": 9, "color": TEXT},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    _save(fig, "fig2_heatmap.png")


def plot_participation_bar(df: pd.DataFrame) -> None:
    """Bar chart: mean ± std score grouped by activity participation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    grp = df.groupby("ParticipationInActivities")[TARGET].agg(["mean", "std"])
    labels = ["Non-Participants", "Participants"]
    colors = [CORAL, TEAL]
    bars = ax.bar(
        labels, grp["mean"], color=colors, width=0.5,
        yerr=grp["std"], capsize=8,
        error_kw={"ecolor": TEXT, "alpha": 0.5},
    )
    for bar, val in zip(bars, grp["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.8,
                f"{val:.1f}", ha="center", fontsize=14, fontweight="bold", color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Average Final Exam Score", fontsize=12)
    ax.set_title("Extracurricular Participation vs Exam Performance",
                 fontsize=14, fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig3_bar.png")


def plot_feature_importance(model: RandomForestRegressor) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(9, 5))
    imp = model.feature_importances_
    idx = np.argsort(imp)
    palette = [TEAL if i == idx[-1] else ACCENT for i in range(len(idx))]
    ax.barh([FEATURES[i] for i in idx], imp[idx], color=palette)
    ax.set_title("Random Forest Feature Importances", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig4_importance.png")


def create_visualisations(df: pd.DataFrame, model: RandomForestRegressor) -> None:
    print("\n" + "=" * 60)
    print("  VISUALISATIONS")
    print("=" * 60)
    plot_scatter(df)
    plot_heatmap(df)
    plot_participation_bar(df)
    plot_feature_importance(model)


    # 5. MODEL TRAINING & EVALUATION  

def train_model(df: pd.DataFrame):
    """
    Split data, train a RandomForestRegressor, evaluate on the test set,
    persist the model to disk, and return (model, metrics_dict).
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    print("  Training RandomForestRegressor (200 trees) …", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("done.")

    y_pred = model.predict(X_test)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2":   r2_score(y_test, y_pred),
    }

    print(f"\n  ── Evaluation Results ──────────────────────────────────")
    print(f"  MAE  (Mean Absolute Error)  : {metrics['MAE']:.4f}")
    print(f"  RMSE (Root Mean Sq. Error)  : {metrics['RMSE']:.4f}")
    print(f"  R²   (Coefficient of Det.)  : {metrics['R2']:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")
    return model, metrics


# 6. INTERACTIVE CONSOLE PREDICTOR   

def _ask_float(prompt: str, lo: float, hi: float) -> float:
    """Prompt the user for a float in [lo, hi], re-asking on invalid input."""
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"    ⚠  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("    ⚠  Invalid input — please enter a number.")


def _ask_binary(prompt: str) -> int:
    """Prompt the user for 0 or 1, re-asking on invalid input."""
    while True:
        raw = input(f"  {prompt} [0 = No / 1 = Yes]: ").strip()
        if raw in ("0", "1"):
            return int(raw)
        print("    ⚠  Please enter 0 or 1.")


def grade_label(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def run_console_predictor(model: RandomForestRegressor) -> None:
    """Loop: collect student profile → predict → show result → repeat or quit."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE SCORE PREDICTOR")
    print("=" * 60)
    print("  Enter student details to predict the Final Exam Score.")
    print("  Type 'quit' at any prompt to exit.\n")

    while True:
        print("─" * 50)
        try:
            study   = _ask_float("Study Hours Per Day",          0,  12)
            attend  = _ask_float("Attendance Percentage",        30, 100)
            sleep   = _ask_float("Sleep Hours Per Day",          3,  10)
            social  = _ask_float("Social Media Hours Per Day",   0,  10)
            prev    = _ask_float("Previous Exam Score",         20, 100)
            inet    = _ask_float("Internet Usage Hours Per Day", 0,  14)
            part    = _ask_binary("Participates in Activities?")
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            return

        row = pd.DataFrame([{
            "StudyHoursPerDay":           study,
            "AttendancePercentage":       attend,
            "SleepHours":                 sleep,
            "SocialMediaHours":           social,
            "PreviousExamScore":          prev,
            "ParticipationInActivities":  part,
            "InternetUsageHours":         inet,
        }])

        predicted = float(model.predict(row)[0])
        predicted = round(max(0, min(100, predicted)), 2)
        g         = grade_label(predicted)

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  Predicted Final Exam Score : {predicted:5.1f}  │")
        print(f"  │  Grade                      :   {g:<5}  │")
        print(f"  └─────────────────────────────────────┘\n")

        again = input("  Predict another student? [y/n]: ").strip().lower()
        if again != "y":
            print("\n  Goodbye!")
            return

# MAIN    

def main():
    parser = argparse.ArgumentParser(description="Student Performance Predictor")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip visualisation step (useful in headless CI)")
    parser.add_argument("--no-console", action="store_true",
                        help="Run pipeline only, skip interactive predictor")
    args = parser.parse_args()

    # ── 1. Generate / load data 
    print("\n" + "=" * 60)
    print("  DATA GENERATION")
    print("=" * 60)
    df_raw = generate_dataset()
    df_raw.to_csv(DATA_PATH, index=False)
    print(f"  Generated {len(df_raw)} student records → {DATA_PATH}")
    print(f"  Missing values before cleaning:\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0].to_string()}")

    # ── 2. Preprocess 
    print("\n" + "=" * 60)
    print("  PREPROCESSING")
    print("=" * 60)
    df = preprocess(df_raw)
    print(f"  Clean dataset shape: {df.shape}")

    # ── 3. EDA 
    run_eda(df)

    # ── 4. Train model 
    model, metrics = train_model(df)

    # ── 5. Visualise 
    if not args.skip_plots:
        create_visualisations(df, model)

    # ── 6. Console predictor 
    if not args.no_console:
        run_console_predictor(model)


if __name__ == "__main__":
    main()
