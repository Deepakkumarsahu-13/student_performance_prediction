# 🎓 StudentScore AI — Student Exam Performance Predictor

A complete end-to-end machine learning project that **generates** a synthetic student dataset,
**analyses** study-habit patterns, **visualises** key trends, and **predicts** final exam scores
using a Random Forest regression model.

---


## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | 1 200 synthetic student records, 9 features |
| **Target** | `FinalExamScore` (0–100) |
| **Model** | Random Forest Regressor (200 trees) |
| **MAE** | ~4.5 pts |
| **RMSE** | ~5.8 pts |
| **R²** | ~0.75 |

The dataset captures realistic relationships between lifestyle choices and academic outcomes.
Features cover study effort, attendance, sleep, social-media use, previous scores, extracurricular
participation, and internet usage.

---

## 📦 Dataset Generation

The dataset is **fully synthetic** — generated via NumPy with a deterministic seed so results are
reproducible. The `FinalExamScore` is defined by the weighted formula:

```
FinalScore = 35
           + 4.20 × StudyHoursPerDay
           + 0.25 × AttendancePercentage
           + 1.50 × SleepHours
           − 2.00 × SocialMediaHours
           + 0.22 × PreviousExamScore
           + 3.50 × ParticipationInActivities
           − 0.80 × InternetUsageHours
           + ε  (Gaussian noise, σ = 5)
```

About **2 % of values** in three columns are deliberately set to `NaN` to simulate real-world
missing data, then imputed with column medians during preprocessing.

### Feature Descriptions

| Feature | Type | Range | Description |
|---------|------|--------|-------------|
| `StudentID` | str | — | Unique identifier |
| `StudyHoursPerDay` | float | 0–12 | Daily study time (hrs) |
| `AttendancePercentage` | float | 30–100 | Class attendance rate |
| `SleepHours` | float | 3–10 | Average daily sleep (hrs) |
| `SocialMediaHours` | float | 0–10 | Daily social media use (hrs) |
| `PreviousExamScore` | float | 20–100 | Score on prior exam |
| `ParticipationInActivities` | int | 0/1 | Extracurricular participation |
| `InternetUsageHours` | float | 0–14 | Daily internet use (hrs) |
| `FinalExamScore` | float | 0–100 | **Target variable** |

---

## 🛠️ Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Data generation & numerical ops |
| `pandas` | ≥2.0 | DataFrame manipulation |
| `matplotlib` | ≥3.7 | Base plotting |
| `seaborn` | ≥0.12 | Statistical visualisations |
| `scikit-learn` | ≥1.3 | ML model, metrics, train/test split |
| `joblib` | ≥1.3 | Model serialisation |
| `nbformat` | ≥5.9 | Notebook generation (build step only) |

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib nbformat
```

---

## 🚀 How to Run

### Option A — Python script (full pipeline + interactive predictor)

```bash
python student_performance.py
```

Flags:

| Flag | Effect |
|------|--------|
| `--skip-plots` | Skip generating PNG visualisations |
| `--no-console` | Run pipeline only; skip interactive predictor |

Example (pipeline only, no plots, no console):

```bash
python student_performance.py --skip-plots --no-console
```

### Option B — Jupyter Notebook

```bash
jupyter notebook student_performance.ipynb
```

Run all cells top-to-bottom. Each section is self-contained and annotated.

### Option C — Predict a single student in Python

```python
from student_performance import predict_score   # or use the helper in the notebook

score = predict_score(
    study_hrs        = 6,
    attendance_pct   = 85,
    sleep_hrs        = 7.5,
    social_media_hrs = 1.5,
    prev_score       = 72,
    participation    = 1,
    internet_hrs     = 3,
)
print(f"Predicted score: {score} / 100")
```

---

## 📊 Key Insights from Analysis

### Correlation with Final Exam Score

| Feature | Correlation |
|---------|-------------|
| StudyHoursPerDay | **+0.64** (strong positive) |
| AttendancePercentage | **+0.31** |
| PreviousExamScore | **+0.28** |
| SleepHours | **+0.16** |
| ParticipationInActivities | **+0.16** |
| InternetUsageHours | **−0.15** |
| SocialMediaHours | **−0.26** (moderate negative) |

### Feature Importance (Random Forest)

1. 🥇 **PreviousExamScore** — 32 % (strongest single predictor)
2. 🥈 **StudyHoursPerDay** — 24 % (consistent effort = results)
3. 🥉 **AttendancePercentage** — 16 %
4. **SleepHours** — 9 %
5. **SocialMediaHours** — 8 %
6. **InternetUsageHours** — 7 %
7. **ParticipationInActivities** — 4 %

### Participation Bonus

Students who participate in extracurricular activities score, on average,
**~3–4 points higher** than non-participants, despite the extra time commitment.

### Practical Takeaways

- ✅ **Study consistently** — every +1 hr/day ≈ +4 exam points
- ✅ **Maintain ≥ 80 % attendance** — has compounding effect
- ✅ **Sleep 7–8 hrs** — optimal recovery improves retention
- ❌ **Limit social media** — each extra hour costs ~2 pts

---

## ⚙️ Model Performance Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | ~4.5 pts | Avg absolute prediction error |
| **RMSE** | ~5.8 pts | Penalises large errors more |
| **R²** | ~0.75 | Explains 75 % of score variance |
| Train / Test split | 80 / 20 | 960 train, 240 test samples |
| Estimators | 200 trees | Balanced accuracy vs speed |

An R² of 0.75 is strong for a purely behavioural / lifestyle dataset where many
real-world factors (topic difficulty, teacher quality, test anxiety) are naturally
absent from the features.

---

## 📄 License

MIT — free to use, modify, and distribute.
