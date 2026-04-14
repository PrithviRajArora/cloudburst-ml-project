# ☁️ CloudBurst Prediction using Machine Learning

A Machine Learning project that builds a binary classification system to predict whether a **cloudburst will occur the next day**, using today's meteorological readings as input features.

---

## 📁 Project Structure

```
Project 3/
├── CloudBurst_ML_Project.ipynb        # Main Jupyter Notebook (all 4 phases)
├── cloudpredictionsystemproject.csv   # Dataset (145,460 rows × 23 columns)
├── README.md                          # This file
└── Report.md                          # Full project report
```

---

## 📊 Dataset

- **Source:** [Kaggle — CloudBurst Dataset](https://www.kaggle.com/datasets/akshat234/cloudburst)
- **Size:** 145,460 rows × 23 columns
- **Location:** Multiple Australian cities (Albury, etc.)
- **Target Variable:** `CloudBurstTomorrow` — Will a cloudburst occur tomorrow? (Yes/No)

The dataset file `cloudpredictionsystemproject.csv` is included in this repository.

---

## 🧰 Requirements

### Python Version
- Python **3.8+** recommended

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## 🚀 How to Run

### Option 1 — Jupyter Notebook (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prithvirajarora/cloudburst-ml-project.git
   cd cloudburst-ml-project
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open** `CloudBurst_ML_Project.ipynb` in the browser tab that opens.

5. **Run all cells:**  
   Go to **Kernel → Restart & Run All** to execute the full pipeline end-to-end.

### Option 2 — VS Code with Jupyter Extension

1. Open the project folder in VS Code.
2. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
3. Open `CloudBurst_ML_Project.ipynb`.
4. Click **Run All** at the top of the notebook.

---

## 📋 Project Phases

The notebook is organized into **4 phases** matching the grading rubric:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Data Understanding & Preprocessing — EDA, missing value handling, encoding |
| **Phase 2** | Supervised Learning — Logistic Regression baseline, metrics & evaluation |
| **Phase 3** | Optimization & Unsupervised Learning — Regularization, PCA, K-Means Clustering |
| **Phase 4** | Advanced Models — Decision Tree, Random Forest, Hyperparameter Tuning, Final Comparison |

---

## 📈 Key Results

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression (Baseline) | ~0.76 | ~0.60 | ~0.81 |
| Logistic Regression (Optimized) | ~0.77 | ~0.61 | ~0.82 |
| Decision Tree (Tuned) | ~0.78 | ~0.62 | ~0.83 |
| **Random Forest (Tuned)** ✅ | **~0.82** | **~0.67** | **~0.88** |

**Final Model:** Tuned Random Forest — best F1 Score and AUC-ROC across all models.

---

## ⚠️ Notes

- The notebook expects the dataset file `cloudpredictionsystemproject.csv` to be in the **same directory** as the notebook.
- Training may take **a few minutes** on the full dataset (145K rows), especially during GridSearchCV for Random Forest.
- All plots are generated inline in the notebook.

---

## 📄 Report

See [`Report.md`](./Report.md) for the full written project report.

---

## 🏫 Course Info

- **Course:** Machine Learning
- **Institution:** JLU (Jagran Lakecity University)
- **Semester:** 6
