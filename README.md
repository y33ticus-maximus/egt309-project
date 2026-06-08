# egt309-project 

PROJECT TIMELINE: 
## Week 6
### (25-26 May) Mon-Tues
- created dockerfile (YS)
- created codespace (YS)
- created & finished upload data section (YS)
- created EDA section, WIP (YS + Nithish)
## Week 7
### (31 May - 1 June) Sun-Mon
- Finished EDA (Nithish + Brendan)
- Started MLP (Nithish + Brendan + YS)
## Week 8


## Project Info

# ElderGuard Activity-Level Prediction — Logistic Regression Pipeline

# PROJECT BACKGROUND:
Many Elderly residents are living independently without adequtae care or awareness of their surroundings. Therefore there is a need to alert them of potentially hazardous conditions that could be detrimental to their health and wellbeing. One such condition could be the presence of high concentrations or volumes of gases such as Carbon Dioxide (CO2), Carbon Mononxide (CO), Metal Oxide. Another Conidtion is high environemntal conditions such as temperature or humidity. In such conditions, Resididents might begin to display signs such as distress or High Activity. 

By comparing and determining the realtionship between sensor readings and the activity level of the resident, we can learn which sensors/gases/conditions directly correlate to potentially dangerous conditions

# HYPOTHESIS/ASSUMPTIONS:
The Activity Level Directly corroleates with the wellbeing of the resident
High activity is a result of signs such as distress or medical episodes caused by unsafe living conditions

# PROJECT OBJECTIVE: 
Develop End-to-end machine learning pipeline that predicts the **Activity Level** (`Low` / `Moderate` / `High`) of elderly residents from environmental sensor and indoor-air-quality data, using **Logistic Regression/Decision Tree Classifier/Random Forest Classifier**. The Goal is to evaluate the perforamnce/importance of various environmnetal sensors in determining whetherelderly residents may be experiencing distress, medical episodes, or unsafe living conditions. 

The Information Gathered will then be used to develop predictive models and early warning systems to alert the residents to the issue and ensure their safety and wellbeing. These models/systems will be dependent on these sensors



## 1. Group Information

- **Members:** Brendan, Yee Sian, Nithish
- `logistic_regression.py` - Brendan
- 'random_forest.py' - Nithish
- 'dt_model.py' - Yee Sian

---

## 2. Project Structure

```
gas-monitoring-pipeline/
├── data/
│   ├── gas_monitoring.db        # SQLite database the pipeline reads from
│   └── cleaned_data.csv         # original imbalanced cleaned dataset (unchanged)
├── outputs/                     # confusion_matrix.png + metrics_summary.json
├── saved_model/                 # trained model + preprocessor (created on run)
├── logistic_regression.py       # THE pipeline — all the code is in this one file
├── config.yaml                  # all settings (change behaviour without editing code)
├── requirements.txt
├── Dockerfile                   # single, simple Docker setup
├── run.sh                       # runs the pipeline
└── README.md
```

Everything is in **one code file** (`logistic_regression.py`) organised into 6
clear sections: load config -> ingest (SQLite) -> preprocess -> train ->
evaluate -> save.

---

## 3. Config file

Parameters defined in **config.yaml** 

- database path / table name (`data.*`)
- resampling strategy (`resampling.method`: smote / random_over / random_under / none)
- Logistic Regression hyperparameters (`model.logistic_regression`: C, solver,
  class_weight, max_iter)
- feature scaling (`preprocessing.scale_features`)

---

## 5. Pipleline

**Step 0 — Data cleaning (`data_cleaning.py`):**
   - reads the raw `gas_monitoring` table from SQLite
   - removes duplicates and invalid/unrealistic values
   - imputes missing values (median / mode)
   - clips outliers using the IQR rule (the brief warns of contaminated data)
   - standardises category labels
   - writes the result to the `cleaned_data` table (+ `cleaned_data.csv`)

**Then each model**
1. **Ingest (SQLite):** reads the `cleaned_data` table.
2. **Prepare features:**
   - drops `Session ID` 
   - one-hot encodes the categorical columns (Time of Day, HVAC Operation Mode, Ambient Light Level)
   - standard-scales numeric features
   - balances classes with SMOTE 
3. **Train:** model .
4. **Evaluate:** accuracy, weighted & macro F1, a per-class report, and a
   confusion matrix.
5. **Save:** the trained model and the fitted preprocessor.

---

## 6. Evaluation

The data is **imbalanced** (Low Activity dominates) and this is a **health
early-warning** problem, so accuracy alone is misleading. We focus on
**weighted / macro F1** and the **per-class recall** (especially for the rare
High-Activity class), which the classification report and confusion matrix make
visible.
















