

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")            # save plots to file (no GUI needed)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

def load_config(path="config.yaml"):                         # load data/variables from config file
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)\
    
config = load_config("config.yaml")

os.makedirs("outputs", exist_ok=True)

# 1. Load the RAW table from SQLite
conn = sqlite3.connect(config['data']['db_path'])
df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
print(f"Loaded raw data: {df.shape}")

# ---------------- Drop Duplicates ----------------
new_df = df.drop_duplicates()
print(f"After dropping duplicates: {new_df.shape}")

# ---------------- Remove Invalid / Unrealistic Values ----------------
# Done before imputation so erroneous values don't distort the medians.
valid_data = (
    (new_df["Temperature"] >= 0) & (new_df["Temperature"] <= 50) &
    (new_df["Humidity"].isna() | ((new_df["Humidity"] >= 0) & (new_df["Humidity"] <= 100))) &
    (new_df["MetalOxideSensor_Unit2"].isna() | (new_df["MetalOxideSensor_Unit2"] >= 0)) &
    (new_df["CO_GasSensor"].isna() | (new_df["CO_GasSensor"] >= 0))
)
new_df = new_df[valid_data]

# ---------------- Median Imputation ----------------
# replace null values with the median value
for col in ["Humidity", "MetalOxideSensor_Unit2", "CO_GasSensor"]:
    new_df[col] = new_df[col].fillna(new_df[col].median())
new_df["Ambient Light Level"] = new_df["Ambient Light Level"].fillna("unknown")

# ---------------- Remove Outliers (IQR) ----------------
cols_to_clean = ["Temperature", "Humidity", "CO2_InfraredSensor", "CO2_ElectroChemicalSensor",
                 "MetalOxideSensor_Unit1", "MetalOxideSensor_Unit2", "MetalOxideSensor_Unit3",
                 "MetalOxideSensor_Unit4", "CO_GasSensor"]
print("Before removing outliers:", new_df.shape)
for col in cols_to_clean:
    Q1, Q3 = new_df[col].quantile(0.25), new_df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    new_df = new_df[(new_df[col].isna()) | ((new_df[col] >= low) & (new_df[col] <= high))]
print("After removing outliers:", new_df.shape)

# Boxplot after outlier removal (saved for the EDA write-up)
plt.figure(figsize=(14, 6))
sns.boxplot(data=new_df[cols_to_clean])
plt.title("Boxplot of Numerical Sensor Columns After Removing Outliers")
plt.xlabel("Sensor Columns"); plt.ylabel("Sensor Values")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("outputs/boxplot_after_outliers.png", dpi=120); plt.close()

# ---------------- Standardise Categories ----------------
new_df["Activity Level"] = new_df["Activity Level"].replace({
    "LowActivity": "Low Activity", "Low_Activity": "Low Activity",
    "ModerateActivity": "Moderate Activity"})
new_df["HVAC Operation Mode"] = new_df["HVAC Operation Mode"].apply(lambda x: str(x).lower())
for col in ["Time of Day", "HVAC Operation Mode", "Ambient Light Level", "Activity Level"]:
    new_df[col] = new_df[col].astype(str).str.strip()

# Correlation heatmap (saved for the EDA write-up)
corr_matrix = new_df[cols_to_clean].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Sensor Features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=120); plt.close()

# ---------------- Save the cleaned dataset ----------------
new_df.to_csv(config['data']['csv_path'], index=False)

# Write cleaned data back to SQLite as `cleaned_data` \
new_df.to_sql("cleaned_data", conn, if_exists="replace", index=False)
conn.close()
print(f"Cleaned data saved: {new_df.shape} -> data/cleaned_data.csv and table 'cleaned_data'")
