import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from datetime import datetime
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, accuracy_score, f1_score

#Import csv 
file_name = 'survey_responses_15_December_2025_shuffled.csv'
df = pd.read_csv(f'./{file_name}', keep_default_na=False)
df = df.drop(df.index[[0, 1]])

#Clean Data  
#drop null value
original_rows = len(df)  
df["Progress"] = pd.to_numeric(df["Progress"], errors="coerce")
df = df[df["Progress"] == 100]
remaining_rows = len(df)
dropped_rows = original_rows - remaining_rows
print(f"Original Responses: {original_rows}")
print(f"Rows dropped (progress < 100): {dropped_rows}")

#check DOB 
current_year = datetime.now().year
age = pd.to_numeric(df['Age'], errors='coerce')
birth_year = pd.to_numeric(df['Q79'], errors='coerce')

current_two_digit = current_year % 100  # e.g., 2025 -> 25
# Fix 2-digit birth years (e.g., 96 -> 1996)
birth_year_fixed = birth_year.copy()
# Identify 2-digit positive numbers
mask_two_digit = birth_year.between(0, 99, inclusive='both')
birth_year_fixed[mask_two_digit] = birth_year[mask_two_digit].apply(lambda x: 1900 + x if x > current_two_digit else 2000 + x)
calculated_age = current_year - birth_year_fixed
df = df[np.abs(calculated_age - age) <= 4]
final_rows = len(df)
dropped = remaining_rows - final_rows
print(f"Rows dropped (age mismatch): {dropped}")
print(f"Valid responses: {final_rows}")
df_clean = df.copy()

#Extract count for each expression 
curiosity_col = [col for col in df.columns if "Curious" in col]
categories = ["Robot 1", "Robot 2", "Both appear equally curious", "Neither appears curious"]
counts = {}
for col in curiosity_col:
    counts[col] = df_clean[col].value_counts()
counts_df = (pd.DataFrame(counts)
               .reindex(index=categories)    # ensure all categories show up
               .fillna(0)                    # missing values to be 0
               .astype(int))
print(counts_df)

valid_responses = len(df_clean)
column_totals = counts_df.sum()
for col, total in column_totals.items():
    if total != valid_responses:
        print(f"Mismatch in '{col}': total = {total}, expected = {valid_responses}")
        
#chi-squared: goodness of fit 
gof_results = []
for expr in counts_df:
    observed = counts_df[expr]
    chi2_stat, p_val = chisquare(observed)
    gof_results.append((expr, chi2_stat, p_val))
    print(f"{expr}: chi = {chi2_stat:.3f}, p = {p_val:.4f}"
          f"{'-> Reject H0' if p_val < 0.05 else '-> Fail to reject H0'}")

#post-hoc tests 
#standard residuals
print("\n=== Post-hoc Standardized Residuals per Expression ===")
for expr in counts_df.columns:
    observed = counts_df[expr]
    expected = np.repeat(observed.sum() / len(observed), len(observed))  # uniform expectation
    residuals = (observed - expected) / np.sqrt(expected)

    print(f"\n{expr}:")
    for cat, r in zip(counts_df.index, residuals):
        sig = "⭐" if (r) > 1.96 else ""
        print(f"  {cat:<12} r = {r:6.2f} {sig}")

    dominant = counts_df.index[np.argmax(residuals)]
    print(f"→ Dominant: {dominant}")
    
#label 
predicted_labels = counts_df.idxmax()
#print(predicted_labels)

comparison_info = { #DOP 12
    "Curious DOP": {    
        "variable": r"$\tau_{pause}$",
        "higher": "Robot 2",
        "lower": "Robot 1"
    },#DOT 12
    "Curious DOT": {    
        "variable": r"$\chi_{tilt}$",
        "higher": "Robot 2",
        "lower": "Robot 1"
    } ,#LAH 21
    "Curious LAH": {    
        "variable": r"$\theta_H$",
        "higher": "Robot 1",
        "lower": "Robot 2"
    }
    ,#LAV 21
    "Curious LAV": {    
        "variable": r"$\theta_V$",
        "higher": "Robot 1",
        "lower": "Robot 2"
    }
    ,#SOA 21
    "Curious SOA": {    
        "variable": r"$\alpha_{approach}$",
        "higher": "Robot 1",
        "lower": "Robot 2"
    }
    ,#SOT 21
    "Curious SOT": {    
        "variable": r"$\dot{\chi}_{tilt}$",
        "higher": "Robot 1",
        "lower": "Robot 2"
    }

}

expression_labels = [r"Pause Duration ($\tau_{pause}$)", r"Tilt Angle ($\chi_{tilt}$)", r"Horizontal Gaze ($\theta_H$)", r"Vertical Gaze ($\theta_V$)", r"Approach Acceleration ($\alpha_{approach}$)", r"Tilt Velocity ($\dot{\chi}_{tilt}$)"]
fig, axes = plt.subplots(3, 2, figsize=(10, 8))
axes = axes.flatten()

for i, expr in enumerate(counts_df.columns):
    info = comparison_info[expr]
    var = info["variable"] 
    colors = ["red" if cat == predicted_labels[expr] else "skyblue" for cat in counts_df.index]
    x_labels = []
    
    for cat in counts_df.index:
        if cat == info["higher"]:
            x_labels.append(f"Higher {var}")
        elif cat == info["lower"]:
            x_labels.append(f"Lower {var}")
        elif "both" in cat.lower():
            x_labels.append("Same")
        elif "neither" in cat.lower():
            x_labels.append("Neither")
        else:
            x_labels.append(cat)  # fallback, just in case

    axes[i].bar(x_labels, counts_df[expr], color=colors)
    #axes[i].bar(short_labels, counts_df[expr], color = colors)
    axes[i].set_title(expression_labels[i], fontsize=14, fontweight="bold")
    axes[i].tick_params(axis='x', rotation=45, labelsize = 15) 

plt.tight_layout()
img = "PartB_Curiosity.pdf"
plt.savefig(img)
plt.show()
print(f"Plot saved as {img}")
