import pandas as pd 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import chi2_contingency, norm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from wordcloud import WordCloud
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.multitest as smm
import re
from collections import Counter


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

#Data Analysis 
#Word cloud 
text_cols = ['Text Entry\xa0_1', 'Text Entry\xa0_2', 'Text Entry\xa0_3', 'Text Entry\xa0_4']
titles = ['Expression A', 'Expression B', 'Expression C', 'Expression D']
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

# Generate one word cloud per column
"""for i, col in enumerate(text_cols):
    text_entry = " ".join(df_clean[col].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, font_path='C:\\Windows\\Fonts\\times.ttf').generate(text_entry)  
    # Display on subplot
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(titles[i], fontsize=16, fontweight='bold')
    axes[i].axis('off')
# Adjust layout neatly
plt.tight_layout()
plt.show()
plt.savefig ("Expression_WordCloud.png")"""
# To save the word cloud as an image file:
# wordcloud.to_file("Expression_WordCloud.png")

"""for i, col in enumerate(text_cols):
    text = " ".join(df_clean[col].dropna().astype(str)).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    counts = Counter(words).most_common(3)
    words_, values = zip(*counts)
    axes[i].bar(words_, values)
    axes[i].set_title(titles[i], fontweight='bold')
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
plt.savefig ("Expression_BarGraphs.png")"""

replacements = {
    'Confidence': 'Confident',
    'Hesitation': 'Hesitant',
    'Fear': 'Fearful',
    'Curiosity': 'Curious'
}

df_new = df_clean.replace(replacements, inplace=True) 

#Extract count for each expression 
expr_cols = ["Expression A: choice",
             "Expression B: choice",
             "Expression C: choice",   
             "Expression D: choice"]


#categories = ["Confidence", "Hesitation", "Fear", "Curiosity", "None", "Other"]
categories = ["Confident", "Curious", "Hesitant", "Fearful", "None", "Other"]
counts = {}
for col in expr_cols:
    counts[col] = df_clean[col].value_counts()
counts_df = (pd.DataFrame(counts)
               .reindex(index=categories)    # ensure all categories show up
               .fillna(0)                    # missing values to be 0
               .astype(int))
#check 
valid_responses = len(df_clean)
column_totals = counts_df.sum()
for col, total in column_totals.items():
    if total != valid_responses:
        print(f"Mismatch in '{col}': total = {total}, expected = {valid_responses}")
print(counts_df)

#chi-squared: goodness of fit 
gof_results = []
for expr in counts_df:
    observed = counts_df[expr]
    chi2_stat, p_val = chisquare(observed)
    W = np.sqrt(chi2_stat / observed.sum())
    gof_results.append((expr, chi2_stat, p_val))
    print(f"{expr}: chi = {chi2_stat:.3f}, p = {p_val:.4f},W = {W:.3f}"
          f"{'-> Reject H0' if p_val < 0.05 else '-> Fail to reject H0'}")
#print (gof_results)

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
    print(f"→ Dominant emotion (highest positive residual): {dominant}")

"""#pariwise z-test
print("\n=== Pairwise Proportion Comparisons per Expression ===")
for expr in counts_df.columns:
    print(f"\n{expr}:")
    observed = counts_df[expr]
    total = observed.sum()
    results = []
    
    # pairwise combinations of categories
    for cat1, cat2 in combinations(counts_df.index, 2):
        count = np.array([observed[cat1], observed[cat2]])
        nobs = np.array([total, total])
        z, p = proportions_ztest(count, nobs)
        results.append((cat1, cat2, z, p))
    
    # Multiple-comparison correction (Bonferroni)
    p_vals = [r[3] for r in results]
    _, p_adj, _, _ = smm.multipletests(p_vals, method='bonferroni')
    
    # Print only significant comparisons
    for (cat1, cat2, z, p), padj in zip(results, p_adj):
        if padj < 0.05:
            print(f"  {cat1} vs {cat2}: z = {z:.2f}, p_adj = {padj:.4f} ⭐")"""
#test of independence
chi2, p, dof, expected = chi2_contingency(counts_df)
print("\n=== Chi-squared Test of Independence ===")
print("Chi-squared statistic:", chi2)
print("Degrees of freedom:", dof)
print("p-value:", p)
if p < 0.05:
    print("Reject H₀ → Distribution of mental state depends on the expression (A–D).")
else:
    print("Fail to reject H₀ → No evidence that mental state differ across expressions.")


#bar graph 
plt.rcParams.update({
    "pdf.fonttype": 42,  # editable text in Illustrator
    "ps.fonttype": 42
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
predicted_labels = counts_df.idxmax()
expression_labels = ["Expression A", "Expression B", "Expression C", "Expression D"]
for i, expr in enumerate(counts_df.columns):
    colors = [
        "red" if cat == predicted_labels[expr] else "skyblue"
        for cat in counts_df.index
    ]
    axes[i].bar(counts_df.index, counts_df[expr], color=colors)
    # Clean title
    axes[i].set_title(expression_labels[i], fontsize=14, fontweight="bold")
    # Increase x-axis label font size
    axes[i].tick_params(axis="x", labelsize=12, rotation=45)
    # Optional: slightly increase y-axis numbers too
    axes[i].tick_params(axis="y", labelsize=11)
plt.tight_layout()

img = "PartA_BarChart_Shuffled.pdf"
plt.savefig(img, bbox_inches="tight")
plt.show()

print(f"Plot saved as {img}")
