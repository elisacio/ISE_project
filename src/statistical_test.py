from scipy.stats import ttest_rel, wilcoxon, shapiro
import numpy as np
import pandas as pd
import ast  # To convert string lists to actual lists


##########  Download & read data ##########
path = f'results/experiments.csv'
data = pd.read_csv(path)
data['CV_list(accuracy)'] = data['CV_list(accuracy)'].apply(ast.literal_eval)

# Flatten lists into a single NumPy array
all_accuracies = np.concatenate(data['CV_list(accuracy)'].values)

# Split into baseline and SVM accuracies
half = len(all_accuracies) // 2  # Find the middle index
baseline_accuracies = np.array(all_accuracies[:half])  # First half
svm_accuracies = np.array(all_accuracies[half:])  # Second half

# Check whether differences between paired values
# (SVM accuracy - Baseline accuracy) follow a normal distribution.
stat, p = shapiro(svm_accuracies - baseline_accuracies)
print(f"\nShapiro-Wilk test p-value: {p}")

if p > 0.05:
    # Perform paired t-test
    t_stat, p_ttest = ttest_rel(svm_accuracies, baseline_accuracies)
    print(f"\nThe Shapiro-Wilk test p_value is superior to 0.05, thus the differences between paired values\n(SVM accuracies-Baseline accuracies) follow a normal distribution. \nWe can use paired t-test.\n")
    print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
    p_value = p_ttest
else:
    # Perform Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(svm_accuracies, baseline_accuracies)
    print(
        f"\nThe Shapiro-Wilk test p_value is inferior to 0.05, thus the differences between paired values\n(SVM accuracies-Baseline accuracies) don't follow a normal distribution. \nWe can use Wilcoxon signed-rank test.\n")
    print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")
    p_value= p_wilcoxon

if p_value < 0.05 :
    print(f"The p-value is inferior to 0.05, the difference in accuracy is statistically significant.\n")
else :
    print(f"The p-value is superior to 0.05, the difference in accuracy is not statistically significant.\n")