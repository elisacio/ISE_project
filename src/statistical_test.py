from scipy.stats import ttest_rel, wilcoxon
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

# Perform paired t-test
t_stat, p_ttest = ttest_rel(svm_accuracies, baseline_accuracies)

# Perform Wilcoxon signed-rank test
w_stat, p_wilcoxon = wilcoxon(svm_accuracies, baseline_accuracies)

# Print results
print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")

if p_ttest < 0.05 and p_wilcoxon < 0.05:
    print(f"The p-value is inferior to 0.05, the difference in accuracy is statistically significant.\n")
else :
    print(f"The p-value is superior to 0.05, the difference in accuracy is not statistically significant.\n")