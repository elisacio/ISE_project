from scipy.stats import ttest_rel, wilcoxon, shapiro
import numpy as np
import pandas as pd
import ast  # To convert string lists to actual lists

##########  Download & read data ##########
path = f'results/experiments.csv'
data = pd.read_csv(path)
data['CV_list(accuracy)'] = data['CV_list(accuracy)'].apply(ast.literal_eval)
data['CV_list(precision)'] = data['CV_list(precision)'].apply(ast.literal_eval)
data['CV_list(recall)'] = data['CV_list(recall)'].apply(ast.literal_eval)
data['CV_list(F1)'] = data['CV_list(F1)'].apply(ast.literal_eval)

# Flatten lists into a single NumPy array
all_accuracies = np.concatenate(data['CV_list(accuracy)'].values)
all_precisions = np.concatenate(data['CV_list(precision)'].values)
all_recalls = np.concatenate(data['CV_list(recall)'].values)
all_f1 = np.concatenate(data['CV_list(F1)'].values)

# Split into baseline and SVM
half = len(all_accuracies) // 2  # Find the middle index

baseline_accuracies = np.array(all_accuracies[:half])  # First half
svm_accuracies = np.array(all_accuracies[half:])  # Second half

baseline_precisions = np.array(all_precisions[:half])  # First half
svm_precisions = np.array(all_precisions[half:])  # Second half

baseline_recalls = np.array(all_recalls[:half])  # First half
svm_recalls = np.array(all_recalls[half:])  # Second half

baseline_f1 = np.array(all_f1[:half])  # First half
svm_f1 = np.array(all_f1[half:])  # Second half

### Accuracy
# Check whether differences between paired values
# (SVM accuracy - Baseline accuracy) follow a normal distribution.

print("\n-------------------------------------------------------------------------------------\n")
print(f"\n              STATISTICAL TEST RESULTS : ACCURACY\n")
stat, p = shapiro(svm_accuracies - baseline_accuracies)
print(f"\nShapiro-Wilk test p-value: {p}")

if p > 0.05:
    # Perform paired t-test
    t_stat, p_ttest = ttest_rel(svm_accuracies, baseline_accuracies)
    print(f"\nThe Shapiro-Wilk test p_value is superior to 0.05, thus the differences between \npaired values follow a normal distribution. \nWe can use paired t-test.\n")
    print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
    p_value = p_ttest
else:
    # Perform Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(svm_accuracies, baseline_accuracies)
    print(f"\nThe Shapiro-Wilk test p_value is inferior to 0.05, thus the differences between \npaired values don't follow a normal distribution. \nWe can use Wilcoxon signed-rank test.\n")
    print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")
    p_value= p_wilcoxon

if p_value < 0.05 :
    print(f"The p-value is inferior to 0.05, the difference in accuracy is statistically significant.\n")
else :
    print(f"The p-value is superior to 0.05, the difference in accuracy is not statistically significant.\n")

### Precision
# Check whether differences between paired values follow a normal distribution.

print("\n-------------------------------------------------------------------------------------\n")
print(f"\n              STATISTICAL TEST RESULTS : PRECISION\n")
stat, p = shapiro(svm_precisions - baseline_precisions)
print(f"\nShapiro-Wilk test p-value: {p}")

if p > 0.05:
    # Perform paired t-test
    t_stat, p_ttest = ttest_rel(svm_precisions, baseline_precisions)
    print(f"\nThe Shapiro-Wilk test p_value is superior to 0.05, thus the differences between \npaired values follow a normal distribution. \nWe can use paired t-test.\n")
    print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
    p_value = p_ttest
else:
    # Perform Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(svm_precisions, baseline_precisions)
    print(f"\nThe Shapiro-Wilk test p_value is inferior to 0.05, thus the differences between \npaired values don't follow a normal distribution. \nWe can use Wilcoxon signed-rank test.\n")
    print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")
    p_value= p_wilcoxon

if p_value < 0.05 :
    print(f"The p-value is inferior to 0.05, the difference in precision is statistically significant.\n")
else :
    print(f"The p-value is superior to 0.05, the difference in precision is not statistically significant.\n")

### Recall
# Check whether differences between paired values follow a normal distribution.

print("\n-------------------------------------------------------------------------------------\n")
print(f"\n              STATISTICAL TEST RESULTS : RECALL\n")
stat, p = shapiro(svm_recalls - baseline_recalls)
print(f"\nShapiro-Wilk test p-value: {p}")

if p > 0.05:
    # Perform paired t-test
    t_stat, p_ttest = ttest_rel(svm_recalls, baseline_recalls)
    print(f"\nThe Shapiro-Wilk test p_value is superior to 0.05, thus the differences between \npaired values follow a normal distribution. \nWe can use paired t-test.\n")
    print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
    p_value = p_ttest
else:
    # Perform Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(svm_recalls, baseline_recalls)
    print(
        f"\nThe Shapiro-Wilk test p_value is inferior to 0.05, thus the differences between \npaired values don't follow a normal distribution. \nWe can use Wilcoxon signed-rank test.\n")
    print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")
    p_value = p_wilcoxon

if p_value < 0.05:
    print(f"The p-value is inferior to 0.05, the difference in recall is statistically significant.\n")
else:
    print(f"The p-value is superior to 0.05, the difference in recall is not statistically significant.\n")

### Recall
# Check whether differences between paired values follow a normal distribution.

print("\n-------------------------------------------------------------------------------------\n")
print(f"\n              STATISTICAL TEST RESULTS : F1-score\n")
stat, p = shapiro(svm_f1 - baseline_f1)
print(f"\nShapiro-Wilk test p-value: {p}")

if p > 0.05:
    # Perform paired t-test
    t_stat, p_ttest = ttest_rel(svm_f1, baseline_f1)
    print(f"\nThe Shapiro-Wilk test p_value is superior to 0.05, thus the differences between \npaired values follow a normal distribution. \nWe can use paired t-test.\n")
    print(f"\nPaired t-test:\n     t-statistic = {t_stat}\n     p-value = {p_ttest}\n")
    p_value = p_ttest
else:
    # Perform Wilcoxon signed-rank test
    w_stat, p_wilcoxon = wilcoxon(svm_f1, baseline_f1)
    print(f"\nThe Shapiro-Wilk test p_value is inferior to 0.05, thus the differences between \npaired values don't follow a normal distribution. \nWe can use Wilcoxon signed-rank test.\n")
    print(f"Wilcoxon signed-rank test:\n     W-statistic = {w_stat}\n     p-value = {p_wilcoxon}\n")
    p_value = p_wilcoxon

if p_value < 0.05:
    print(f"The p-value is inferior to 0.05, the difference in F1-score is statistically significant.\n")
else:
    print(f"The p-value is superior to 0.05, the difference in F1-score is not statistically significant.\n")

print("\n-------------------------------------------------------------------------------------\n")