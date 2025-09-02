# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr

# # Function to calculate p-values for the correlation matrix
# def calculate_pvalues(df, method='pearson'):
#     df_cols = pd.DataFrame(columns=df.columns)
#     pvalues = df_cols.transpose().join(df_cols, how='outer')
#     for r in df.columns:
#         for c in df.columns:
#             if r == c:
#                 pvalues[r][c] = np.nan
#             else:
#                 if method == 'pearson':
#                     _, pvalues[r][c] = pearsonr(df[r], df[c])
#                 elif method == 'spearman':
#                     _, pvalues[r][c] = spearmanr(df[r], df[c])
#     return pvalues

# # Function to apply star notation to p-values
# def apply_star_notation(corr, pvalues):
#     annotations = corr.copy().astype(str)
#     for r in corr.index:
#         for c in corr.columns:
#             if pd.isnull(pvalues.at[r, c]):
#                 annotations.at[r, c] = ''
#             elif pvalues.at[r, c] < 0.001:
#                 annotations.at[r, c] = f'{corr.at[r, c]:.2f}***'
#             elif pvalues.at[r, c] < 0.01:
#                 annotations.at[r, c] = f'{corr.at[r, c]:.2f}**'
#             elif pvalues.at[r, c] < 0.05:
#                 annotations.at[r, c] = f'{corr.at[r, c]:.2f}*'
#             else:
#                 annotations.at[r, c] = f'{corr.at[r, c]:.2f}'
#     return annotations

# # Custom heatmap function with adjusted cell width and reordered labels
# def custom_heatmap(data, annotations, filename):
#     # Ensure exact matches for row labels
#     selected_col = {
#     'gaze_angle_x_mean': 'GazeX',
#     'gaze_angle_y_mean': 'GazeY',
#     'pose_Rx_mean': 'HeadX',
#     'pose_Ry_mean': 'HeadY',
#     'pose_Rz_mean': 'HeadZ',
#     'gaze_angle_x_std': 'GazeX',
#     'gaze_angle_y_std': 'GazeY',
#     'pose_Rx_std': 'HeadX',
#     'pose_Ry_std': 'HeadY',
#     'pose_Rz_std': 'HeadZ'

#     }

    

#     row_labels = [
#         selected_col[col] if col in selected_col else col.split('_')[0]
#         for col in data.index
#     ]
    

#     fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figsize as needed
#     ax = sns.heatmap(
#         data, annot=annotations, fmt='', cmap='coolwarm', vmin=-1, vmax=1,
#         xticklabels=['Open.', 'Consc.', 'Extro.', 'Agree.', 'Stab.'],  # Standard labels for personality traits
#         yticklabels=row_labels,  # Custom row labels based on the logic
#         cbar=False, annot_kws={"fontsize": 14}  # Adjust font size
#     )
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)  # Keep y-axis labels horizontal
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=14)  # Rotate x-axis labels
#     plt.tight_layout()  # Ensure everything fits within the figure area
#     plt.savefig(filename, format='jpeg')
#     plt.show()

# # Load the dataset
# data = pd.read_csv('attn_pers.csv')

# # Define personality trait columns
# personality_traits = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

# # Filter action unit columns
# action_units = data.filter(regex='_r|_angle|_R', axis=1)  # Ensure to select the columns ending with '_r'

# # Group by 'File' column and calculate the mean and standard deviation of action units
# grouped_means = data.groupby('File')[action_units.columns].mean().reset_index()
# grouped_stds = data.groupby('File')[action_units.columns].std().reset_index()

# # Extract the first row of personality traits for each video (assuming traits are constant per video)
# personality_data = data.groupby('File')[personality_traits].first().reset_index()

# # Merge the mean and standard deviation data with personality traits
# merged_data = pd.merge(grouped_means, grouped_stds, on='File', suffixes=('_mean', '_std'))
# merged_data = pd.merge(merged_data, personality_data, on='File')

# # Descriptive statistics
# print(merged_data.describe())

# # Pearson correlation analysis (excluding 'File' column)
# pearson_correlation = merged_data.drop(columns='File').corr(method='pearson')
# pearson_pvalues = calculate_pvalues(merged_data.drop(columns='File'), method='pearson')

# # Spearman correlation analysis (excluding 'File' column)
# spearman_correlation = merged_data.drop(columns='File').corr(method='spearman')
# spearman_pvalues = calculate_pvalues(merged_data.drop(columns='File'), method='spearman')

# # Define reordered column names based on the actual columns in correlation
# mean_correlation_cols = ['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']
# std_correlation_cols = [col for col in merged_data.columns if '_std' in col]

# # Access correlations using reordered column names
# print(action_units.columns)
# mean_pearson_correlation = pearson_correlation.loc[action_units.columns + '_mean', mean_correlation_cols]
# std_pearson_correlation = pearson_correlation.loc[action_units.columns + '_std', mean_correlation_cols]

# mean_spearman_correlation = spearman_correlation.loc[action_units.columns + '_mean', mean_correlation_cols]
# std_spearman_correlation = spearman_correlation.loc[action_units.columns + '_std', mean_correlation_cols]

# # Calculate p-values for reordered data
# mean_pearson_pvalues = pearson_pvalues.loc[mean_pearson_correlation.index, mean_correlation_cols]
# std_pearson_pvalues = pearson_pvalues.loc[std_pearson_correlation.index, mean_correlation_cols]

# mean_spearman_pvalues = spearman_pvalues.loc[mean_spearman_correlation.index, mean_correlation_cols]
# std_spearman_pvalues = spearman_pvalues.loc[std_spearman_correlation.index, mean_correlation_cols]

# # Apply star notation to reordered correlations
# mean_pearson_annotations = apply_star_notation(mean_pearson_correlation, mean_pearson_pvalues)
# std_pearson_annotations = apply_star_notation(std_pearson_correlation, std_pearson_pvalues)

# mean_spearman_annotations = apply_star_notation(mean_spearman_correlation, mean_spearman_pvalues)
# std_spearman_annotations = apply_star_notation(std_spearman_correlation, std_spearman_pvalues)

# # Heatmap of the Pearson correlation matrix for mean action units and personality traits
# custom_heatmap(mean_pearson_correlation, mean_pearson_annotations, 'mean_pearson_action_units_personality_correlation.jpg')

# # Heatmap of the Pearson correlation matrix for standard deviation action units and personality traits
# custom_heatmap(std_pearson_correlation, std_pearson_annotations, 'std_pearson_action_units_personality_correlation.jpg')

# # Heatmap of the Spearman correlation matrix for mean action units and personality traits
# custom_heatmap(mean_spearman_correlation, mean_spearman_annotations, 'mean_spearman_action_units_personality_correlation.jpg')

# # Heatmap of the Spearman correlation matrix for standard deviation action units and personality traits
# custom_heatmap(std_spearman_correlation, std_spearman_annotations, 'std_spearman_action_units_personality_correlation.jpg')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Function to calculate p-values
def calculate_pvalues(df, method='pearson'):
    df_cols = pd.DataFrame(columns=df.columns)
    pvalues = df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if r == c:
                pvalues[r][c] = np.nan
            else:
                if method == 'pearson':
                    _, pvalues[r][c] = pearsonr(df[r], df[c])
                elif method == 'spearman':
                    _, pvalues[r][c] = spearmanr(df[r], df[c])
    return pvalues

# Function to apply star notation to correlations
def apply_star_notation(corr, pvalues):
    annotations = corr.copy().astype(str)
    for r in corr.index:
        for c in corr.columns:
            if pd.isnull(pvalues.at[r, c]):
                annotations.at[r, c] = ''
            elif pvalues.at[r, c] < 0.001:
                annotations.at[r, c] = f'{corr.at[r, c]:.2f}***'
            elif pvalues.at[r, c] < 0.01:
                annotations.at[r, c] = f'{corr.at[r, c]:.2f}**'
            elif pvalues.at[r, c] < 0.05:
                annotations.at[r, c] = f'{corr.at[r, c]:.2f}*'
            else:
                annotations.at[r, c] = f'{corr.at[r, c]:.2f}'
    return annotations

# Custom heatmap function
def custom_heatmap(data, annotations, filename):
    # Mapping of AU and feature names
    selected_col = {
    'AU01_r_mean': 'Inner Brow Raiser (AU01)',
    'AU02_r_mean': 'Outer Brow Raiser (AU02)',
    'AU04_r_mean': 'Brow Lowerer (AU04)',
    'AU05_r_mean': 'Upper Lid Raiser (AU05)',
    'AU06_r_mean': 'Cheek Raiser (AU06)',
    'AU07_r_mean': 'Lid Tightener (AU07)',
    'AU09_r_mean': 'Nose Wrinkler (AU09)',
    'AU10_r_mean': 'Upper Lip Raiser (AU10)',
    'AU12_r_mean': 'Lip Corner Puller (AU12)',
    'AU14_r_mean': 'Dimpler (AU14)',
    'AU15_r_mean': 'Lip Corner Depressor (AU15)',
    'AU17_r_mean': 'Chin Raiser (AU17)',
    'AU20_r_mean': 'Lip Stretcher (AU20)',
    'AU23_r_mean': 'Lip Tightener (AU23)',
    'AU25_r_mean': 'Lips Part (AU25)',
    'AU26_r_mean': 'Jaw Drop (AU26)',
    'AU45_r_mean': 'Blink (AU45)',
    'AU01_r_std': 'Inner Brow Raiser (AU01)',
    'AU02_r_std': 'Outer Brow Raiser (AU02)',
    'AU04_r_std': 'Brow Lowerer (AU04)',
    'AU05_r_std': 'Upper Lid Raiser (AU05)',
    'AU06_r_std': 'Cheek Raiser (AU06)',
    'AU07_r_std': 'Lid Tightener (AU07)',
    'AU09_r_std': 'Nose Wrinkler (AU09)',
    'AU10_r_std': 'Upper Lip Raiser (AU10)',
    'AU12_r_std': 'Lip Corner Puller (AU12)',
    'AU14_r_std': 'Dimpler (AU14)',
    'AU15_r_std': 'Lip Corner Depressor (AU15)',
    'AU17_r_std': 'Chin Raiser (AU17)',
    'AU20_r_std': 'Lip Stretcher (AU20)',
    'AU23_r_std': 'Lip Tightener (AU23)',
    'AU25_r_std': 'Lips Part (AU25)',
    'AU26_r_std': 'Jaw Drop (AU26)',
    'AU45_r_std': 'Blink (AU45)',
    'gaze_angle_x_mean': 'GazeX',
    'gaze_angle_y_mean': 'GazeY',
    'pose_Rx_mean': 'HeadX',
    'pose_Ry_mean': 'HeadY',
    'pose_Rz_mean': 'HeadZ',
    'gaze_angle_x_std': 'GazeX',
    'gaze_angle_y_std': 'GazeY',
    'pose_Rx_std': 'HeadX',
    'pose_Ry_std': 'HeadY',
    'pose_Rz_std': 'HeadZ'
}


    row_labels = [selected_col.get(col, col.split('_')[0]) for col in data.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        data, annot=annotations, fmt='', cmap='coolwarm', vmin=-1, vmax=1,
        xticklabels=['Open.', 'Consc.', 'Extro.', 'Agree.', 'Stab.'],
        yticklabels=row_labels, cbar=False, annot_kws={"fontsize": 14}
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, format='jpeg')
    plt.show()

# Load your data
data = pd.read_csv('output_pers.csv')

# Define personality trait columns
personality_traits = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']

# Filter AU + head/gaze columns
action_units = data.filter(regex='_r|_angle|_R', axis=1)

# Group by File (video) and get mean & std for each
grouped_means = data.groupby('File')[action_units.columns].mean().reset_index()
grouped_stds = data.groupby('File')[action_units.columns].std().reset_index()
personality_data = data.groupby('File')[personality_traits].first().reset_index()

# Merge all
merged_data = pd.merge(grouped_means, grouped_stds, on='File', suffixes=('_mean', '_std'))
merged_data = pd.merge(merged_data, personality_data, on='File')

# Pearson and Spearman correlations
pearson_corr = merged_data.drop(columns='File').corr(method='pearson')
spearman_corr = merged_data.drop(columns='File').corr(method='spearman')
pearson_pval = calculate_pvalues(merged_data.drop(columns='File'), method='pearson')
spearman_pval = calculate_pvalues(merged_data.drop(columns='File'), method='spearman')

# Select personality trait order
trait_cols = ['openness', 'conscientiousness', 'extroversion', 'agreeableness', 'neuroticism']

# AU feature columns
mean_au_cols = [col for col in merged_data.columns if col.endswith('_mean') and col != 'File']
std_au_cols = [col for col in merged_data.columns if col.endswith('_std') and col != 'File']

# Subset correlations
mean_pearson_corr = pearson_corr.loc[mean_au_cols, trait_cols]
std_pearson_corr = pearson_corr.loc[std_au_cols, trait_cols]
mean_spearman_corr = spearman_corr.loc[mean_au_cols, trait_cols]
std_spearman_corr = spearman_corr.loc[std_au_cols, trait_cols]

# Subset p-values
mean_pearson_pval = pearson_pval.loc[mean_au_cols, trait_cols]
std_pearson_pval = pearson_pval.loc[std_au_cols, trait_cols]
mean_spearman_pval = spearman_pval.loc[mean_au_cols, trait_cols]
std_spearman_pval = spearman_pval.loc[std_au_cols, trait_cols]

# Apply stars
mean_pearson_annot = apply_star_notation(mean_pearson_corr, mean_pearson_pval)
std_pearson_annot = apply_star_notation(std_pearson_corr, std_pearson_pval)
mean_spearman_annot = apply_star_notation(mean_spearman_corr, mean_spearman_pval)
std_spearman_annot = apply_star_notation(std_spearman_corr, std_spearman_pval)

# Save heatmaps
custom_heatmap(mean_pearson_corr, mean_pearson_annot, 'mean_pearson_action_units_personality_correlation.jpg')
custom_heatmap(std_pearson_corr, std_pearson_annot, 'std_pearson_action_units_personality_correlation.jpg')
custom_heatmap(mean_spearman_corr, mean_spearman_annot, 'mean_spearman_action_units_personality_correlation.jpg')
custom_heatmap(std_spearman_corr, std_spearman_annot, 'std_spearman_action_units_personality_correlation.jpg')

