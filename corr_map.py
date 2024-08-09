import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from screeninfo import get_monitors
import numpy as np

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/regression.xlsx'

# Specify the sheet name (optional if there's only one sheet)
sheet_name = 'Sheet1'

# Read the specified columns from the Excel sheet
reg_df = pd.read_excel(path)

# Define qCT and clinical variables
qct_variables = [
    'Dysanapsis', 'AIA', 'AIECD', 'AMID', 'AOA', 'AWAF',
    'CLL', 'TAC', 'Pi10(leq20)', 'Pi10(all)', 'Total_Tissue_Volume_Entire',
    'Total_Air_Volume_Entire', 'Total_Volume'
]

clinical_variables = [
    'Age', 'Weight', 'Height', 'BMI', 'Sex', 'Sex_Match',
    'Donor_Type', 'Don_Cig_Use', 'Don_Pack_Year', 'Don_Age',
    'evlp', 'isch_time', 'clad_Status', 'A_Score', 'Recip_TLC',
    'Don_TLC', 'Extub_Time', 'TLC_Ratio', 'Time_to_Baseline'
]

pft_osc_1 = [
    'FVC', 'FVC_Normal', '%_FVC', 'FEV1', 'FEV1_Normal', '%_FEV1', 'FEV1_FVC',
    'FEV1_FVC_Normal', '%_FEV1_FVC', 'FEF25_75', 'FEF25_75_Normal', '%_FEF25_75',
    'RV_TLC', 'RV_TLC_Normal', '%_RV_TLC', '%_DLCO', 'R5', '%_R5', 'R5_Z', 'R5_19',
    'R5_19_Normal',
]

pft_osc_2 = [
    'X5', 'X5_Normal', '%_X5', 'X5_Z', 'AX', 'AX_Normal', '%_AX',
    'FRES', 'ReE', 'ReI', 'ReE-ReI', '(ReE-ReI)/VT', 'ARV', "ARV'", 'XeE', 'XeI', 'XeE-XeI',
    '(XeE-XeI)/VT', 'AXV', "AXV'"
]

# Combine variables to define the columns for correlation calculation
corr_cv_columns = ['blad_status'] + qct_variables + clinical_variables + pft_osc_1 + pft_osc_2

# Create the dataframe with the specified columns
corr_cv = reg_df[corr_cv_columns]

# Calculate Pearson correlation coefficients
pearson_corr = corr_cv.corr(method='pearson')

# Extract the sub-matrix for qCT vs clinical variables
qct_vs_clinical_corr = pearson_corr.loc[qct_variables, clinical_variables]

# Extract the sub-matrix for qCT vs PFT and Osc. (Batch 1)
qct_vs_pft_corr_1 = pearson_corr.loc[qct_variables, pft_osc_1]

# Extract the sub-matrix for qCT vs PFT and Osc. (Batch 2)
qct_vs_pft_corr_2 = pearson_corr.loc[qct_variables, pft_osc_2]

# Define font properties
font_prop = FontProperties()
font_prop.set_family('Arial')  # Set font family
font_prop.set_size('14')    # Set font size
font_prop.set_weight('bold')    # Set to Bold

# Get screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Set figure size to screen size (in inches)
dpi = plt.rcParams['figure.dpi']
fig_width = screen_width / dpi
fig_height = screen_height / dpi

# Function to plot heatmap with strong correlation values and annotations
def plot_heatmap(data, title, xticks, yticks, font_prop):
    # Filter strong correlations
    strong_corr_mask = np.abs(data) >= 0.5
    data_filtered = data.mask(~strong_corr_mask, np.nan)

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(data_filtered, annot=True, fmt=".2f", cmap='PuBu', vmin=-1, vmax=1, xticklabels=xticks, yticklabels=yticks)
    plt.title(title, fontproperties=font_prop, fontsize=18, fontweight='bold')
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16)

    # Make colorbar numbers bold
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(FontProperties(weight='bold'))

    plt.show()

# Plot heatmaps with customized colorbar font size
plot_heatmap(qct_vs_clinical_corr, 'Correlation Matrix Heatmap: qCT vs Clinical Variables', clinical_variables, qct_variables, font_prop)
plot_heatmap(qct_vs_pft_corr_1, 'Correlation Matrix Heatmap: qCT vs PFTs/Osc (Batch 1)', pft_osc_1, qct_variables, font_prop)
plot_heatmap(qct_vs_pft_corr_2, 'Correlation Matrix Heatmap: qCT vs PFTs/Osc (Batch 2)', pft_osc_2, qct_variables, font_prop)
