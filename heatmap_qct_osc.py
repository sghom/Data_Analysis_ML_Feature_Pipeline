import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/regression.xlsx'
sheet_names = ['PFT_Osc_qCT']  # Add your sheet names here

# Set the threshold for correlation
threshold = 0.65

# Specify the path where you want to save the plots and tables
save_path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/heatmaps/'
save_path2 = 'Z:/ra/uhn/blad/Meeting - Aug 16th/'

# Excel writer object to write multiple sheets
excel_writer = pd.ExcelWriter(save_path2 + 'correlation_pairs.xlsx', engine='xlsxwriter')

for sheet_name in sheet_names:
    # Read the specified columns from the Excel sheet
    reg_df = pd.read_excel(path, sheet_name=sheet_name)
  
    # Extract clinical and qCT columns
    qct_data = ['AIA','AIECD','AMID','AOA','AWAF',
                'CLL','TAC','Pi10','TTV','TAV','TLV']

    osc_data = ['fvc','fev1','fev1_fvc','fef50','fef75','fef25_75','tlc','rv','rv_tlc','r5',
                'x5','r5_19','re_e','re_i','re_e_re_i','x_re_e_re_i_vt','xe_e','xe_i','ax',
                'xe_e_xe_i','x_xe_e_xe_i_vt']
    
    # Correlation Analysis (if needed)
    correlation_matrix = reg_df[qct_data + osc_data].corr()

    # Filter correlation matrix to include only qCT_data on x-axis and osc_data on y-axis
    clinical_corr_matrix = pd.DataFrame(index=qct_data, columns=osc_data, dtype=np.float64)

    for col_x in qct_data:
        for col_y in osc_data:
            if col_x in correlation_matrix.index and col_y in correlation_matrix.columns:
                clinical_corr_matrix.loc[col_x, col_y] = correlation_matrix.loc[col_x, col_y]
            else:
                clinical_corr_matrix.loc[col_x, col_y] = np.nan

    # # Remove rows and columns with all NaN values
    # clinical_corr_matrix = clinical_corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Define font properties
    font_prop = FontProperties()
    font_prop.set_family('Arial')  # Set font family
    font_prop.set_size(18)  # Set font size
    font_prop.set_weight('bold')  # Set to Bold

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(clinical_corr_matrix, dtype=bool))

    # Plot heatmap for the clinical correlation matrix if it's not empty
    if not clinical_corr_matrix.empty:
        plt.figure(figsize=(12, 14))
        ax = sns.heatmap(clinical_corr_matrix, mask=mask, annot=False, cmap='PuBu', vmin=-1, vmax=1,
                         linecolor='grey', linewidths=0.5)  # Adjust linewidths here
        plt.title(f'qcT vs PFT/Osc Correlation Matrix Heatmap', fontproperties=font_prop, fontsize=18, fontweight='bold')
        plt.xticks(fontproperties=font_prop)
        plt.yticks(fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Customize colorbar font properties and tick labels
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(labelsize=12, direction='out')  # Adjust font size and direction as needed
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), fontproperties=font_prop)  # Set font properties for tick labels
        cbar.set_label('', fontproperties=font_prop)  # Example label with font properties

        plt.savefig(save_path + 'heatmap_qct_osc_1.tif', format='tif', bbox_inches='tight')
        plt.show()

        # Filter the clinical_corr_matrix to keep only pairs with absolute correlation >= threshold
        filtered_corr_matrix = clinical_corr_matrix[(clinical_corr_matrix.abs() >= threshold) & (clinical_corr_matrix.abs() < 1)]

        # Create a mask for the upper triangle of the filtered matrix
        mask_filtered = np.triu(np.ones_like(filtered_corr_matrix, dtype=bool))

        # Plot heatmap for the filtered correlation matrix
        plt.figure(figsize=(12, 14))
        ax = sns.heatmap(filtered_corr_matrix, mask=mask_filtered, annot=False, cmap='PuBu', vmin=-1, vmax=1,
                         linecolor='grey', linewidths=0.5)  # Adjust linewidths here
        plt.title(f'qcT vs PFT/Osc Correlation Matrix (Threshold: {threshold})', fontproperties=font_prop,
                  fontsize=18, fontweight='bold')
        plt.xticks(fontproperties=font_prop)
        plt.yticks(fontproperties=font_prop)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Customize colorbar font properties and tick labels for filtered heatmap
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(labelsize=12, direction='out')  # Adjust font size and direction as needed
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), fontproperties=font_prop)  # Set font properties for tick labels
        cbar.set_label('', fontproperties=font_prop)  # Example label with font properties

        plt.savefig(save_path + 'heatmap_qct_osc_2.tif', format='tif', bbox_inches='tight')
        plt.show()

        # Create a table of pairs with correlation values
        corr_pairs = clinical_corr_matrix.stack().reset_index()
        corr_pairs.columns = ['qct_variable', 'osc_variable', 'correlation']
        corr_pairs = corr_pairs.sort_values(by='correlation', ascending=False)

        corr_pairs_filtered = filtered_corr_matrix.stack().reset_index()
        corr_pairs_filtered.columns = ['qct_variable', 'osc_variable', 'correlation']
        corr_pairs_filtered = corr_pairs_filtered.sort_values(by='correlation', ascending=False)

        # Save the tables to separate sheets in the Excel file
        corr_pairs.to_excel(excel_writer, sheet_name=f'{sheet_name}_all_pairs', index=False)
        corr_pairs_filtered.to_excel(excel_writer, sheet_name=f'{sheet_name}_filtered_pairs', index=False)
        
        # Save the correlation matrix to a new sheet with values rounded to 2 decimal places
        clinical_corr_matrix_rounded = clinical_corr_matrix.round(2)
        clinical_corr_matrix_rounded.transpose().to_excel(excel_writer, sheet_name=f'{sheet_name}_corr_matrix')
    else:
        print(f"No valid correlation matrix found for {sheet_name}. Skipping plotting.")

# Save and close the Excel writer
excel_writer.close()
