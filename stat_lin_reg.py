import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/regression.xlsx'
sheet_names = ['Clinical_PFT_Osc_qCT']  # Add your sheet names here

for sheet_name in sheet_names:
    # Read the specified columns from the Excel sheet
    reg_df = pd.read_excel(path, sheet_name=sheet_name)
    reg_df = reg_df.dropna(axis=1)

    dependent_var = 'blad_status'
    covariates = reg_df.columns.difference([dependent_var])

    # Correlation Analysis
    correlation_matrix = reg_df[[dependent_var] + list(covariates)].corr()

    # Find highly correlated pairs
    threshold = 0.9  # Adjust as needed
    highly_correlated_pairs = {}
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                pair_name = f"{colname_i} - {colname_j}"
                highly_correlated_pairs[pair_name] = (colname_i, colname_j)

    # Remove highly correlated pairs from covariates list
    for var1, var2 in highly_correlated_pairs.values():
        if var1 in covariates:
            covariates = covariates.difference([var2])
        if var2 in covariates:
            covariates = covariates.difference([var1])

    # Function to calculate VIF
    def calculate_vif(df, features):
        X = df[features]
        X = sm.add_constant(X)
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif

    # Calculate VIF for updated covariates list
    vif_df = calculate_vif(reg_df, list(covariates))

    # Drop covariates with VIF greater than 5
    high_vif_vars = vif_df[vif_df["VIF"] > 5]["Variable"].tolist()

    # Remove high VIF variables from covariates list
    filtered_covariates = [var for var in covariates if var not in high_vif_vars]

    # Recalculate VIF for filtered covariates
    filtered_vif_df = calculate_vif(reg_df, filtered_covariates)

    # Print VIF results
    print(f"\nFiltered Covariates and their VIF for sheet '{sheet_name}':")
    print(filtered_vif_df)

    # Prepare data for modeling
    X = reg_df[filtered_covariates]
    y = reg_df[dependent_var]
    
    # Linear Regression for hypothesis testing
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    print(f"\nOLS Regression Summary for sheet '{sheet_name}':")
    print(model.summary())
