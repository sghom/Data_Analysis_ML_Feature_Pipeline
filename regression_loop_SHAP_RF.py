import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import shap

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

    # print(f"\nUpdated Covariates List and their VIF after removing high VIF variables for sheet '{sheet_name}':")
    # print(filtered_vif_df)

    X = reg_df[filtered_covariates]
    y = reg_df[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    odds = (len(np.array(y)) - sum(np.array(y))) / sum(np.array(y))
    weights = {0: odds, 1: 1}

    # Get coefficients and store in a dataframe
    coef_df = pd.DataFrame({'Variable': X.columns})

    # Example with RandomForestClassifier for feature selection
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=weights)
    rf_classifier.fit(X_train, y_train)

    # Select features based on importance
    sfm_rf = SelectFromModel(rf_classifier, prefit=True)
    selected_features_rf = X_train.columns[sfm_rf.get_support()]

    # Create a dictionary to hold selected features for each model
    selected_features_dict = {
        'Random Forest': list(selected_features_rf)
    }

    # Create DataFrame from dictionary
    selected_features_df = pd.DataFrame(selected_features_dict)

    # Print selected features for each model
    print(f"\nSelected Features by the Random Forest Model: '{sheet_name}':")
    print(selected_features_df)

    # Calculate ROC curve and AUC
    y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # Get the probability estimates for the positive class
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC for Random Forest Model on sheet '{sheet_name}': {roc_auc:.4f}")

    # SHAP Analysis
    explainer = shap.TreeExplainer(rf_classifier)
    shap_values = explainer.shap_values(X_test)

    # Convert shap_values to a dataframe to easily filter columns
    shap_values_df = pd.DataFrame(shap_values[:, :, 0], columns=X_test.columns)

    # Get the selected features
    selected_features = selected_features_df['Random Forest'].tolist()

    # Filter SHAP values to only include selected features
    filtered_shap_values = shap_values_df[selected_features]
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Customize x-axis tick labels
    plt.gca().tick_params(labelsize=14)
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), fontsize=16)
    plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontweight='bold', fontsize=16)

    # Set x-axis limits
    plt.xlim(-0.17868, 0.09088)  # Adjust these values as needed

    # Plot SHAP values for the selected features
    shap.summary_plot(filtered_shap_values.values, filtered_shap_values, feature_names=filtered_shap_values.columns, plot_type="violin", show=False)

    # Customize the feature name lines
    for line in plt.gca().get_lines():
        line.set_color('black')   # Set color to black
        line.set_linestyle('--')  # Set linestyle to dotted

    # Customize axes labels
    plt.xlabel('SHAP Value', fontsize=16, fontweight='bold')

    # Access the colorbar
    colorbar = plt.gcf().axes[1]  # Assuming the colorbar is the second axis

    # Modify the colorbar's y-label
    colorbar.set_ylabel('Feature Value', fontsize=16, fontweight='bold')

    # Modify the colorbar ticks and labels
    colorbar.set_yticks([-0.05,1.05])  # Set your desired tick positions
    colorbar.set_yticklabels(['Low','High'], fontweight='bold', fontsize=14)  # Set your desired tick labels

    # Save the plot to the specified path
    save_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/shap_RF/{sheet_name}_shap.tif'
    plt.savefig(save_path, format='tif', bbox_inches='tight')
    plt.close()