import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNetCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error
from matplotlib.font_manager import FontProperties
import os
from openpyxl import Workbook, load_workbook

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/regression.xlsx'
sheet_names = ['Clinical','qCT','PFT','Osc','Clinical_PFT','Clinical_Osc',
               'Clinical_qCT','PFT_Osc','PFT_qCT','Osc_qCT','Clinical_PFT_Osc',
               'Clinical_PFT_qCT','Clinical_Osc_qCT','PFT_Osc_qCT',
               'Clinical_PFT_Osc_qCT']  # Add your sheet names here

# Store ROC curve data for each model and sheet
roc_data = {}

for sheet_name in sheet_names:
    # Read the specified columns from the Excel sheet
    reg_df = pd.read_excel(path, sheet_name=sheet_name)
    reg_df = reg_df.dropna(axis=1)

    dependent_var = 'blad_status'
    covariates = reg_df.columns.difference([dependent_var])

    # Correlation Analysis
    correlation_matrix = reg_df[[dependent_var,] + list(covariates)].corr()

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

    # # Add independent variable to covariates list
    # covariates = covariates.append(pd.Index([independent_var]))

    # Calculate VIF for updated covariates list
    vif_df = calculate_vif(reg_df, list(covariates))

    print(vif_df)

    # Drop covariates with VIF greater than 5
    high_vif_vars = vif_df[vif_df["VIF"] > 5]["Variable"].tolist()

    # Remove high VIF variables from covariates list
    filtered_covariates = [var for var in covariates if var not in high_vif_vars]

    # Recalculate VIF for filtered covariates
    filtered_vif_df = calculate_vif(reg_df, filtered_covariates)

    print(f"\nUpdated Covariates List and their VIF after removing high VIF variables for sheet '{sheet_name}':")
    print(filtered_vif_df)

    X = reg_df[filtered_covariates]
    y = reg_df[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    odds = (len(np.array(y)) - sum(np.array(y))) / sum(np.array(y))
    weights = {0: odds, 1: 1}

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(penalty='l2', max_iter=100000, random_state=0, class_weight=weights),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight=weights),
        'Elastic Net': ElasticNetCV(l1_ratio=0.5, cv=10, random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'Lasso': Lasso(alpha=0.1, random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0, class_weight=weights)
    }

    # Fit models and store AUC and MSE values
    auc_dict = {}
    mse_dict = {}
    feature_counts = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        if name in ['Elastic Net', 'Lasso']:
            y_pred_proba = model.predict(X_test)
            y_pred = model.predict(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        auc_dict[name] = roc_auc
        
        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_dict[name] = mse
        
        # Count the number of features selected by the model (if applicable)
        if hasattr(model, 'coef_'):
            feature_count = np.sum(model.coef_ != 0)
        elif hasattr(model, 'feature_importances_'):
            feature_count = np.sum(model.feature_importances_ > 0)
        else:
            feature_count = len(filtered_covariates)  # Use total number of features if model does not provide this information
        feature_counts[name] = feature_count

        # Store ROC curve data for each model
        roc_data[(sheet_name, name)] = (fpr, tpr, roc_auc, mse)
    
    # Find the best and second-best model
    sorted_auc = sorted(auc_dict.items(), key=lambda x: x[1], reverse=True)
    best_model_name, best_auc = sorted_auc[0]
    best_mse = mse_dict[best_model_name]
    best_feature_count = feature_counts[best_model_name]

    # Find the second-best model if the best model has only one feature
    if best_feature_count <= 1:
        if len(sorted_auc) > 1:
            second_best_model_name, second_best_auc = sorted_auc[1]
            second_best_mse = mse_dict[second_best_model_name]
            second_best_feature_count = feature_counts[second_best_model_name]
            print(f"Best model ({best_model_name}) has only {best_feature_count} feature(s). Skipping to second-best model ({second_best_model_name}).")
            best_model_name, best_auc = second_best_model_name, second_best_auc
            best_mse = second_best_mse
            best_feature_count = second_best_feature_count
        else:
            print("Only one model with enough features available.")
            continue

    # Print the AUC and MSE values
    print(f"\nAUC and MSE values for sheet '{sheet_name}':")
    for name, auc_value in auc_dict.items():
        print(f"{name}: AUC = {auc_value:.2f}, MSE = {mse_dict[name]:.2f}")
    print(f"Best model: {best_model_name} with AUC = {best_auc:.2f} and MSE = {best_mse:.2f}")

    # Plot the ROC curve for the best model
    plt.figure(figsize=(13, 10))
    best_model = models[best_model_name]

    if best_model_name in ['Elastic Net', 'Lasso']:
        y_pred_proba = best_model.predict(X_test)
    else:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Update label to include both AUC and MSE
    plt.plot(fpr, tpr, lw=4, color='blue', 
            label=f'{best_model_name} (AUC = {best_auc:.2f}, MSE = {best_mse:.2f})')

    font_prop = FontProperties()
    font_prop.set_family('Arial')
    font_prop.set_size('14')

    plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontproperties=font_prop, fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontproperties=font_prop, fontsize=18, fontweight='bold')
    plt.title(f'{sheet_name} - ROC Curve', fontproperties=font_prop, fontsize=18, fontweight='bold')
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.legend(loc="lower right", fontsize=18)

    # Save the plot to the specified path
    save_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/roc_curves_best/{sheet_name}_Best_ROC_curve.tif'
    plt.savefig(save_path, format='tif', bbox_inches='tight')
    
    # # Display the plot
    # plt.show()

# Prompt user to select a sheet and model for generating ROC curve
print("\nAvailable sheets:")
for i, sheet in enumerate(sheet_names):
    print(f"{i+1}. {sheet}")

sheet_choice = int(input("Enter the number of the sheet you want to analyze: ")) - 1
selected_sheet = sheet_names[sheet_choice]

print("\nAvailable models:")
available_models = [model_name for sheet, model_name in roc_data.keys() if sheet == selected_sheet]
for i, model in enumerate(available_models):
    print(f"{i+1}. {model}")

model_choice = int(input("Enter the number of the model you want to analyze: ")) - 1
selected_model = available_models[model_choice]

# Retrieve ROC curve data
fpr, tpr, roc_auc, mse = roc_data[(selected_sheet, selected_model)]

# Plot the ROC curve for the selected model and sheet
plt.figure(figsize=(13, 10))
plt.plot(fpr, tpr, lw=4, color='blue', label=f'{selected_model} (AUC = {roc_auc:.2f}, MSE = {mse:.2f})')

font_prop = FontProperties()
font_prop.set_family('Arial')
font_prop.set_size('14')

plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate', fontproperties=font_prop, fontsize=18, fontweight='bold')
plt.ylabel('True Positive Rate', fontproperties=font_prop, fontsize=18, fontweight='bold')
plt.title(f'{selected_sheet} - ROC Curve for {selected_model}', fontproperties=font_prop, fontsize=18, fontweight='bold')
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(loc="lower right", fontsize=18)

# Save the plot to the specified path
save_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/roc_curves_best/{selected_sheet}_{selected_model}_ROC_curve.tif'
plt.savefig(save_path, format='tif', bbox_inches='tight')