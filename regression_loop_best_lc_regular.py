import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNetCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_curve, auc, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.font_manager import FontProperties
import os
from openpyxl import Workbook, load_workbook

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/regression.xlsx'
sheet_names = ['Clinical_PFT_Osc_qCT']  # Add your sheet names here

# Store ROC curve data for each model and sheet
roc_data = {}
conf_matrix_data = {}

# Define font properties
font_prop = FontProperties()
font_prop.set_family('Arial')
font_prop.set_size('18')

# Initialize list to store metrics
metrics_list = []

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

    print(f"\nUpdated Covariates List and their VIF after removing high VIF variables for sheet '{sheet_name}':")
    print(filtered_vif_df)

    X = reg_df[filtered_covariates]
    y = reg_df[dependent_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    odds = (len(np.array(y)) - sum(np.array(y))) / sum(np.array(y))
    weights = {0: odds, 1: 1}

    # Define models with tree constraints
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=0, class_weight=weights),
        # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, min_samples_split=5, min_samples_leaf=2, random_state=0),
        # 'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=0, class_weight=weights),
        # 'Elastic Net': ElasticNetCV(l1_ratio=0.5, cv=10, random_state=0),
        # 'Lasso': Lasso(alpha=0.1, random_state=0),
    }

    # Fit models and store AUC and MSE values
    auc_dict = {}
    mse_dict = {}
    feature_counts = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Predict probabilities or binary class labels based on model type
        if name in ['Elastic Net', 'Lasso']:
            y_pred_proba = model.predict(X_test)  # Continuous predictions
            y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary predictions
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
            y_pred = (y_pred_proba > 0.5).astype(int)  # Convert to binary predictions
        
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
        
        # Store confusion matrix data
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_data[(sheet_name, name)] = conf_matrix
    
    # Find the best and second-best model
    sorted_auc = sorted(auc_dict.items(), key=lambda x: x[1], reverse=True)
    best_model_name, best_auc = sorted_auc[0]
    best_mse = mse_dict[best_model_name]
    best_feature_count = feature_counts[best_model_name]

    # Initialize second-best model metrics
    second_best_model_name = None
    second_best_auc = None
    second_best_mse = None
    second_best_feature_count = None
    
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
    else:
        # If the best model is valid, set the second-best model
        if len(sorted_auc) > 1:
            second_best_model_name, second_best_auc = sorted_auc[1]
            second_best_mse = mse_dict[second_best_model_name]
            second_best_feature_count = feature_counts[second_best_model_name]

    # Compute learning curves for the best model
    model = models[best_model_name]
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy')
    
    # Calculate mean and std for plotting
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    print(auc_dict)

    # Plot learning curves
    plt.figure(figsize=(13, 10))
    plt.plot(train_sizes, train_mean, 'o-', lw = 3, color='black', label='Training Score')
    plt.plot(train_sizes, valid_mean, 'o-', lw = 3, color='red', label='Validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='black')
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color='red')
    plt.title(f'Learning Curves for {sheet_name} Data - {best_model_name} Model', fontproperties=font_prop, fontsize=20, fontweight='bold')
    plt.xlabel('Training Set Size', fontproperties=font_prop, fontsize=20, fontweight='bold')
    plt.ylabel('Score', fontproperties=font_prop, fontsize=20, fontweight='bold')
    plt.xticks(fontproperties=font_prop, fontsize=18)
    plt.yticks(fontproperties=font_prop, fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(True)
    
    # Save the plot
    plt_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/learning_curves/{sheet_name}_{best_model_name}_lc.tiff'
    plt.savefig(plt_path, format='tiff', bbox_inches='tight')
    plt.close()
