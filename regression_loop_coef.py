import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Define the path to your Excel file
path = 'Z:/ra/uhn/blad/Meeting - Aug 16th/regression.xlsx'
sheet_names = ['Clinical_PFT_Osc_qCT']  # Add your sheet names here

font_prop = FontProperties()
font_prop.set_family('Arial')
font_prop.set_size('18')

def plot_coefficients_or_importances(df, title, save_path):
    df = df.sort_values(by=df.columns[0], ascending=False)
    plt.figure(figsize=(14, 10))
    plt.barh(df.index, df.iloc[:, 0], color='black')
    plt.xlabel('Coefficient Values', fontproperties=font_prop, fontsize=20, fontweight='bold')
    plt.ylabel('Selected Features', fontproperties=font_prop, fontsize=20, fontweight='bold')    
    plt.title(title, fontproperties=font_prop, fontsize=20, fontweight='bold')
    plt.xticks(fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(save_path, format='tiff')
    plt.close()

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
        for j in range(i + 1, len(correlation_matrix.columns)):
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

    # Define a dictionary to store AUC values, selected features, and coefficients/feature importances
    auc_values = {}
    selected_features_dict = {}
    coef_dict = {}
    importance_dict = {}

    # Logistic Regression
    model_lr = LogisticRegression(penalty='l2', max_iter=100000, random_state=0, class_weight=weights)
    model_lr.fit(X_train, y_train)
    y_pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_values['Logistic Regression'] = auc_lr
    selected_features_dict['Logistic Regression'] = filtered_covariates
    coef_dict['Logistic Regression'] = model_lr.coef_[0]

    # Extra Trees Classifier
    extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight=weights)
    extra_trees.fit(X_train, y_train)
    sfm_extra_trees = SelectFromModel(extra_trees, prefit=True)
    y_pred_prob_extra_trees = extra_trees.predict_proba(X_test)[:, 1]
    fpr_extra_trees, tpr_extra_trees, _ = roc_curve(y_test, y_pred_prob_extra_trees)
    auc_extra_trees = auc(fpr_extra_trees, tpr_extra_trees)
    auc_values['Extra Trees'] = auc_extra_trees
    selected_features_dict['Extra Trees'] = X.columns[sfm_extra_trees.get_support()].tolist()
    importance_dict['Extra Trees'] = extra_trees.feature_importances_

    # Elastic Net
    elastic_net = ElasticNetCV(l1_ratio=0.5, cv=10, random_state=0)
    elastic_net.fit(X_train, y_train)
    y_pred_prob_elastic_net = elastic_net.predict(X_test)
    fpr_elastic_net, tpr_elastic_net, _ = roc_curve(y_test, y_pred_prob_elastic_net)
    auc_elastic_net = auc(fpr_elastic_net, tpr_elastic_net)
    auc_values['Elastic Net'] = auc_elastic_net
    selected_features_dict['Elastic Net'] = X.columns[elastic_net.coef_ != 0].tolist()
    coef_dict['Elastic Net'] = elastic_net.coef_

    # Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gb_classifier.fit(X_train, y_train)
    sfm_gb = SelectFromModel(gb_classifier, prefit=True)
    y_pred_prob_gb = gb_classifier.predict_proba(X_test)[:, 1]
    fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_prob_gb)
    auc_gb = auc(fpr_gb, tpr_gb)
    auc_values['Gradient Boosting'] = auc_gb
    selected_features_dict['Gradient Boosting'] = X.columns[sfm_gb.get_support()].tolist()
    importance_dict['Gradient Boosting'] = gb_classifier.feature_importances_

    # Lasso
    lasso = Lasso(alpha=0.1, random_state=0)
    lasso.fit(X_train, y_train)
    sfm_lasso = SelectFromModel(lasso, prefit=True)
    y_pred_prob_lasso = lasso.predict(X_test)
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_prob_lasso)
    auc_lasso = auc(fpr_lasso, tpr_lasso)
    auc_values['Lasso'] = auc_lasso
    selected_features_dict['Lasso'] = X.columns[sfm_lasso.get_support()].tolist()
    coef_dict['Lasso'] = lasso.coef_

    # Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=weights)
    rf_classifier.fit(X_train, y_train)
    sfm_rf = SelectFromModel(rf_classifier, prefit=True)
    y_pred_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_values['Random Forest'] = auc_rf
    selected_features_dict['Random Forest'] = X.columns[sfm_rf.get_support()].tolist()
    importance_dict['Random Forest'] = rf_classifier.feature_importances_

    # Identify the best and second-best models based on AUC
    sorted_models = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
    best_model = sorted_models[0][0]
    best_auc = sorted_models[0][1]
    second_best_model = sorted_models[1][0] if len(sorted_models) > 1 else None
    second_best_auc = sorted_models[1][1] if len(sorted_models) > 1 else None

    best_features = selected_features_dict[best_model]
    best_num_features = len(best_features)

    if best_num_features == 1 and second_best_model:
        print(f"\nBest Model based on AUC for sheet '{sheet_name}' selects only one feature.")
        print(f"Switching to the second-best model: {second_best_model} with AUC: {second_best_auc}")
        best_model = second_best_model
        best_auc = second_best_auc
        best_features = selected_features_dict[best_model]

    print(f"\nBest Model based on AUC for sheet '{sheet_name}': {best_model} with AUC: {best_auc}")
    print(f"Selected Features by the Best Model ({best_model}) for sheet '{sheet_name}':")
    print(best_features)

    # Create DataFrame for selected features for the best model
    selected_features_df = pd.DataFrame({
        'Feature': best_features
    }).set_index('Feature')

    # Create DataFrame for coefficient values or feature importances of the best model
    if best_model in coef_dict:
        coef_series = pd.Series(coef_dict[best_model], index=X.columns)
        coef_df = pd.DataFrame({
            'Feature': coef_series.index,
            'Coefficient': coef_series.values
        }).set_index('Feature')
        coef_df = coef_df.loc[best_features]

        print(f"\nCoefficient Values for Selected Features of the Best Model for sheet '{sheet_name}':")
        print(coef_df)

        # Plot the coefficients
        plot_title = f'Coefficients for Selected Features of ({sheet_name}) Data - {best_model} Model'
        plot_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/coefficient_plots/{sheet_name}_{best_model}_coef.tiff'
        plot_coefficients_or_importances(coef_df, plot_title, plot_path)
        
    elif best_model in importance_dict:
        importance_series = pd.Series(importance_dict[best_model], index=X.columns)
        importance_df = pd.DataFrame({
            'Feature': importance_series.index,
            'Importance': importance_series.values
        }).set_index('Feature')
        importance_df = importance_df.loc[best_features]

        print(f"\nFeature Importances for Selected Features of the Best Model for sheet '{sheet_name}':")
        print(importance_df)

        # Plot the feature importances
        plot_title = f'Coefficients for Selected Features of ({sheet_name}) Data - {best_model} Model'
        plot_path = f'Z:/ra/uhn/blad/Meeting - Aug 16th/figures/coefficient_plots/{sheet_name}_{best_model}_coef.tiff'
        plot_coefficients_or_importances(importance_df, plot_title, plot_path)
