# Data Analysis and Feature Selection Pipeline
This repository contains a comprehensive data analysis and feature selection pipeline implemented in Python. The code demonstrates various techniques for feature selection, model training, and evaluation using data from an Excel file. Below is a summary of the key components and functionalities of the code.

1. Dependencies
The code requires the following Python libraries:
- pandas
- matplotlib
- numpy
- statsmodels
- scikit-learn
- openpyxl
Install the required libraries using pip

2. Data File
The data used in this pipeline.
The file contains multiple sheets, each representing different datasets.

3. Functionality
- Data Preparation and Correlation Analysis
  - Reads data from specified sheets in the Excel file.
  - Drops columns with all NaN values.
  - Performs correlation analysis to identify and remove highly correlated features.
  - Calculates Variance Inflation Factor (VIF) to assess multicollinearity and filters out variables with high VIF.
- Feature Selection
  - Implements various feature selection methods including:
    - Logistic Regression
    - Extra Trees Classifier
    - Elastic Net
    - Gradient Boosting Classifier
    - Lasso
    - Random Forest Classifier
- Model Training and Evaluation
  - Trains models using the filtered features.
  - Evaluates models by plotting ROC curves and calculating the Area Under the Curve (AUC).
  - Saves ROC curve plots as TIFF files.
- Excel Output
  - Saves selected features for each model in separate Excel files.
  - Merges individual Excel files containing selected features into a single consolidated file.
    
4. Additional Analysis and Plots
- SHAP Analysis
Implements SHAP (SHapley Additive exPlanations) to interpret model predictions and understand feature contributions.
Generates SHAP summary plots with custom styling, including black and dotted feature name lines.

- Brier Score
Computes the Brier score for model predictions, a metric used to assess the accuracy of probabilistic predictions.

- Learning Curves
Plots learning curves to visualize model performance and how it changes with varying amounts of training data.

- Confusion Matrices
Generates confusion matrices to evaluate the performance of classification models by comparing predicted and actual labels.

- Regularization
Implements various regularization techniques, including Lasso and Elastic Net, to perform feature selection and prevent overfitting.

- Coefficient Plots
Creates plots of model coefficients to visualize the impact of each feature on the predictions.
Includes options to customize the plot appearance, such as switching axes and adjusting labels.

- Linear Regression P-values Calculation
Calculates p-values for coefficients in linear regression models to assess the statistical significance of each predictor.

5. Code Overview
- Data Preparation: Handles missing values, performs correlation analysis, and calculates VIF.
- Feature Selection: Uses various machine learning algorithms and regularization methods to select features.
- Model Training: Fits models and plots ROC curves to evaluate performance.
- Excel Management: Saves results in Excel format and merges files for easy access.
- SHAP Analysis: Analyzes feature importance and model predictions using SHAP values.
- Brier Score: Evaluates the quality of probabilistic predictions.
- Learning Curves: Visualizes model performance with respect to training data size.
- Confusion Matrices: Displays true vs. predicted classifications.
- Regularization: Applies Lasso and Elastic Net for feature selection and model regularization.
- Coefficient Plots: Provides visual insights into feature impact and model coefficients.
- Linear Regression P-values: Computes and reports the statistical significance of linear regression coefficients.

6. Usage
Update the path variable with the location of your Excel file.
Modify sheet_names to include the sheets you want to process.
Run the script to perform the analysis, feature selection, and model evaluation.
Check the selected_features and figures directories for output files.
