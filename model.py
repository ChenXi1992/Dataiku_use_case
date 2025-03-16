# Import necessary libraries
import numpy as np  # For numerical operations
import xgboost as xgb  # XGBoost classifier
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import StratifiedKFold, GridSearchCV  # Cross-validation and hyperparameter tuning

def logistic_regression_model(X_train, y_train):
    """
    Train and evaluate a Logistic Regression model using GridSearchCV.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training features.
    y_train (pd.Series or np.ndarray): Training labels.
    
    Returns:
    best_model (LogisticRegression): Best trained logistic regression model.
    """
    
    print("Training Logistic Regression model...")

    # Define hyperparameter grid for tuning
    param_grid = {
        'C': [1],  # Regularization strength
        'solver': ['liblinear'],  # Suitable for small datasets with L1/L2 penalty
        'penalty': ['l1', 'l2']  # Regularization type
    }

    # Initialize the logistic regression model
    base_model = LogisticRegression(max_iter=1000, random_state=42)

    # Perform grid search with stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Retrieve the best model and hyperparameters
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}") 

    return best_model


def xgboost_model(X_train, y_train):
    """
    Train and evaluate an XGBoost model using GridSearchCV.
    
    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training features.
    y_train (pd.Series or np.ndarray): Training labels.
    
    Returns:
    best_model (xgb.XGBClassifier): Best trained XGBoost classifier.
    """
    
    print("Training XGBoost model...")

    # Define hyperparameter grid for tuning
    param_grid = {
        'max_depth': [3],  # Depth of trees
        'learning_rate': [0.01, 0.1],  # Step size shrinkage to prevent overfitting
        'min_child_weight': [1, 3]  # Minimum sum of instance weight needed in a child node
    }

    # Initialize the XGBoost classifier
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',  # Logistic regression for binary classification
        eval_metric='auc',  # Evaluation metric: Area Under Curve
        random_state=42
    )

    # Perform grid search with stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Retrieve the best model and hyperparameters
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    return best_model
