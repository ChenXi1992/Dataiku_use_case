# Import necessary libraries
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualizations
from sklearn.metrics import confusion_matrix, classification_report  # For evaluating classification models
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def plot_categorical_columns(df_train):
    """
    Plots bar charts for categorical columns in a given DataFrame.

    Parameters:
    df_train (pd.DataFrame): The input DataFrame containing categorical columns.

    Returns:
    None
    """

    # Identify categorical columns (object dtype)
    categorical_columns = df_train.select_dtypes(include=["object"]).columns
    
    # Set up the plotting grid
    num_cols = len(categorical_columns)
    fig, axes = plt.subplots(nrows=(num_cols // 3) + 1, ncols=3, figsize=(15, num_cols * 1.5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Generate bar plots for each categorical column
    for i, col in enumerate(categorical_columns):
        sns.countplot(y=df_train[col], ax=axes[i], order=df_train[col].value_counts().index)  # Bar plot
        axes[i].set_title(col)  # Set title as column name
        axes[i].set_ylabel("")  # Remove y-axis label
        axes[i].set_xlabel("Count")  # Set x-axis label

    # Hide any unused subplots (if the number of categorical columns is not a multiple of 3)
    for j in range(i + 1, len(axes)): 
        fig.delaxes(axes[j])  # Remove unused axes
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the plots


def plot_custom_confusion_matrix(y_true, y_pred, title):
    """
    Plots a custom confusion matrix with percentage values overlaid.

    Parameters:
    y_true (array-like): Ground truth (actual) labels.
    y_pred (array-like): Predicted labels.
    title (str): Title of the plot.

    Returns:
    None
    """

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert confusion matrix values to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  

    # Define a custom color map: Green for correct, Red for incorrect
    colors = np.array([["#76c776", "#f08080"],  # True Negative (green) | False Positive (red)
                       ["#f08080", "#76c776"]]) # False Negative (red) | True Positive (green)

    # Create a figure for the heatmap
    plt.figure(figsize=(5, 4))

    # Plot heatmap with custom colors
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", cbar=False,
                     xticklabels=["<50K", ">50K"], yticklabels=["<50K", ">50K"],
                     linewidths=1, linecolor='black', annot_kws={"size": 14})

    # Overlay percentage values on the heatmap
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.75, f"{cm_percent[i, j]:.1f}%", 
                     horizontalalignment='center', fontsize=12, color="black")

    # Set axis labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

    # Print Classification Report
    print(f"{title} Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["<50K", ">50K"]))


def plot_pr_roc_curves(df, models, colors):
    """
    Plots Precision-Recall and ROC Curves for given models.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the true labels and model predictions.
    models (list): List of column names representing model predictions.
    colors (list): List of colors for plotting curves.
    """
    y_true = df['label']
    
    plt.figure(figsize=(12, 6))
    
    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 1)
    for model, color in zip(models, colors):
        y_scores = df[model]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=color, label=f'{model} (PR-AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    
    # Plot ROC Curve
    plt.subplot(1, 2, 2)
    for model, color in zip(models, colors):
        y_scores = df[model]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        plt.plot(fpr, tpr, color=color, label=f'{model} (ROC-AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

