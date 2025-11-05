import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import warnings

def create_results_directory(dir_path):
    """Creates the results directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"INFO: Created directory for results: {dir_path}")

def plot_confusion_matrix(cm, classes, model_name, results_dir, f_out, suffix=""):
    """Saves a plot of the confusion matrix."""
    plt.figure(figsize=(max(8, len(classes) * 1.5), max(6, len(classes) * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix: {model_name}{suffix}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'{model_name.replace(" ", "_")}{suffix}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    f_out.write(f"Confusion matrix plot saved to: {plot_path}\n")
    print(f"  Saved confusion matrix for {model_name}{suffix} to {plot_path}")

def get_model_pipeline(numerical_features, categorical_features, model):
    """Creates a full preprocessing and modeling pipeline."""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough'
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def train_and_evaluate_single_model(X_df, y_series, numerical_features, categorical_features, 
                                   model_name, model, ordered_target_names, numeric_labels, 
                                   results_dir, f_out, suffix="", random_state=42):
    """Trains and evaluates a single model pipeline, returning key metrics."""
    f_out.write(f"\n--- Model: {model_name}{suffix} ---\n")
    print(f"\nTraining and evaluating: {model_name}{suffix}")
    
    pipeline = get_model_pipeline(numerical_features, categorical_features, model)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_y_true, all_y_pred, accuracies = [], [], []

    for train_idx, test_idx in tqdm(cv.split(X_df, y_series), total=cv.get_n_splits(), desc=f"  CV for {model_name}{suffix}"):
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        accuracies.append(accuracy_score(y_test, y_pred))

    if not all_y_pred:
        f_out.write("Model training failed for all folds. Skipping evaluation.\n")
        return 0.0, None, None

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    f_out.write(f"Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})\n\n")
    print(f"  Cross-validated Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    report = classification_report(all_y_true, all_y_pred, target_names=ordered_target_names, labels=numeric_labels, zero_division=0)
    f_out.write("Classification Report (from aggregated CV predictions):\n" + report + "\n\n")
    
    cm = confusion_matrix(all_y_true, all_y_pred, labels=numeric_labels)
    f_out.write("Confusion Matrix (from aggregated CV predictions):\n" + np.array2string(cm) + "\n")
    plot_confusion_matrix(cm, ordered_target_names, model_name, results_dir, f_out, suffix)
    
    return mean_accuracy, report, cm

def get_best_statistical_models():
    """Returns the best performing statistical models from experiments 1 and 2."""
    return {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000, 
                                                 multi_class='ovr', class_weight='balanced'),
        "LDA": LinearDiscriminantAnalysis(solver='svd'),
        "Linear SVM": SVC(kernel='linear', random_state=42, probability=False, class_weight='balanced')
    }
