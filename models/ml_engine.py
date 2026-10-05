"""
Machine Learning Engine Module.
Supports model selection, training, hyperparameter configuration, and standard performance metrics calculation.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_classifier(model_name, X_train, y_train, sample_weight=None):
    """
    Instantiates and fits the requested classification model with fast, optimized hyperparameters.
    Supported model_name: 'LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting'
    """
    if model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=15, max_depth=5, n_jobs=-1, random_state=42)
    elif model_name == 'GradientBoosting':
        # HistGradientBoostingClassifier is significantly faster than GradientBoostingClassifier
        # and supports sample_weight natively in modern scikit-learn (>= 1.3)
        clf = HistGradientBoostingClassifier(max_iter=10, max_depth=4, random_state=42)
    else:
        clf = LogisticRegression(max_iter=50, tol=5e-2, solver='lbfgs', random_state=42)
    
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
        
    return clf

def evaluate_performance(clf, X_test, y_test, y_pred=None):
    """
    Evaluates classifier accuracy, precision, recall, f1-score, and confusion matrix.
    Optionally accepts custom y_pred array (for post-processing threshold/fairness mitigation).
    Returns structured performance metrics dictionary.
    """
    if y_pred is None:
        y_pred = clf.predict(X_test)
        
    if clf is not None and hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_test)
        y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    else:
        y_prob = y_pred.astype(float)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    cm_dict = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'matrix': cm.tolist()
    }
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm_dict,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
