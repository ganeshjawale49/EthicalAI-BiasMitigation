"""
Machine Learning Engine Module.
Supports model selection, training, hyperparameter configuration, and standard performance metrics calculation.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_classifier(model_name, X_train, y_train, sample_weight=None):
    """
    Instantiates and fits the requested classification model.
    Supported model_name: 'LogisticRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting'
    """
    model_map = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    clf = model_map.get(model_name, LogisticRegression(max_iter=1000, random_state=42))
    
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
        
    y_prob = clf.predict_proba(X_test)[:, 1] if (clf is not None and hasattr(clf, 'predict_proba')) else y_pred.astype(float)
    
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
