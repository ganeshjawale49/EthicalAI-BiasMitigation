"""
Bias Mitigation Engine Module using Fairlearn and Custom Optimization.
Provides Pre-processing (Reweighing) and Post-processing (Threshold Optimization) 
to eliminate algorithmic bias and recalculate Bias-Free AI metrics.
"""
import numpy as np
from models.ml_engine import train_classifier, evaluate_performance
from models.bias_detector import evaluate_fairness

def compute_reweighing_weights(y_train, A_train):
    """
    Computes Kamiran & Calders Reweighing sample weights.
    W(a, y) = ( P(A=a) * P(Y=y) ) / P(A=a, Y=y)
    """
    n_samples = len(y_train)
    weights = np.ones(n_samples, dtype=float)
    
    for a_val in [0, 1]:
        for y_val in [0, 1]:
            mask_a = (A_train == a_val)
            mask_y = (y_train == y_val)
            mask_ay = mask_a & mask_y
            
            p_a = np.mean(mask_a)
            p_y = np.mean(mask_y)
            p_ay = np.mean(mask_ay)
            
            if p_ay > 0:
                expected_prob = p_a * p_y
                actual_prob = p_ay
                w = expected_prob / actual_prob
                weights[mask_ay] = w
                
    return weights

def apply_reweighing_mitigation(model_name, X_train, y_train, A_train, X_test, y_test, A_test):
    """
    Mitigates bias using Pre-processing Reweighing algorithm.
    Retrains model with inverse frequency sample weights.
    Returns (mitigated_clf, perf_metrics, fairness_metrics)
    """
    sample_weights = compute_reweighing_weights(y_train, A_train)
    clf = train_classifier(model_name, X_train, y_train, sample_weight=sample_weights)
    
    perf = evaluate_performance(clf, X_test, y_test)
    fairness = evaluate_fairness(y_test, perf['y_pred'], A_test)
    
    return clf, perf, fairness

def apply_threshold_mitigation(clf, X_test, y_test, A_test, target_di=0.95):
    """
    Mitigates bias using Post-processing Joint 2D Threshold Calibration.
    Performs a joint grid search over both privileged and unprivileged probability thresholds
    to find the combination that achieves the closest Disparate Impact to 1.0 (full fairness).
    Returns (mitigated_y_pred, perf_metrics, fairness_metrics, optimal_thresholds)
    """
    if not hasattr(clf, 'predict_proba'):
        y_pred = clf.predict(X_test)
        perf = evaluate_performance(clf, X_test, y_test, y_pred=y_pred)
        fairness = evaluate_fairness(y_test, y_pred, A_test)
        return y_pred, perf, fairness, {'priv_thresh': 0.5, 'unpriv_thresh': 0.5}
        
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_true_arr = np.array(y_test, dtype=int)
    A_arr = np.array(A_test, dtype=int)
    
    priv_mask = (A_arr == 1)
    unpriv_mask = (A_arr == 0)
    
    best_priv_thresh = 0.5
    best_unpriv_thresh = 0.5
    best_di_diff = 999.0
    best_y_pred = (y_prob >= 0.5).astype(int)
    
    # Optimized grid search: coarser grid with inlined DI calculation
    # Privileged threshold ranges 0.40 - 0.80, Unprivileged threshold ranges 0.05 - 0.55
    for priv_thresh in np.linspace(0.40, 0.80, 6):
        priv_preds = (y_prob[priv_mask] >= priv_thresh).astype(int)
        priv_sr = float(np.mean(priv_preds)) if priv_preds.size > 0 else 0.0
        
        if priv_sr == 0:
            continue
            
        for unpriv_thresh in np.linspace(0.05, 0.55, 7):
            unpriv_preds = (y_prob[unpriv_mask] >= unpriv_thresh).astype(int)
            unpriv_sr = float(np.mean(unpriv_preds)) if unpriv_preds.size > 0 else 0.0
            
            # Inline DI calculation (much faster than calling evaluate_fairness)
            di = unpriv_sr / priv_sr if priv_sr > 0 else (1.0 if unpriv_sr == 0 else 0.0)
            
            diff = abs(di - 1.0)
            if diff < best_di_diff:
                best_di_diff = diff
                best_priv_thresh = priv_thresh
                best_unpriv_thresh = unpriv_thresh
                test_pred = np.zeros_like(y_prob, dtype=int)
                test_pred[priv_mask] = priv_preds
                test_pred[unpriv_mask] = unpriv_preds
                best_y_pred = test_pred
            
    # Compute final metrics with best predictions
    perf = evaluate_performance(clf, X_test, y_test, y_pred=best_y_pred)
    fairness = evaluate_fairness(y_test, best_y_pred, A_test)
    
    optimal_thresholds = {
        'priv_thresh': round(float(best_priv_thresh), 3),
        'unpriv_thresh': round(float(best_unpriv_thresh), 3)
    }
    
    return best_y_pred, perf, fairness, optimal_thresholds
