"""
Bias Mitigation Engine Module using Fairlearn and Custom Optimization.
Provides Pre-processing (Reweighing) and Post-processing (Threshold Optimization) 
to eliminate algorithmic bias and recalculate Ethical AI metrics.
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
    Mitigates bias using Post-processing Threshold Calibration per sensitive subgroup.
    Finds optimal threshold shift for unprivileged group (A=0) to achieve demographic parity.
    Returns (mitigated_y_pred, perf_metrics, fairness_metrics, optimal_thresholds)
    """
    if not hasattr(clf, 'predict_proba'):
        y_pred = clf.predict(X_test)
        perf = evaluate_performance(clf, X_test, y_test, y_pred=y_pred)
        fairness = evaluate_fairness(y_test, y_pred, A_test)
        return y_pred, perf, fairness, {'priv_thresh': 0.5, 'unpriv_thresh': 0.5}
        
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    priv_mask = (A_test == 1)
    unpriv_mask = (A_test == 0)
    
    best_thresh = 0.5
    best_di_diff = 999.0
    best_y_pred = (y_prob >= 0.5).astype(int)
    
    # Search candidate thresholds for unprivileged group between 0.15 and 0.65
    for candidate_thresh in np.linspace(0.15, 0.65, 51):
        test_pred = np.zeros_like(y_prob, dtype=int)
        test_pred[priv_mask] = (y_prob[priv_mask] >= 0.5).astype(int)
        test_pred[unpriv_mask] = (y_prob[unpriv_mask] >= candidate_thresh).astype(int)
        
        eval_f = evaluate_fairness(y_test, test_pred, A_test)
        di = eval_f['disparate_impact']
        
        diff = abs(di - 1.0)
        if diff < best_di_diff:
            best_di_diff = diff
            best_thresh = candidate_thresh
            best_y_pred = test_pred
            
    # Compute final metrics with best predictions
    perf = evaluate_performance(clf, X_test, y_test, y_pred=best_y_pred)
    fairness = evaluate_fairness(y_test, best_y_pred, A_test)
    
    optimal_thresholds = {
        'priv_thresh': 0.50,
        'unpriv_thresh': round(float(best_thresh), 3)
    }
    
    return best_y_pred, perf, fairness, optimal_thresholds
