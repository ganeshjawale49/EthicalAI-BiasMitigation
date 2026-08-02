"""
Fairness Metrics & Algorithmic Bias Detection Engine Module using Fairlearn.
Calculates key Ethical AI metrics: Disparate Impact, Demographic Parity, Equalized Odds, Equal Opportunity.
"""
import numpy as np
import fairlearn.metrics as flm

def compute_group_metrics(y_true, y_pred, A):
    """
    Computes confusion matrices, positive prediction rates (selection rates), 
    True Positive Rates (TPR), and False Positive Rates (FPR) for privileged (A=1) 
    and unprivileged (A=0) groups using Fairlearn metrics.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    A = np.array(A, dtype=int)
    
    priv_mask = (A == 1)
    unpriv_mask = (A == 0)
    
    # Fairlearn selection rates
    priv_sr = float(flm.selection_rate(y_true[priv_mask], y_pred[priv_mask])) if np.sum(priv_mask) > 0 else 0.0
    unpriv_sr = float(flm.selection_rate(y_true[unpriv_mask], y_pred[unpriv_mask])) if np.sum(unpriv_mask) > 0 else 0.0
    
    # Privileged Group confusion matrix
    priv_tp = int(np.sum((y_true[priv_mask] == 1) & (y_pred[priv_mask] == 1)))
    priv_fn = int(np.sum((y_true[priv_mask] == 1) & (y_pred[priv_mask] == 0)))
    priv_fp = int(np.sum((y_true[priv_mask] == 0) & (y_pred[priv_mask] == 1)))
    priv_tn = int(np.sum((y_true[priv_mask] == 0) & (y_pred[priv_mask] == 0)))
    
    priv_tpr = float(priv_tp / (priv_tp + priv_fn)) if (priv_tp + priv_fn) > 0 else 0.0
    priv_fpr = float(priv_fp / (priv_fp + priv_tn)) if (priv_fp + priv_tn) > 0 else 0.0
    
    # Unprivileged Group confusion matrix
    unpriv_tp = int(np.sum((y_true[unpriv_mask] == 1) & (y_pred[unpriv_mask] == 1)))
    unpriv_fn = int(np.sum((y_true[unpriv_mask] == 1) & (y_pred[unpriv_mask] == 0)))
    unpriv_fp = int(np.sum((y_true[unpriv_mask] == 0) & (y_pred[unpriv_mask] == 1)))
    unpriv_tn = int(np.sum((y_true[unpriv_mask] == 0) & (y_pred[unpriv_mask] == 0)))
    
    unpriv_tpr = float(unpriv_tp / (unpriv_tp + unpriv_fn)) if (unpriv_tp + unpriv_fn) > 0 else 0.0
    unpriv_fpr = float(unpriv_fp / (unpriv_fp + unpriv_tn)) if (unpriv_fp + unpriv_tn) > 0 else 0.0
    
    return {
        'priv_selection_rate': round(priv_sr, 4),
        'unpriv_selection_rate': round(unpriv_sr, 4),
        'priv_tpr': round(priv_tpr, 4),
        'unpriv_tpr': round(unpriv_tpr, 4),
        'priv_fpr': round(priv_fpr, 4),
        'unpriv_fpr': round(unpriv_fpr, 4),
        'priv_counts': {'tp': priv_tp, 'fp': priv_fp, 'fn': priv_fn, 'tn': priv_tn},
        'unpriv_counts': {'tp': unpriv_tp, 'fp': unpriv_fp, 'fn': unpriv_fn, 'tn': unpriv_tn}
    }

def evaluate_fairness(y_true, y_pred, A):
    """
    Calculates standardized Ethical AI fairness metrics using Fairlearn library:
    - Disparate Impact (DI) = Unprivileged SR / Privileged SR
    - Demographic Parity Difference = fairlearn.metrics.demographic_parity_difference
    - Equalized Odds Difference = fairlearn.metrics.equalized_odds_difference
    - Equal Opportunity Difference = fairlearn.metrics.true_positive_rate_difference
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    A = np.array(A, dtype=int)
    
    g_metrics = compute_group_metrics(y_true, y_pred, A)
    priv_sr = g_metrics['priv_selection_rate']
    unpriv_sr = g_metrics['unpriv_selection_rate']
    
    # Fairlearn calculations
    try:
        dpd = float(flm.demographic_parity_difference(y_true, y_pred, sensitive_features=A))
    except Exception:
        dpd = abs(priv_sr - unpriv_sr)
        
    try:
        eod = float(flm.equalized_odds_difference(y_true, y_pred, sensitive_features=A))
    except Exception:
        eod = (abs(g_metrics['priv_tpr'] - g_metrics['unpriv_tpr']) + abs(g_metrics['priv_fpr'] - g_metrics['unpriv_fpr'])) / 2.0
        
    try:
        eod_opp = float(flm.true_positive_rate_difference(y_true, y_pred, sensitive_features=A))
    except Exception:
        eod_opp = abs(g_metrics['priv_tpr'] - g_metrics['unpriv_tpr'])
        
    # Disparate Impact (Ratio of unprivileged selection rate to privileged selection rate)
    if priv_sr > 0:
        disparate_impact = round(unpriv_sr / priv_sr, 4)
    else:
        disparate_impact = 1.0 if unpriv_sr == 0 else 0.0
        
    demographic_parity_diff = round(dpd, 4)
    equalized_odds_diff = round(eod, 4)
    equal_opportunity_diff = round(eod_opp, 4)
    
    # Determine Fairness Status based on EEOC 80% Rule (Disparate Impact < 0.80)
    if disparate_impact < 0.80 or demographic_parity_diff > 0.15:
        fairness_status = 'BIASED (High Disparity)'
    elif disparate_impact < 0.90 or demographic_parity_diff > 0.10:
        fairness_status = 'MODERATE BIAS'
    else:
        fairness_status = 'FAIR (Balanced)'
        
    return {
        'disparate_impact': disparate_impact,
        'demographic_parity_diff': demographic_parity_diff,
        'equal_opportunity_diff': equal_opportunity_diff,
        'equalized_odds_diff': equalized_odds_diff,
        'fairness_status': fairness_status,
        'group_metrics': g_metrics
    }
