"""
Main Flask Web Application File.
Mitigation of Bias and Improve Fairness in Machine Learning using Large Language Models Towards Ethical AI Systems.
"""
import os
import json
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from config import Config
from models.auth import init_db, register_user, authenticate_user, get_user_by_id, get_db_connection
from models.dataset_manager import (
    save_uploaded_dataset, get_user_datasets, get_dataset_by_id, 
    inspect_dataset, update_dataset_config, preprocess_dataset
)
from models.ml_engine import train_classifier, evaluate_performance
from models.bias_detector import evaluate_fairness, compute_group_metrics
from models.bias_mitigator import apply_reweighing_mitigation, apply_threshold_mitigation
from models.llm_explainer import analyze_bias_with_llm, save_llm_audit
from models.ppt_generator import create_presentation_deck

app = Flask(__name__)
app.config.from_object(Config)

# Ensure SQLite tables exist on start
with app.app_context():
    init_db()

def login_required(f):
    """Decorator to enforce login requirement on protected routes."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ----------------------------
# AUTHENTICATION ROUTES
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET'])
def login_page():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username_or_email = request.form.get('username_or_email')
    password = request.form.get('password')
    
    success, res = authenticate_user(username_or_email, password)
    if success:
        session['user_id'] = res['id']
        session['username'] = res['username']
        flash(f"Welcome back, {res['username']}!", 'success')
        return redirect(url_for('dashboard'))
    else:
        flash(res, 'error')
        return redirect(url_for('login_page'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    
    success, res = register_user(username, email, password)
    if success:
        session['user_id'] = res
        session['username'] = username
        flash('Account registered successfully!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash(f"Registration failed: {res}", 'error')
        return redirect(url_for('login_page'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

# ----------------------------
# DASHBOARD & DATASETS
# ----------------------------
@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    datasets = get_user_datasets(user_id)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT m.*, d.filename FROM model_runs m 
           JOIN datasets d ON m.dataset_id = d.id 
           WHERE m.user_id = ? ORDER BY m.created_at DESC""",
        (user_id,)
    )
    model_runs = [dict(r) for r in cursor.fetchall()]
    conn.close()
    
    mitigated_count = sum(1 for r in model_runs if r['is_mitigated'])
    biased_count = sum(1 for r in model_runs if 'BIASED' in r['fairness_status'])
    
    return render_template(
        'dashboard.html', 
        datasets=datasets, 
        model_runs=model_runs,
        mitigated_count=mitigated_count,
        biased_count=biased_count
    )

@app.route('/upload')
@login_required
def upload_page():
    datasets = get_user_datasets(session['user_id'])
    return render_template('upload.html', datasets=datasets)

@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part selected.', 'error')
        return redirect(url_for('upload_page'))
    
    file_obj = request.files['file']
    action = request.form.get('action', 'configure')

    success, res = save_uploaded_dataset(session['user_id'], file_obj)
    if success:
        flash('Dataset uploaded successfully!', 'success')
        if action == 'train':
            return redirect(url_for('train_page', dataset_id=res['id']))
        return redirect(url_for('preprocess_page', dataset_id=res['id']))
    else:
        flash(res, 'error')
        return redirect(url_for('upload_page'))

@app.route('/load_sample', methods=['GET', 'POST'])
@login_required
def load_sample_dataset():
    user_id = session['user_id']
    filename = 'sample_credit_bias.csv'
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    base_sample_path = os.path.join(Config.BASE_DIR, 'static', 'uploads', filename)
    
    csv_content = ""
    target_path = filepath if os.path.exists(filepath) else base_sample_path
    if os.path.exists(target_path):
        try:
            with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                csv_content = f.read()
        except Exception:
            pass

    if not os.path.exists(filepath):
        if os.path.exists(base_sample_path):
            import shutil
            try:
                shutil.copy2(base_sample_path, filepath)
            except Exception:
                filepath = base_sample_path
        else:
            flash('Benchmark dataset file missing.', 'error')
            return redirect(url_for('upload_page'))
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO datasets 
           (user_id, filename, filepath, csv_content, row_count, column_count, target_column, sensitive_column, privileged_group, unprivileged_group)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, filename, filepath, csv_content, 1000, 8, 'Credit_Approved', 'Gender', 'Male', 'Female')
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    
    flash('Benchmark credit scoring dataset loaded successfully!', 'success')
    return redirect(url_for('preprocess_page', dataset_id=dataset_id))


# ----------------------------
# PREPROCESSING & TRAINING
# ----------------------------
@app.route('/preprocess/<int:dataset_id>')
@login_required
def preprocess_page(dataset_id):
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset not found.', 'error')
        return redirect(url_for('upload_page'))
        
    try:
        info = inspect_dataset(ds['filepath'])
        return render_template('preprocess.html', dataset=ds, info=info)
    except Exception as e:
        flash(f"Unable to load dataset details: {str(e)}", 'error')
        return redirect(url_for('upload_page'))

@app.route('/save_preprocess/<int:dataset_id>', methods=['POST'])
@login_required
def save_preprocess_config(dataset_id):
    target_col = request.form.get('target_column')
    sensitive_col = request.form.get('sensitive_column')
    priv_group = request.form.get('privileged_group')
    unpriv_group = request.form.get('unprivileged_group')
    
    update_dataset_config(dataset_id, target_col, sensitive_col, priv_group, unpriv_group)
    flash('Dataset configuration saved.', 'success')
    return redirect(url_for('train_page', dataset_id=dataset_id))

@app.route('/train/<int:dataset_id>')
@login_required
def train_page(dataset_id):
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset not found.', 'error')
        return redirect(url_for('upload_page'))
    return render_template('train.html', dataset=ds)

@app.route('/train_model_action/<int:dataset_id>', methods=['POST'])
@login_required
def train_model_action(dataset_id):
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset record not found.', 'error')
        return redirect(url_for('upload_page'))
        
    model_name = request.form.get('model_name', 'RandomForest')
    
    # --- Step 1: Preprocess ---
    try:
        prep = preprocess_dataset(
            ds['filepath'], ds['target_column'],
            ds['sensitive_column'], ds['privileged_group']
        )
    except FileNotFoundError as e:
        flash(f'Dataset file not found. Please re-upload the CSV. ({str(e)})', 'error')
        return redirect(url_for('train_page', dataset_id=dataset_id))
    except Exception as e:
        traceback.print_exc()
        flash(f'Preprocessing failed: {str(e)}', 'error')
        return redirect(url_for('train_page', dataset_id=dataset_id))

    # --- Step 2: Train ---
    try:
        clf = train_classifier(model_name, prep['X_train'], prep['y_train'])
    except Exception as e:
        traceback.print_exc()
        flash(f'Model training failed for "{model_name}": {str(e)}', 'error')
        return redirect(url_for('train_page', dataset_id=dataset_id))

    # --- Step 3: Evaluate Performance & Fairness ---
    try:
        perf = evaluate_performance(clf, prep['X_test'], prep['y_test'])
        fairness = evaluate_fairness(prep['y_test'], perf['y_pred'], prep['A_test'])
    except Exception as e:
        traceback.print_exc()
        flash(f'Evaluation failed: {str(e)}', 'error')
        return redirect(url_for('train_page', dataset_id=dataset_id))

    # --- Step 4: Save Model Run to DB ---
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_runs 
               (user_id, dataset_id, model_name, accuracy, precision_score, recall_score, f1_score, 
                disparate_impact, demographic_parity_diff, equalized_odds_diff, equal_opportunity_diff, 
                confusion_matrix_json, fairness_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session['user_id'], dataset_id, model_name,
                perf['accuracy'], perf['precision'], perf['recall'], perf['f1_score'],
                fairness['disparate_impact'], fairness['demographic_parity_diff'],
                fairness['equalized_odds_diff'], fairness['equal_opportunity_diff'],
                json.dumps(perf['confusion_matrix']), fairness['fairness_status']
            )
        )
        conn.commit()
        model_run_id = cursor.lastrowid
        conn.close()
    except Exception as e:
        traceback.print_exc()
        flash(f'Database error while saving model run: {str(e)}', 'error')
        return redirect(url_for('train_page', dataset_id=dataset_id))
        
    flash(
        f'✅ {model_name} trained successfully! '
        f'Accuracy: {perf["accuracy"]*100:.1f}% | '
        f'Disparate Impact: {fairness["disparate_impact"]} | '
        f'Status: {fairness["fairness_status"]}',
        'success'
    )
    return redirect(url_for('bias_detection_page', model_run_id=model_run_id))


# ----------------------------
# AJAX TRAIN API  (called by train.html via fetch)
# ----------------------------
@app.route('/api/train/<int:dataset_id>', methods=['POST'])
@login_required
def api_train(dataset_id):
    """JSON endpoint used by train page AJAX to run training and return results."""
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        return jsonify({'success': False, 'step': 'load', 'error': 'Dataset record not found in database.'}), 404

    model_name = request.json.get('model_name', 'RandomForest') if request.is_json else request.form.get('model_name', 'RandomForest')

    # Step 1 — Preprocess
    try:
        prep = preprocess_dataset(
            ds['filepath'], ds['target_column'],
            ds['sensitive_column'], ds['privileged_group']
        )
    except FileNotFoundError as e:
        return jsonify({'success': False, 'step': 'preprocess',
                        'error': f'Dataset file not found. Please re-upload the CSV file. Detail: {str(e)}'}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'step': 'preprocess', 'error': f'Preprocessing failed: {str(e)}'}), 400

    # Step 2 — Train
    try:
        clf = train_classifier(model_name, prep['X_train'], prep['y_train'])
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'step': 'train', 'error': f'Model training failed for "{model_name}": {str(e)}'}), 400

    # Step 3 — Evaluate
    try:
        perf = evaluate_performance(clf, prep['X_test'], prep['y_test'])
        fairness = evaluate_fairness(prep['y_test'], perf['y_pred'], prep['A_test'])
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'step': 'evaluate', 'error': f'Evaluation failed: {str(e)}'}), 400

    # Step 4 — Save to DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_runs
               (user_id, dataset_id, model_name, accuracy, precision_score, recall_score, f1_score,
                disparate_impact, demographic_parity_diff, equalized_odds_diff, equal_opportunity_diff,
                confusion_matrix_json, fairness_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session['user_id'], dataset_id, model_name,
                perf['accuracy'], perf['precision'], perf['recall'], perf['f1_score'],
                fairness['disparate_impact'], fairness['demographic_parity_diff'],
                fairness['equalized_odds_diff'], fairness['equal_opportunity_diff'],
                json.dumps(perf['confusion_matrix']), fairness['fairness_status']
            )
        )
        conn.commit()
        model_run_id = cursor.lastrowid
        conn.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'step': 'save', 'error': f'Database save failed: {str(e)}'}), 500

    return jsonify({
        'success': True,
        'model_run_id': model_run_id,
        'redirect_url': url_for('bias_detection_page', model_run_id=model_run_id),
        'metrics': {
            'accuracy': round(perf['accuracy'] * 100, 1),
            'precision': round(perf['precision'] * 100, 1),
            'recall': round(perf['recall'] * 100, 1),
            'f1_score': round(perf['f1_score'] * 100, 1),
            'disparate_impact': fairness['disparate_impact'],
            'demographic_parity_diff': fairness['demographic_parity_diff'],
            'fairness_status': fairness['fairness_status'],
        }
    })


# ----------------------------
# DELETE DATASET
# ----------------------------
@app.route('/delete_dataset/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    """Deletes a dataset and all its associated model runs from the database and filesystem."""
    user_id = session['user_id']
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        flash('Dataset not found.', 'error')
        return redirect(request.referrer or url_for('dashboard'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Delete related LLM audits first to honor foreign key constraints
        cursor.execute("DELETE FROM llm_audits WHERE model_run_id IN (SELECT id FROM model_runs WHERE dataset_id = ? AND user_id = ?)", (dataset_id, user_id))
        # Delete related model runs
        cursor.execute("DELETE FROM model_runs WHERE dataset_id = ? AND user_id = ?", (dataset_id, user_id))
        # Delete dataset record
        cursor.execute("DELETE FROM datasets WHERE id = ? AND user_id = ?", (dataset_id, user_id))
        conn.commit()
        conn.close()

        # Remove physical file from disk if it exists
        filepath = ds.get('filepath')
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

        flash(f'Dataset "{ds["filename"]}" and all related model runs deleted successfully.', 'success')
    except Exception as e:
        traceback.print_exc()
        flash(f'Error deleting dataset: {str(e)}', 'error')

    return redirect(request.referrer or url_for('dashboard'))


# ----------------------------
# BIAS DETECTION & LLM AUDIT
# ----------------------------
@app.route('/bias_detection/<int:model_run_id>')
@login_required
def bias_detection_page(model_run_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_runs WHERE id = ?", (model_run_id,))
    run = cursor.fetchone()
    conn.close()
    
    if not run:
        flash('Model run not found.', 'error')
        return redirect(url_for('dashboard'))
        
    ds = get_dataset_by_id(run['dataset_id'])
    if not ds:
        flash('Dataset record not found.', 'error')
        return redirect(url_for('dashboard'))

    try:
        prep = preprocess_dataset(ds['filepath'], ds['target_column'], ds['sensitive_column'], ds['privileged_group'])
        clf = train_classifier(run['model_name'], prep['X_train'], prep['y_train'])
        perf = evaluate_performance(clf, prep['X_test'], prep['y_test'])
        g_metrics = compute_group_metrics(prep['y_test'], perf['y_pred'], prep['A_test'])
        
        return render_template(
            'bias_detection.html', 
            run=dict(run), 
            dataset=ds, 
            cm=json.loads(run['confusion_matrix_json']),
            g_metrics=g_metrics
        )
    except Exception as e:
        flash(f"Unable to load bias evaluation details: {str(e)}", 'error')
        return redirect(url_for('dashboard'))

@app.route('/llm_audit/<int:model_run_id>')
@login_required
def llm_audit_page(model_run_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_runs WHERE id = ?", (model_run_id,))
    run = cursor.fetchone()
    conn.close()
    
    if not run:
        flash('Model run not found.', 'error')
        return redirect(url_for('dashboard'))
        
    ds = get_dataset_by_id(run['dataset_id'])
    if not ds:
        flash('Dataset record not found.', 'error')
        return redirect(url_for('dashboard'))

    try:
        prep = preprocess_dataset(ds['filepath'], ds['target_column'], ds['sensitive_column'], ds['privileged_group'])
        clf = train_classifier(run['model_name'], prep['X_train'], prep['y_train'])
        perf = evaluate_performance(clf, prep['X_test'], prep['y_test'])
        fairness = evaluate_fairness(prep['y_test'], perf['y_pred'], prep['A_test'])
        
        # Check if existing audit saved
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM llm_audits WHERE model_run_id = ?", (model_run_id,))
        existing_audit = cursor.fetchone()
        conn.close()
        
        if existing_audit:
            audit_dict = {
                'summary_text': existing_audit['summary_text'],
                'explanation_text': existing_audit['explanation_text'],
                'root_cause': existing_audit['root_cause'],
                'recommendations': json.loads(existing_audit['recommendations_json']),
                'source': 'Saved LLM Audit'
            }
        else:
            audit = analyze_bias_with_llm(
                run['model_name'], ds['sensitive_column'], ds['privileged_group'], ds['unprivileged_group'],
                perf, fairness
            )
            save_llm_audit(model_run_id, audit)
            audit_dict = {
                'summary_text': audit['summary'],
                'explanation_text': audit['explanation'],
                'root_cause': audit['root_cause'],
                'recommendations': audit['recommendations'],
                'source': audit['source']
            }
            
        return render_template('llm_audit.html', run=dict(run), dataset=ds, audit=audit_dict, recommendations=audit_dict['recommendations'])
    except Exception as e:
        flash(f"Unable to generate LLM audit report: {str(e)}", 'error')
        return redirect(url_for('dashboard'))

# ----------------------------
# BIAS MITIGATION
# ----------------------------
@app.route('/mitigation/<int:model_run_id>')
@login_required
def mitigation_page(model_run_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_runs WHERE id = ?", (model_run_id,))
    run = cursor.fetchone()
    conn.close()
    
    if not run:
        flash('Model run not found.', 'error')
        return redirect(url_for('dashboard'))

    ds = get_dataset_by_id(run['dataset_id'])
    return render_template('mitigation.html', run=dict(run), dataset=ds, mitigated_run=None)

@app.route('/run_mitigation/<int:model_run_id>', methods=['POST'])
@login_required
def run_mitigation(model_run_id):
    mitigation_method = request.form.get('mitigation_method', 'Threshold_Optimization')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_runs WHERE id = ?", (model_run_id,))
    orig_run = cursor.fetchone()
    conn.close()
    
    if not orig_run:
        flash('Model run not found.', 'error')
        return redirect(url_for('dashboard'))

    ds = get_dataset_by_id(orig_run['dataset_id'])
    if not ds:
        flash('Dataset record not found.', 'error')
        return redirect(url_for('dashboard'))

    try:
        prep = preprocess_dataset(ds['filepath'], ds['target_column'], ds['sensitive_column'], ds['privileged_group'])
        
        if mitigation_method == 'Reweighing':
            clf, perf, fairness = apply_reweighing_mitigation(
                orig_run['model_name'], prep['X_train'], prep['y_train'], prep['A_train'],
                prep['X_test'], prep['y_test'], prep['A_test']
            )
        else:
            # Threshold Optimization
            clf = train_classifier(orig_run['model_name'], prep['X_train'], prep['y_train'])
            _, perf, fairness, _ = apply_threshold_mitigation(clf, prep['X_test'], prep['y_test'], prep['A_test'])
            
        # Save Mitigated Run to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_runs 
               (user_id, dataset_id, model_name, accuracy, precision_score, recall_score, f1_score, 
                disparate_impact, demographic_parity_diff, equalized_odds_diff, equal_opportunity_diff, 
                is_mitigated, mitigation_method, confusion_matrix_json, fairness_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (
                session['user_id'], orig_run['dataset_id'], f"{orig_run['model_name']} (Mitigated)",
                perf['accuracy'], perf['precision'], perf['recall'], perf['f1_score'],
                fairness['disparate_impact'], fairness['demographic_parity_diff'],
                fairness['equalized_odds_diff'], fairness['equal_opportunity_diff'],
                mitigation_method, json.dumps(perf['confusion_matrix']), fairness['fairness_status']
            )
        )
        conn.commit()
        mitigated_run_id = cursor.lastrowid
        
        cursor.execute("SELECT * FROM model_runs WHERE id = ?", (mitigated_run_id,))
        mitigated_run = cursor.fetchone()
        conn.close()
        
        flash(f"Bias mitigation applied using {mitigation_method}! Disparate Impact improved to {fairness['disparate_impact']}", 'success')
        return render_template('mitigation.html', run=dict(orig_run), dataset=ds, mitigated_run=dict(mitigated_run))
    except Exception as e:
        flash(f"Error executing bias mitigation: {str(e)}", 'error')
        return redirect(url_for('mitigation_page', model_run_id=model_run_id))

# ----------------------------
# EVALUATION & REPORTS
# ----------------------------
@app.route('/evaluation')
@login_required
def evaluation_page():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT m.*, d.filename FROM model_runs m 
           JOIN datasets d ON m.dataset_id = d.id 
           WHERE m.user_id = ? ORDER BY m.created_at DESC""",
        (session['user_id'],)
    )
    model_runs = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template('evaluation.html', model_runs=model_runs)

@app.route('/reports')
@login_required
def reports_page():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT m.*, d.filename FROM model_runs m 
           JOIN datasets d ON m.dataset_id = d.id 
           WHERE m.user_id = ? ORDER BY m.created_at DESC""",
        (session['user_id'],)
    )
    model_runs = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template('reports.html', model_runs=model_runs)

@app.route('/download_ppt')
def download_ppt():
    ppt_path = os.path.join(Config.UPLOAD_FOLDER, 'Ethical_AI_Bias_Mitigation_Presentation.pptx')
    create_presentation_deck(ppt_path)
    return send_file(ppt_path, as_attachment=True, download_name='Ethical_AI_Bias_Mitigation_15_Slides.pptx')

@app.route('/generate_printable_report')
@login_required
def generate_printable_report():
    model_run_id = request.args.get('model_run_id', type=int)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_runs WHERE id = ?", (model_run_id,))
    run = cursor.fetchone()
    
    if not run:
        return "Report not found", 444
        
    ds = get_dataset_by_id(run['dataset_id'])
    cursor.execute("SELECT * FROM llm_audits WHERE model_run_id = ?", (model_run_id,))
    audit = cursor.fetchone()
    conn.close()
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ethical AI Bias Audit Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #1e293b; }}
            h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
            .metric-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .metric-table th, .metric-table td {{ border: 1px solid #cbd5e1; padding: 10px; text-align: left; }}
            .metric-table th {{ background-color: #0f172a; color: white; }}
            .badge {{ padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }}
            .badge-danger {{ background-color: #ef4444; }}
            .badge-success {{ background-color: #10b981; }}
        </style>
    </head>
    <body onload="window.print()">
        <h1>Ethical AI System: Algorithmic Bias Audit Report</h1>
        <p><strong>Project Title:</strong> Mitigation of Bias & Improve Fairness in ML using LLMs Towards Ethical AI Systems</p>
        <hr>
        <h3>Model Overview</h3>
        <p><strong>Model Algorithm:</strong> {run['model_name']}</p>
        <p><strong>Dataset:</strong> {ds['filename']}</p>
        <p><strong>Sensitive Attribute:</strong> {ds['sensitive_column']} ({ds['privileged_group']} vs {ds['unprivileged_group']})</p>
        
        <h3>Performance & Fairness Metrics</h3>
        <table class="metric-table">
            <tr><th>Metric</th><th>Score</th><th>Threshold / Target</th></tr>
            <tr><td>Accuracy</td><td>{run['accuracy'] * 100:.1f}%</td><td>> 80%</td></tr>
            <tr><td>Disparate Impact Ratio</td><td><strong>{run['disparate_impact']}</strong></td><td>&ge; 0.80 (EEOC 80% Rule)</td></tr>
            <tr><td>Demographic Parity Difference</td><td>{run['demographic_parity_diff']}</td><td>&le; 0.10</td></tr>
            <tr><td>Fairness Status</td><td colspan="2">{run['fairness_status']}</td></tr>
        </table>
        
        <h3>LLM Audit Summary</h3>
        <p>{audit['summary_text'] if audit else 'Baseline mathematical metrics recorded.'}</p>
        
        <br><hr>
        <p style="text-align: center; color: #64748b; font-size: 12px;">Generated by Ethical AI Fairness System • Final Year Project Deliverable</p>
    </body>
    </html>
    """
    return report_html

if __name__ == '__main__':
    print("Starting Ethical AI Bias Mitigation Web Server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
