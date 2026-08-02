-- Database Schema for Ethical AI Bias Mitigation System
-- Database: SQLite

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(255) NOT NULL,
    row_count INTEGER DEFAULT 0,
    column_count INTEGER DEFAULT 0,
    target_column VARCHAR(100),
    sensitive_column VARCHAR(100),
    privileged_group VARCHAR(100),
    unprivileged_group VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    disparate_impact REAL NOT NULL,
    demographic_parity_diff REAL NOT NULL,
    equalized_odds_diff REAL NOT NULL,
    equal_opportunity_diff REAL NOT NULL,
    is_mitigated BOOLEAN DEFAULT 0,
    mitigation_method VARCHAR(100) DEFAULT 'None',
    confusion_matrix_json TEXT NOT NULL,
    fairness_status VARCHAR(50) DEFAULT 'Unchecked',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_run_id INTEGER NOT NULL,
    summary_text TEXT NOT NULL,
    explanation_text TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    recommendations_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(id) ON DELETE CASCADE
);
