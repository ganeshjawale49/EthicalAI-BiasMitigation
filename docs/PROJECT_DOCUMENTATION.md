# System Software Requirement Specification (SRS) & Architecture Documentation

## Project Title
**Mitigation of Bias and Improve Fairness in Machine Learning using Large Language Models Towards Ethical AI Systems**

---

## 1. System Requirements Specification

### 1.1 Software Requirements
- **Operating System**: Cross-platform (Windows 10/11, Linux, macOS)
- **Programming Language**: Python 3.12+
- **Web Framework**: Flask 3.0.3
- **Database Engine**: SQLite3
- **Machine Learning Libraries**: Scikit-Learn 1.5.0, Pandas 2.3.1, NumPy 1.26.4, Fairlearn 0.14.0
- **Presentation Exporter**: Python-PPTX 1.0.2
- **Frontend Technologies**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6), Bootstrap 5.3, FontAwesome 6.4, Chart.js 4.4

### 1.2 Functional Requirements
1. **User Authentication**: User registration, login, session management, password hashing via Werkzeug.
2. **Dataset Management**: CSV file upload, schema profiling, missing value check, column selection for target and sensitive features.
3. **Model Training**: Multi-algorithm classification (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting), performance metric calculation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
4. **Fairness Auditing**: Automated calculation of Disparate Impact Ratio, Demographic Parity Difference, Equalized Odds Difference, Equal Opportunity Difference.
5. **LLM Bias Explanation**: Google Gemini API REST integration for generating executive audit summaries, plain English disparity explanations, root cause diagnostics, and recommendations.
6. **Bias Mitigation Engine**: Pre-processing Reweighing (Kamiran & Calders sample weighting) and Post-processing Subgroup Threshold Calibration.
7. **Reporting & Deliverables**: Interactive dashboards, printable HTML reports, and automatic generation of a 15-Slide PowerPoint presentation deck (`.pptx`).

---

## 2. System Architecture Design

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|              (Bootstrap 5 • Jinja2 Templates • Chart.js Visualizations)          |
+----------------------------------------+------------------------------------------+
                                         | HTTP Requests / AJAX
                                         v
+-----------------------------------------------------------------------------------+
|                                  FLASK APP SERVER                                 |
|               (Routes • Session Auth • Configuration • Flash Alerts)              |
+--------+------------------+------------------+------------------+-----------------+
         |                  |                  |                  |
         v                  v                  v                  v
+------------------+ +--------------+ +------------------+ +-------------------+
|  DATASET MANAGER | |  ML ENGINE   | |  BIAS DETECTOR   | |  BIAS MITIGATOR   |
| (Pandas Ingest • | | (Logistic,   | | (Disparate Impact| | (Pre: Reweighing •  |
|  Preprocessing)  | |  RandomForest)| | DemographicParity)| | Post: Threshold)  |
+------------------+ +--------------+ +------------------+ +-------------------+
         |                                     |                            |
         +-----------------+-------------------+----------------------------+
                           |
                           v
+-----------------------------------------------------------------------------------+
|                             LLM EXPLAINER & DB LAYER                              |
|   (Google Gemini REST API Integration • SQLite Database: Users, Runs, Audits)    |
+-----------------------------------------------------------------------------------+
```

---

## 3. Database Schema

### Table: `users`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique User Identifier |
| `username` | VARCHAR(80) | UNIQUE, NOT NULL | Account Username |
| `email` | VARCHAR(120) | UNIQUE, NOT NULL | User Email Address |
| `password_hash` | VARCHAR(255) | NOT NULL | Werkzeug Hashed Password |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Registration Date |

### Table: `datasets`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Dataset Record Identifier |
| `user_id` | INTEGER | FOREIGN KEY | Associated User |
| `filename` | VARCHAR(255) | NOT NULL | CSV Storage Filename |
| `target_column` | VARCHAR(100) | NULL | Target Outcome Label |
| `sensitive_column`| VARCHAR(100) | NULL | Protected Demographic Attribute |
| `privileged_group`| VARCHAR(100) | NULL | Baseline Favor Group (Male/White) |

### Table: `model_runs`
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Execution ID |
| `model_name` | VARCHAR(100) | NOT NULL | Classifier Algorithm Name |
| `accuracy` | REAL | NOT NULL | Test Classification Accuracy |
| `disparate_impact`| REAL | NOT NULL | Ethical Disparate Impact Ratio |
| `is_mitigated` | BOOLEAN | DEFAULT 0 | Mitigation Flag |
| `mitigation_method`| VARCHAR(100)| DEFAULT 'None' | Mitigation Algorithm Used |
