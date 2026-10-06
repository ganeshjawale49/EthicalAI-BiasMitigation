"""
User Authentication and SQLite/PostgreSQL Database Connection Module.
Handles user registration, password hashing with Werkzeug, and user authentication.
Supports PostgreSQL (e.g., Vercel Postgres) for persistent storage on serverless environments.
"""
import os
import shutil
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

# Support for PostgreSQL on Vercel
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

class Psycopg2CursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor
        
    def execute(self, query, params=()):
        # Convert sqlite ? to postgres %s
        query = query.replace('?', '%s')
        self._cursor.execute(query, params)
        return self
        
    def fetchone(self):
        return self._cursor.fetchone()
        
    def fetchall(self):
        return self._cursor.fetchall()
        
    @property
    def lastrowid(self):
        # In psycopg2, cursor.lastrowid is usually None unless using OIDs.
        # We assume queries that need lastrowid use 'RETURNING id' in execute() manually, 
        # but since we can't change all app queries easily, we must catch it.
        # Alternatively, for PostgreSQL, we can use an internal query to get the last val.
        # Since we modified register_user, it's safer to just return None or fetch.
        return None
        
    def __getattr__(self, name):
        return getattr(self._cursor, name)

class Psycopg2ConnectionWrapper:
    def __init__(self, conn):
        self._conn = conn
        
    def cursor(self):
        return Psycopg2CursorWrapper(self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor))
        
    def commit(self):
        self._conn.commit()
        
    def rollback(self):
        self._conn.rollback()
        
    def close(self):
        self._conn.close()
        
    def execute(self, query, params=()):
        cursor = self.cursor()
        cursor.execute(query, params)
        return cursor
        
    def executescript(self, script):
        cursor = self.cursor()
        # executescript in sqlite doesn't take parameters and runs multiple statements
        cursor._cursor.execute(script)
        self.commit()
        return cursor

def get_db_connection():
    """Establishes and returns a connection to the database (PostgreSQL or SQLite)."""
    db_url = os.environ.get('DATABASE_URL')
    if db_url and HAS_PSYCOPG2:
        conn = psycopg2.connect(db_url)
        return Psycopg2ConnectionWrapper(conn)
    
    # Fallback to SQLite
    conn = sqlite3.connect(Config.DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database tables using schema.sql if database does not exist."""
    base_db = os.path.join(Config.BASE_DIR, 'database.db')
    if Config.DATABASE_PATH != base_db and not os.path.exists(Config.DATABASE_PATH) and os.path.exists(base_db):
        try:
            shutil.copy2(base_db, Config.DATABASE_PATH)
        except Exception:
            pass

    conn = get_db_connection()
    schema_path = os.path.join(Config.BASE_DIR, 'schema.sql')
    if os.path.exists(schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_script = f.read()
            if os.environ.get('DATABASE_URL') and HAS_PSYCOPG2:
                # PostgreSQL requires splitting statements or using a transaction
                cursor = conn.cursor()
                # Simple replacement for autoincrement in Postgres
                schema_script = schema_script.replace('AUTOINCREMENT', 'GENERATED ALWAYS AS IDENTITY')
                try:
                    cursor.execute(schema_script)
                    conn.commit()
                except Exception as e:
                    print("Postgres Init Error:", e)
                    conn.rollback()
            else:
                conn.executescript(schema_script)
            
    # Auto-migration: ensure csv_content column exists in datasets table
    try:
        if os.environ.get('DATABASE_URL') and HAS_PSYCOPG2:
            conn.cursor().execute("ALTER TABLE datasets ADD COLUMN csv_content TEXT")
        else:
            conn.execute("ALTER TABLE datasets ADD COLUMN csv_content TEXT")
        conn.commit()
    except Exception:
        pass

    # Auto-migration: ensure group_metrics_json column exists in model_runs table
    try:
        if os.environ.get('DATABASE_URL') and HAS_PSYCOPG2:
            conn.cursor().execute("ALTER TABLE model_runs ADD COLUMN group_metrics_json TEXT DEFAULT NULL")
        else:
            conn.execute("ALTER TABLE model_runs ADD COLUMN group_metrics_json TEXT DEFAULT NULL")
        conn.commit()
    except Exception:
        pass

    conn.commit()
    conn.close()


def register_user(username, email, password):
    """
    Registers a new user account with hashed password.
    Returns (success_flag, message_or_user_id)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if username or email already exists
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return False, "Username or Email already registered."
    
    password_hash = generate_password_hash(password)
    try:
        is_pg = os.environ.get('DATABASE_URL') and HAS_PSYCOPG2
        if is_pg:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?) RETURNING id",
                (username, email, password_hash)
            )
            user_id = cursor.fetchone()['id']
        else:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            user_id = cursor.lastrowid
            
        conn.commit()
        conn.close()
        return True, user_id
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, str(e)

def authenticate_user(username_or_email, password):
    """
    Validates user credentials.
    Returns (success_flag, user_dict_or_error_msg)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email, username_or_email)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return True, dict(user)
    return False, "Invalid username/email or password."

def get_user_by_id(user_id):
    """Fetches user record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None
