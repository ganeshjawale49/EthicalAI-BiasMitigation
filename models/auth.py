"""
User Authentication and SQLite Database Connection Module.
Handles user registration, password hashing with Werkzeug, and user authentication.
"""
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

def get_db_connection():
    """Establishes and returns a connection to the SQLite database with Row factory enabled."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database tables using schema.sql if database does not exist."""
    conn = get_db_connection()
    with open('schema.sql', 'r', encoding='utf-8') as f:
        conn.executescript(f.read())
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
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except Exception as e:
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
