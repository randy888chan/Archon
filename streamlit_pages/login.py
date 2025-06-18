import streamlit as st
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

# Path to the users file
USERS_FILE = os.path.join("data", "users.json")

def load_users():
    """Load users from the JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Save users to the JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def login_tab():
    """Display the login page and handle user authentication."""
    st.title("Welcome to Archon")

    if 'user' in st.session_state:
        st.write(f"You are logged in as: {st.session_state['user']}")
        if st.button("Logout"):
            del st.session_state['user']
            st.rerun()
        return

    login_form, registration_form = st.tabs(["Login", "Register"])

    with login_form:
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button("Login")

            if login_button:
                users = load_users()
                if email in users and check_password_hash(users[email], password):
                    st.session_state['user'] = email
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with registration_form:
        with st.form("registration_form"):
            st.subheader("Create an Account")
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            register_button = st.form_submit_button("Register")

            if register_button:
                users = load_users()
                if new_email in users:
                    st.error("Email already registered.")
                else:
                    users[new_email] = generate_password_hash(new_password)
                    save_users(users)
                    st.success("Registration successful! You can now log in.")