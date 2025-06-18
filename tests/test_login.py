import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
from streamlit.testing.v1 import AppTest
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_ui import main
from streamlit_pages.login import login_tab

class TestLoginFlow(unittest.TestCase):

    def setUp(self):
        """Set up a mock Supabase client before each test."""
        self.mock_supabase = MagicMock()
        self.patcher = patch('utils.utils.get_clients', return_value=(None, self.mock_supabase))
        self.mock_get_clients = self.patcher.start()

    def tearDown(self):
        """Stop the patcher after each test."""
        self.patcher.stop()

    def test_login_page_loads(self):
        """Test that the login page is displayed for an unauthenticated user."""
        at = AppTest.from_function(main)
        at.run()
        self.assertIn("Welcome to Archon", at.title[0].value)
        self.assertEqual(len(at.button), 2) # Login and Register buttons
        self.assertFalse(at.session_state) # No user in session state

    def test_successful_login(self):
        """Test that a user can log in successfully."""
        mock_user = MagicMock()
        self.mock_supabase.auth.sign_in_with_password.return_value = mock_user

        at = AppTest.from_function(main)
        at.run()
        
        # Simulate entering credentials and clicking login
        at.text_input(key="login_email").input("test@example.com").run()
        at.text_input(key="login_password").input("password").run()
        at.button(key="login_form-Login").click().run()

        # Verify user is in session state and main app is shown
        self.assertIn("user", at.session_state)
        self.assertEqual(at.session_state.user, mock_user)
        self.assertIn("Archon - Introduction", at.title[0].value) # Check for main app title

    def test_failed_login(self):
        """Test that an error is shown for a failed login."""
        self.mock_supabase.auth.sign_in_with_password.side_effect = Exception("Invalid credentials")

        at = AppTest.from_function(main)
        at.run()

        at.text_input(key="login_email").input("wrong@example.com").run()
        at.text_input(key="login_password").input("wrong").run()
        at.button(key="login_form-Login").click().run()

        self.assertNotIn("user", at.session_state)
        self.assertIn("Login failed: Invalid credentials", at.error[0].value)

    def test_successful_registration(self):
        """Test that a user can register successfully."""
        self.mock_supabase.auth.sign_up.return_value = (None, None) # Simulate successful signup

        at = AppTest.from_function(main)
        at.run()

        # Switch to registration tab
        at.tabs[0].selectbox(0).select("Register").run()

        at.text_input(key="register_email").input("new@example.com").run()
        at.text_input(key="register_password").input("new_password").run()
        at.button(key="registration_form-Register").click().run()

        self.assertIn("Registration successful!", at.success[0].value)

    def test_logout(self):
        """Test that a user can log out."""
        at = AppTest.from_function(main)
        at.session_state.user = MagicMock() # Simulate logged-in user
        at.run()

        # Verify main app is shown
        self.assertIn("Archon - Introduction", at.title[0].value)
        
        # Click logout
        at.sidebar.button[8].click().run() # The 9th button is logout

        # Verify user is logged out and login page is shown
        self.assertNotIn("user", at.session_state)
        self.assertIn("Welcome to Archon", at.title[0].value)

if __name__ == '__main__':
    unittest.main()