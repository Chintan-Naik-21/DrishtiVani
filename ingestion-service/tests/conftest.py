"""Pytest fixtures and test configuration."""

import os
import sys

# Ensure ingestion-service root and app package are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variables for testing
os.environ["APP_ENV"] = "test"
os.environ["SKIP_STARTUP_CHECKS"] = "true"
os.environ["DATABASE_URL"] = "postgresql://test_user:test_pass@localhost:6543/postgres"
os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "mock-service-role-key"
