# project/algorand/security/security_audit.py

import os
import subprocess

def run_static_analysis():
    """
    Run static analysis tools to identify potential vulnerabilities in the code.
    """
    print("Running static analysis...")
    result = subprocess.run(['bandit', '-r', 'project/algorand/core'], capture_output=True, text=True)
    print(result.stdout)

def check_dependencies():
    """
    Check for known vulnerabilities in project dependencies using safety.
    """
    print("Checking dependencies for vulnerabilities...")
    result = subprocess.run(['safety', 'check'], capture_output=True, text=True)
    print(result.stdout)

def audit_codebase():
    """
    Perform a comprehensive security audit of the codebase.
    """
    run_static_analysis()
    check_dependencies()
