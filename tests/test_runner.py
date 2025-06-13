#!/usr/bin/env python3
"""
Test runner script for Recipe Recommendation API
Provides different test execution modes and reporting
"""

import pytest
import sys
import os
from pathlib import Path

def run_tests():
    """Main test runner function"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Base pytest arguments
    pytest_args = [
        str(tests_dir),
        "-v",  # Verbose output
        "-x",  # Stop on first failure
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker validation
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "coverage":
            # Run with coverage report
            pytest_args.extend([
                "--cov=app",
                "--cov=services",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-fail-under=70"
            ])
            print("🧪 Running tests with coverage analysis...")
            
        elif mode == "fast":
            # Run only fast tests (exclude integration)
            pytest_args.extend([
                "-m", "not integration",
                "--tb=line"
            ])
            print("⚡ Running fast tests only...")
            
        elif mode == "integration":
            # Run only integration tests
            pytest_args.extend([
                "-m", "integration",
                "--tb=long"
            ])
            print("🔗 Running integration tests only...")
            
        elif mode == "smoke":
            # Run basic smoke tests
            pytest_args.extend([
                "tests/test_health_endpoints.py::TestHealthEndpoints::test_health_check_with_model",
                "tests/test_recommendation_endpoints.py::TestRecommendationEndpoints::test_existing_user_recommendations_success",
                "--tb=line"
            ])
            print("💨 Running smoke tests...")
            
        elif mode == "parallel":
            # Run tests in parallel (requires pytest-xdist)
            pytest_args.extend([
                "-n", "auto",  # Use all available CPUs
            ])
            print("🚀 Running tests in parallel...")
            
        elif mode == "debug":
            # Run with debug options
            pytest_args.extend([
                "-s",  # Don't capture output
                "--tb=long",
                "--pdb-trace"  # Drop into debugger on failure
            ])
            print("🐛 Running tests in debug mode...")
            
        elif mode == "report":
            # Generate detailed HTML report
            pytest_args.extend([
                "--html=reports/test_report.html",
                "--self-contained-html",
                "--cov=app",
                "--cov=services",
                "--cov-report=html:reports/coverage_html"
            ])
            # Create reports directory
            os.makedirs("reports", exist_ok=True)
            print("📊 Running tests with detailed reporting...")
            
        else:
            print(f"❌ Unknown mode: {mode}")
            print_usage()
            return 1
    else:
        print("🧪 Running all tests with standard configuration...")
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    # Print summary based on exit code
    if exit_code == 0:
        print("\n✅ All tests passed!")
    elif exit_code == 1:
        print("\n❌ Some tests failed!")
    elif exit_code == 2:
        print("\n⚠️  Test execution was interrupted!")
    elif exit_code == 3:
        print("\n🔧 Internal pytest error!")
    elif exit_code == 4:
        print("\n⚙️  pytest usage error!")
    else:
        print(f"\n❓ Unknown exit code: {exit_code}")
    
    return exit_code

def print_usage():
    """Print usage information"""
    print("""
Usage: python test_runner.py [mode]

Available modes:
  coverage     - Run tests with coverage analysis
  fast         - Run only fast tests (exclude integration)
  integration  - Run only integration tests
  smoke        - Run basic smoke tests
  parallel     - Run tests in parallel
  debug        - Run tests with debug options
  report       - Generate detailed HTML reports
  
Examples:
  python test_runner.py coverage
  python test_runner.py fast
  python test_runner.py integration
    """)

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)