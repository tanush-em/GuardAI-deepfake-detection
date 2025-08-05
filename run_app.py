#!/usr/bin/env python3
"""
Startup script for GuardAI Deepfake Detection Application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'opencv-python',
        'face_recognition', 'numpy', 'pandas', 'plotly',
        'matplotlib', 'seaborn', 'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_models():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    
    models_dir = Path("trained-models")
    if not models_dir.exists():
        print("⚠️  trained-models/ directory not found")
        print("Creating directory...")
        models_dir.mkdir(exist_ok=True)
        print("✅ Directory created")
    
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("⚠️  No model files found in trained-models/")
        print("Please add your trained model files (.pt) to the trained-models/ directory")
        print("The application will still run but won't be able to perform analysis")
    else:
        print(f"✅ Found {len(model_files)} model file(s):")
        for model_file in model_files:
            print(f"   - {model_file.name}")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs", "reports", "temp", "uploads", 
        "cache", "sessions", "trained-models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def initialize_config():
    """Initialize application configuration"""
    print("\n⚙️  Initializing configuration...")
    
    try:
        from config import initialize_config as init_config
        init_config()
        print("✅ Configuration initialized")
        return True
    except Exception as e:
        print(f"❌ Configuration initialization failed: {e}")
        return False

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_app.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_streamlit(port=8501, host="localhost", headless=False):
    """Start the Streamlit application"""
    print(f"\n🚀 Starting GuardAI Deepfake Detection...")
    print(f"   Port: {port}")
    print(f"   Host: {host}")
    print(f"   Headless: {headless}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", host
    ]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    try:
        print("\n🌐 Application will be available at:")
        print(f"   http://{host}:{port}")
        print("\nPress Ctrl+C to stop the application")
        print("=" * 50)
        
        # Start the application
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="GuardAI Deepfake Detection Startup")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on (default: 8501)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-checks", action="store_true", help="Skip all checks and start directly")
    
    args = parser.parse_args()
    
    print("🛡️  GuardAI Deepfake Detection - Startup")
    print("=" * 50)
    
    if not args.skip_checks:
        # Run startup checks
        if not check_dependencies():
            print("\n❌ Dependency check failed. Please install missing packages.")
            return 1
        
        if not create_directories():
            print("\n❌ Directory creation failed.")
            return 1
        
        if not initialize_config():
            print("\n❌ Configuration initialization failed.")
            return 1
        
        check_models()  # This is just a warning, not a failure
        
        if not args.skip_tests:
            if not run_tests():
                print("\n⚠️  Tests failed, but continuing...")
    
    # Start the application
    success = start_streamlit(args.port, args.host, args.headless)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 