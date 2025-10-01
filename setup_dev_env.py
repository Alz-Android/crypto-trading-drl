#!/usr/bin/env python3
"""
Development setup script for crypto-trading-drl project.
Run this script to configure your development environment.
"""

import os
import sys
import json

def create_vscode_settings():
    """Create VSCode settings for Python path resolution."""
    vscode_dir = ".vscode"
    settings_file = os.path.join(vscode_dir, "settings.json")
    
    # Create .vscode directory if it doesn't exist
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
        print(f"✅ Created {vscode_dir} directory")
    
    # VSCode settings for better Python development
    settings = {
        "python.analysis.extraPaths": ["./src"],
        "python.defaultInterpreterPath": "python",
        "python.analysis.autoImportCompletions": True,
        "python.analysis.typeCheckingMode": "basic",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": False,
        "python.linting.flake8Enabled": True,
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            "**/node_modules": True,
            "**/.git": True,
            "**/.DS_Store": True,
            "**/Thumbs.db": True
        }
    }
    
    # Write settings file
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f"✅ Created {settings_file} with Python path configuration")
    return settings_file

def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up Crypto Trading DRL Development Environment")
    print("=" * 60)
    
    # Get current directory
    project_root = os.getcwd()
    src_path = os.path.join(project_root, "src")
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Shared src path: {src_path}")
    
    # Check if src directory exists
    if os.path.exists(src_path):
        print("✅ Shared src/ directory found")
        modules = os.listdir(src_path)
        print(f"📦 Available modules: {[m for m in modules if m.endswith('.py')]}")
    else:
        print("❌ Shared src/ directory not found!")
        return False
    
    # Create VSCode settings
    try:
        settings_file = create_vscode_settings()
        print(f"✅ VSCode settings configured: {settings_file}")
    except Exception as e:
        print(f"⚠️  Could not create VSCode settings: {e}")
    
    # Test imports
    print("\n🧪 Testing shared module imports...")
    sys.path.insert(0, src_path)
    
    try:
        from data_fetcher import DataFetcher
        print("✅ DataFetcher import successful")
    except ImportError as e:
        print(f"❌ DataFetcher import failed: {e}")
    
    try:
        from trading_env import TradingEnvironment
        print("✅ TradingEnvironment import successful")
    except ImportError as e:
        print(f"❌ TradingEnvironment import failed: {e}")
    
    try:
        from neural_networks import PPOAgent
        print("✅ PPOAgent import successful")
    except ImportError as e:
        print(f"❌ PPOAgent import failed: {e}")
    
    try:
        from technical_indicators import TechnicalIndicators
        print("✅ TechnicalIndicators import successful")
    except ImportError as e:
        print(f"❌ TechnicalIndicators import failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Development environment setup complete!")
    print("\n📝 Next steps:")
    print("1. Restart your IDE/editor to pick up the new settings")
    print("2. The 'could not be resolved' errors should now be fixed")
    print("3. All training scripts should work without import issues")
    print("\n🔧 Troubleshooting:")
    print("- If issues persist, restart VSCode completely")
    print("- Check that your Python interpreter points to the correct environment")
    
    return True

if __name__ == "__main__":
    setup_environment()