#!/usr/bin/env python3
"""
Test script to verify the setup of the Document Chat with RAG application.
Run this script to check if all dependencies and Ollama are properly configured.
"""

import sys
import importlib
import subprocess
import requests
import json

def test_python_dependencies():
    """Test if all required Python packages are installed."""
    print("🔍 Testing Python dependencies...")
    
    required_packages = [
        'streamlit',
        'ollama', 
        'PyPDF2',
        'numpy',
        'langchain',
        'langchain_text_splitters'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All Python dependencies are installed!")
        return True

def test_ollama_installation():
    """Test if Ollama is installed and accessible."""
    print("\n🔍 Testing Ollama installation...")
    
    try:
        # Try to run ollama --version
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print(f"  ❌ Ollama command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ❌ Ollama is not installed or not in PATH")
        print("  Please install Ollama from https://ollama.ai/download")
        return False
    except subprocess.TimeoutExpired:
        print("  ❌ Ollama command timed out")
        return False

def test_ollama_server():
    """Test if Ollama server is running."""
    print("\n🔍 Testing Ollama server...")
    
    try:
        # Test if Ollama server is responding
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("  ✅ Ollama server is running")
            return True
        else:
            print(f"  ❌ Ollama server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to Ollama server")
        print("  Please start Ollama server with: ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("  ❌ Ollama server connection timed out")
        return False

def test_ollama_models():
    """Test if required models are available."""
    print("\n🔍 Testing Ollama models...")
    
    try:
        # Get list of available models
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            required_models = ['qwen2.5:3b']
            missing_models = []
            
            for model in required_models:
                if model in available_models:
                    print(f"  ✅ {model}")
                else:
                    print(f"  ❌ {model} - NOT FOUND")
                    missing_models.append(model)
            
            if missing_models:
                print(f"\n❌ Missing models: {', '.join(missing_models)}")
                print("Please pull missing models with:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
                return False
            else:
                print("✅ All required models are available!")
                return True
        else:
            print(f"  ❌ Failed to get model list: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Error checking models: {str(e)}")
        return False

def test_database_creation():
    """Test if database can be created."""
    print("\n🔍 Testing database creation...")
    
    try:
        from database import DatabaseManager
        
        # Try to create a database manager
        db_manager = DatabaseManager("test.db")
        print("  ✅ Database manager created successfully")
        
        # Clean up test database
        import os
        if os.path.exists("test.db"):
            os.remove("test.db")
        
        return True
    except Exception as e:
        print(f"  ❌ Database creation failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Document Chat with RAG - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Dependencies", test_python_dependencies),
        ("Ollama Installation", test_ollama_installation),
        ("Ollama Server", test_ollama_server),
        ("Ollama Models", test_ollama_models),
        ("Database Creation", test_database_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("You can now run the application with: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the application.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 