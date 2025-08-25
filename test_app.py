"""
Test script for GuardAI Deepfake Detection Application
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import face_recognition
        print("✅ Face recognition imported successfully")
    except ImportError as e:
        print(f"❌ Face recognition import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if the advanced model can be created"""
    print("\nTesting model creation...")
    
    try:
        from app import Model
        
        # Create model
        model = Model(num_classes=2)
        print("✅ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size, seq_length, channels, height, width = 1, 10, 3, 224, 224
        dummy_input = torch.randn(batch_size, seq_length, channels, height, width)
        
        with torch.no_grad():
            fmap, logits = model(dummy_input)
        
        print(f"✅ Model forward pass successful")
        print(f"   - Feature map shape: {fmap.shape}")
        print(f"   - Logits shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_dataset_creation():
    """Test if the dataset class can be created"""
    print("\nTesting dataset creation...")
    
    try:
        from app import AdvancedValidationDataset
        
        # Create a dummy video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            # Create a simple video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_file.name, fourcc, 20.0, (640, 480))
            
            # Add some frames
            for i in range(30):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
            
            # Test dataset creation
            transforms_compose = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            dataset = AdvancedValidationDataset(
                tmp_file.name,
                sequence_length=10,
                transform=transforms_compose,
                face_detection=False  # Disable face detection for testing
            )
            
            print("✅ AdvancedValidationDataset created successfully")
            
            # Test getting an item
            frames, metadata = dataset[0]
            print(f"✅ Dataset item retrieval successful")
            print(f"   - Frames shape: {frames.shape}")
            print(f"   - Metadata length: {len(metadata)}")
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return True
            
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False

def test_prediction_function():
    """Test the prediction function"""
    print("\nTesting prediction function...")
    
    try:
        from app import advanced_predict, Model
        
        # Create model and dummy data
        model = Model(num_classes=2)
        batch_size, seq_length, channels, height, width = 1, 10, 3, 224, 224
        dummy_frames = torch.randn(batch_size, seq_length, channels, height, width)
        
        # Test prediction
        device = torch.device("cpu")
        results = advanced_predict(model, dummy_frames, device)
        
        print("✅ Prediction function successful")
        print(f"   - Prediction: {results['prediction']}")
        print(f"   - Confidence: {results['confidence']:.2f}%")
        print(f"   - Entropy: {results['entropy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction function failed: {e}")
        return False

def test_visualization_functions():
    """Test visualization functions"""
    print("\nTesting visualization functions...")
    
    try:
        from app import create_visualization_plots
        
        # Create dummy prediction results
        dummy_results = {
            'prediction': 1,
            'confidence': 85.5,
            'probabilities': np.array([[0.15, 0.85]]),
            'entropy': 0.3,
            'features': np.random.rand(1, 2048, 7, 7)
        }
        
        # Create dummy metadata
        dummy_metadata = [
            {
                'timestamp': i * 0.1,
                'brightness': np.random.uniform(50, 200),
                'contrast': np.random.uniform(10, 50),
                'face_detected': True
            }
            for i in range(10)
        ]
        
        # Test visualization creation
        plots = create_visualization_plots(dummy_results, dummy_metadata)
        
        print("✅ Visualization functions successful")
        print(f"   - Number of plots created: {len(plots)}")
        print(f"   - Plot types: {list(plots.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization functions failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from config import (
            APP_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG,
            ANALYSIS_CONFIG, UI_CONFIG, REPORT_CONFIG,
            create_directories, get_environment_config
        )
        
        print("✅ Configuration imports successful")
        
        # Test directory creation
        create_directories()
        print("✅ Directory creation successful")
        
        # Test environment config
        env_config = get_environment_config()
        print(f"✅ Environment config retrieved: {list(env_config.keys())}")
        
        # Test key configurations
        print(f"   - App name: {APP_CONFIG['name']}")
        print(f"   - Model sequence length: {MODEL_CONFIG['sequence_length']}")
        print(f"   - UI theme: {UI_CONFIG['theme']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import (
            create_analysis_summary, save_analysis_session,
            load_analysis_session, create_performance_metrics
        )
        
        # Create dummy reports
        dummy_reports = [
            {
                'prediction': {'result': 'REAL', 'confidence': 85.5},
                'model_info': {'name': 'test_model.pt'},
                'timestamp': '2024-01-01T00:00:00'
            },
            {
                'prediction': {'result': 'FAKE', 'confidence': 92.3},
                'model_info': {'name': 'test_model.pt'},
                'timestamp': '2024-01-01T01:00:00'
            }
        ]
        
        # Test summary creation
        summary = create_analysis_summary(dummy_reports)
        print("✅ Analysis summary creation successful")
        print(f"   - Total analyses: {summary['total_analyses']}")
        print(f"   - Real count: {summary['real_count']}")
        print(f"   - Fake count: {summary['fake_count']}")
        
        # Test session saving/loading
        session_file = save_analysis_session(dummy_reports, "test_session")
        print(f"✅ Session saved to: {session_file}")
        
        loaded_session = load_analysis_session(session_file)
        print("✅ Session loading successful")
        
        # Test performance metrics
        metrics = create_performance_metrics(dummy_reports)
        print("✅ Performance metrics creation successful")
        
        # Clean up
        if os.path.exists(session_file):
            os.unlink(session_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 GuardAI Deepfake Detection - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_dataset_creation,
        test_prediction_function,
        test_visualization_functions,
        test_configuration,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nMake sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 