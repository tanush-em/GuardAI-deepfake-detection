#!/usr/bin/env python3
"""
Test script for improved model selection logic
"""

import os
import glob
import re

def test_model_selection():
    """Test the improved model selection logic"""
    
    print("🧪 Testing Model Selection Logic...")
    
    # Get actual model files
    model_files = glob.glob(os.path.join("trained-models", "*.pt"))
    
    if not model_files:
        print("❌ No model files found in trained-models/ directory")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    for model_file in model_files:
        print(f"   - {os.path.basename(model_file)}")
    
    # Test the improved extraction logic
    def extract_accuracy(filename):
        try:
            # Extract accuracy from filename like "model_97_acc_100_frames_FF_data.pt"
            parts = os.path.basename(filename).split('_')
            for i, part in enumerate(parts):
                if part == 'acc' and i > 0:
                    return int(parts[i-1])
            # Fallback: try to find any number in the filename
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else 0
        except (ValueError, IndexError):
            return 0
    
    # Test each model file
    print("\n🔍 Testing accuracy extraction:")
    for model_file in model_files:
        accuracy = extract_accuracy(model_file)
        print(f"   {os.path.basename(model_file)} -> Accuracy: {accuracy}")
    
    # Test model selection
    try:
        best_model = max(model_files, key=extract_accuracy)
        best_accuracy = extract_accuracy(best_model)
        print(f"\n✅ Best model selected: {os.path.basename(best_model)} (Accuracy: {best_accuracy})")
    except Exception as e:
        print(f"\n❌ Error selecting best model: {e}")
        return
    
    print("\n🎉 Model selection test completed successfully!")

if __name__ == "__main__":
    test_model_selection()
