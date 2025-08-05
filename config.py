"""
Configuration file for GuardAI Deepfake Detection Application
"""

import os
from pathlib import Path

# Application Configuration
APP_CONFIG = {
    'name': 'GuardAI Deepfake Detection',
    'version': '2.0.0',
    'description': 'Advanced AI-powered deepfake detection with comprehensive analysis',
    'author': 'GuardAI Team',
    'contact': 'support@guardai.com'
}

# Model Configuration
MODEL_CONFIG = {
    'default_model': 'efficientnet_b4',
    'sequence_length': 30,
    'image_size': 224,
    'batch_size': 1,
    'num_classes': 2,
    'latent_dim': 2048,
    'lstm_layers': 2,
    'hidden_dim': 2048,
    'bidirectional': True,
    'dropout_rate': 0.3,
    'attention_heads': 8
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'face_detection': True,
    'face_crop_margin': 0.1,
    'min_face_size': 50,
    'max_faces_per_frame': 1
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'confidence_threshold': 0.7,
    'entropy_threshold': 0.5,
    'min_frames_required': 10,
    'max_frames_analyzed': 100,
    'frame_sampling_strategy': 'uniform',  # 'uniform', 'random', 'keyframe'
    'enable_attention_visualization': True,
    'enable_frame_analysis': True,
    'enable_metadata_extraction': True
}

# UI Configuration
UI_CONFIG = {
    'theme': 'light',
    'language': 'en',
    'show_advanced_options': True,
    'auto_save_reports': True,
    'enable_animations': True,
    'max_file_size_mb': 100,
    'supported_formats': ['mp4', 'avi', 'mov', 'mkv', 'webm']
}

# Report Configuration
REPORT_CONFIG = {
    'include_visualizations': True,
    'include_metadata': True,
    'include_model_info': True,
    'include_performance_metrics': True,
    'report_format': 'json',  # 'json', 'pdf', 'html'
    'auto_generate_pdf': False,
    'include_attention_maps': True,
    'include_frame_analysis': True
}

# Security Configuration
SECURITY_CONFIG = {
    'enable_encryption': False,
    'max_file_upload_size': 100 * 1024 * 1024,  # 100MB
    'allowed_file_types': ['video'],
    'enable_audit_logging': True,
    'session_timeout_minutes': 30,
    'max_concurrent_uploads': 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_gpu_acceleration': True,
    'max_memory_usage_gb': 8,
    'enable_batch_processing': True,
    'max_batch_size': 10,
    'enable_caching': True,
    'cache_duration_hours': 24,
    'enable_parallel_processing': True,
    'max_worker_threads': 4
}

# Database Configuration (for future use)
DATABASE_CONFIG = {
    'enabled': False,
    'type': 'sqlite',  # 'sqlite', 'postgresql', 'mysql'
    'host': 'localhost',
    'port': 5432,
    'database': 'guardai_db',
    'username': '',
    'password': '',
    'connection_pool_size': 10
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_enabled': True,
    'file_path': 'logs/guardai.log',
    'max_file_size_mb': 10,
    'backup_count': 5,
    'console_enabled': True
}

# API Configuration (for future use)
API_CONFIG = {
    'enabled': False,
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'cors_enabled': True,
    'rate_limit_enabled': True,
    'rate_limit_requests': 100,
    'rate_limit_window': 3600  # 1 hour
}

# Paths Configuration
PATHS_CONFIG = {
    'models_dir': 'trained-models',
    'reports_dir': 'reports',
    'logs_dir': 'logs',
    'temp_dir': 'temp',
    'uploads_dir': 'uploads',
    'cache_dir': 'cache',
    'sessions_dir': 'sessions'
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    for path_name, path_value in PATHS_CONFIG.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)

# Environment-specific configurations
def get_environment_config():
    """Get configuration based on environment"""
    env = os.getenv('GUARDAI_ENV', 'development')
    
    if env == 'production':
        return {
            'debug': False,
            'logging_level': 'WARNING',
            'enable_gpu_acceleration': True,
            'max_file_upload_size': 50 * 1024 * 1024,  # 50MB
            'enable_audit_logging': True,
            'session_timeout_minutes': 15
        }
    elif env == 'staging':
        return {
            'debug': True,
            'logging_level': 'INFO',
            'enable_gpu_acceleration': True,
            'max_file_upload_size': 100 * 1024 * 1024,  # 100MB
            'enable_audit_logging': True,
            'session_timeout_minutes': 30
        }
    else:  # development
        return {
            'debug': True,
            'logging_level': 'DEBUG',
            'enable_gpu_acceleration': False,
            'max_file_upload_size': 200 * 1024 * 1024,  # 200MB
            'enable_audit_logging': False,
            'session_timeout_minutes': 60
        }

# Model-specific configurations
MODEL_SPECIFIC_CONFIG = {
    'efficientnet_b4': {
        'pretrained': True,
        'feature_dim': 1792,
        'optimizer': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    },
    'resnext50_32x4d': {
        'pretrained': True,
        'feature_dim': 2048,
        'optimizer': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    },
    'resnet50': {
        'pretrained': True,
        'feature_dim': 2048,
        'optimizer': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    }
}

# Feature flags for experimental features
FEATURE_FLAGS = {
    'enable_real_time_analysis': False,
    'enable_advanced_visualizations': True,
    'enable_model_comparison': True,
    'enable_batch_processing': True,
    'enable_pdf_reports': True,
    'enable_session_management': True,
    'enable_performance_monitoring': True,
    'enable_auto_model_selection': False,
    'enable_attention_visualization': True,
    'enable_frame_metadata_extraction': True
}

# Validation rules
VALIDATION_RULES = {
    'min_video_duration': 1.0,  # seconds
    'max_video_duration': 300.0,  # seconds (5 minutes)
    'min_resolution': (64, 64),
    'max_resolution': (4096, 4096),
    'min_fps': 1,
    'max_fps': 120,
    'allowed_codecs': ['avc1', 'h264', 'mp4v', 'xvid', 'divx'],
    'max_file_size_bytes': 100 * 1024 * 1024  # 100MB
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds maximum allowed size of {max_size}MB',
    'unsupported_format': 'Unsupported video format. Please use: {formats}',
    'no_faces_detected': 'No faces detected in the video. Please ensure the video contains clear facial features.',
    'video_too_short': 'Video is too short. Minimum duration is {min_duration} seconds.',
    'video_too_long': 'Video is too long. Maximum duration is {max_duration} seconds.',
    'model_not_found': 'Selected model not found. Please check model files.',
    'processing_error': 'Error during video processing: {error}',
    'insufficient_memory': 'Insufficient memory to process video. Try a shorter video or lower resolution.',
    'gpu_not_available': 'GPU acceleration not available. Processing will continue on CPU.',
    'invalid_sequence_length': 'Invalid sequence length. Must be between {min} and {max}.'
}

# Success messages
SUCCESS_MESSAGES = {
    'analysis_complete': 'Video analysis completed successfully!',
    'report_generated': 'Report generated and saved successfully!',
    'batch_complete': 'Batch processing completed successfully!',
    'model_loaded': 'Model loaded successfully!',
    'settings_saved': 'Settings saved successfully!',
    'session_saved': 'Analysis session saved successfully!'
}

# Initialize configuration
def initialize_config():
    """Initialize application configuration"""
    create_directories()
    env_config = get_environment_config()
    
    # Update configurations with environment-specific settings
    for key, value in env_config.items():
        if key == 'logging_level':
            LOGGING_CONFIG['level'] = value
        elif key == 'max_file_upload_size':
            SECURITY_CONFIG['max_file_upload_size'] = value
        elif key == 'session_timeout_minutes':
            SECURITY_CONFIG['session_timeout_minutes'] = value
        elif key == 'enable_gpu_acceleration':
            PERFORMANCE_CONFIG['enable_gpu_acceleration'] = value
        elif key == 'enable_audit_logging':
            SECURITY_CONFIG['enable_audit_logging'] = value
        elif key == 'debug':
            API_CONFIG['debug'] = value

# Export all configurations
__all__ = [
    'APP_CONFIG',
    'MODEL_CONFIG', 
    'PREPROCESSING_CONFIG',
    'ANALYSIS_CONFIG',
    'UI_CONFIG',
    'REPORT_CONFIG',
    'SECURITY_CONFIG',
    'PERFORMANCE_CONFIG',
    'DATABASE_CONFIG',
    'LOGGING_CONFIG',
    'API_CONFIG',
    'PATHS_CONFIG',
    'MODEL_SPECIFIC_CONFIG',
    'FEATURE_FLAGS',
    'VALIDATION_RULES',
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    'initialize_config',
    'create_directories',
    'get_environment_config'
] 