import os

class Config:
    """Configuration settings for the fraud detection application"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fraud-detection-secret-key-2024'
    DEBUG = True
    
    # Model settings
    MODEL_PATH = 'models/'
    DATA_PATH = 'data/'
    
    # Model parameters (from original research)
    MODEL_CONFIG = {
        'architecture': [128, 64, 32, 1],
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'epochs': 50,
        'batch_size': 512,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    }
    
    # Feature engineering settings
    NUMERICAL_FEATURES = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest'
    ]
    
    CATEGORICAL_FEATURES = ['type']
    
    TRANSACTION_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    
    # Fraud detection thresholds (based on research findings)
    FRAUD_THRESHOLD = 0.5
    HIGH_RISK_THRESHOLD = 0.3
    LOW_RISK_THRESHOLD = 0.1
    
    # Business rules (from original research insights)
    HIGH_RISK_TYPES = ['CASH_OUT', 'TRANSFER']  # Most fraud occurs here
    FRAUD_AMOUNT_RANGE = (130000, 360000)  # ₹1.3L - ₹3.6L high fraud range
    
    # Performance metrics from trained model
    MODEL_METRICS = {
        'accuracy': 0.9870,
        'precision_fraud': 0.09,
        'recall_fraud': 0.99,
        'f1_score_fraud': 0.16,
        'precision_legitimate': 1.00,
        'recall_legitimate': 0.99,
        'f1_score_legitimate': 0.99
    }
    
    # Database settings (for future expansion)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///fraud_detection.db'
    
    # API settings
    MAX_REQUESTS_PER_MINUTE = 100
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/fraud_detection.log'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    # Add production database URL
    DATABASE_URL = os.environ.get('DATABASE_URL')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}