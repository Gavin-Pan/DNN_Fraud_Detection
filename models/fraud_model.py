import numpy as np
import pandas as pd
import os

class FraudDetectionModel:
    """
    Wrapper class for the fraud detection model
    Handles loading, preprocessing, and prediction
    """
    
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        print("✅ FraudDetectionModel initialized")
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # For now, we'll use a fallback since we don't have trained models yet
            print("⚠️ Using rule-based fallback model (trained model not found)")
            self.feature_names = [
                'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
                'balance_diff_orig', 'balance_diff_dest',
                'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                'type_PAYMENT', 'type_TRANSFER'
            ]
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def predict_fraud_probability(self, transaction_data):
        """
        Predict fraud probability for a transaction
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Rule-based prediction using research insights
            risk_score = 0.0
            
            transaction_type = transaction_data.get('type', 'PAYMENT')
            amount = transaction_data.get('amount', 0)
            
            # High-risk transaction types (CASH_OUT, TRANSFER)
            if transaction_type in ['CASH_OUT', 'TRANSFER']:
                risk_score += 0.4
            
            # Amount in high-fraud range (100k - 400k based on research)
            if 100000 <= amount <= 400000:
                risk_score += 0.3
            elif amount > 1000000:  # Very high amounts
                risk_score += 0.2
            
            # System flagged
            if transaction_data.get('isFlaggedFraud', 0) == 1:
                risk_score += 0.2
            
            # Balance inconsistencies
            old_bal_orig = transaction_data.get('oldbalanceOrg', 0)
            new_bal_orig = transaction_data.get('newbalanceOrig', 0)
            
            if old_bal_orig > 0 and new_bal_orig == 0:  # Account drained
                risk_score += 0.3
            
            # Add some controlled randomness
            risk_score += np.random.uniform(-0.05, 0.05)
            risk_score = max(0.0, min(1.0, risk_score))
            
            # Classification
            if risk_score >= 0.5:
                classification = 'FRAUD'
                risk_level = 'High'
            elif risk_score >= 0.3:
                classification = 'SUSPICIOUS'
                risk_level = 'Medium'
            else:
                classification = 'LEGITIMATE'
                risk_level = 'Low'
            
            return {
                'probability': risk_score,
                'classification': classification,
                'risk_level': risk_level,
                'confidence': 0.75,  # Lower confidence for rule-based
                'model_used': 'rule_based_fallback'
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return {
                'probability': 0.5,
                'classification': 'UNKNOWN',
                'risk_level': 'Medium',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_type': 'Rule-based Fallback',
            'input_features': len(self.feature_names) if self.feature_names else 0,
            'is_loaded': self.is_loaded,
            'features': self.feature_names or [],
            'message': 'Using business rules based on research findings'
        }

# Test function
def test_model():
    """Test the model with sample data"""
    print("🧪 Testing Fraud Detection Model...")
    
    # Initialize model
    model = FraudDetectionModel()
    model.load_model()
    
    # Test transactions
    test_transactions = [
        {
            'type': 'TRANSFER',
            'amount': 250000,
            'oldbalanceOrg': 500000,
            'newbalanceOrig': 250000,
            'oldbalanceDest': 100000,
            'newbalanceDest': 350000,
            'isFlaggedFraud': 0,
            'step': 150
        },
        {
            'type': 'PAYMENT',
            'amount': 1000,
            'oldbalanceOrg': 50000,
            'newbalanceOrig': 49000,
            'oldbalanceDest': 0,
            'newbalanceDest': 1000,
            'isFlaggedFraud': 0,
            'step': 100
        }
    ]
    
    for i, transaction in enumerate(test_transactions):
        print(f"\n📊 Test Transaction {i+1}:")
        result = model.predict_fraud_probability(transaction)
        print(f"   Type: {transaction['type']}")
        print(f"   Amount: ${transaction['amount']:,}")
        print(f"   Result: {result['classification']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    test_model()