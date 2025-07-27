import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
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
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        print("🚀 Starting model loading process...")
        
        try:
            # Check current working directory and paths
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Model path setting: {self.model_path}")
            
            # Load the trained model
            model_file = os.path.join(self.model_path, 'fraud_detection_model.h5')
            print(f"🔍 Looking for model at: {os.path.abspath(model_file)}")
            print(f"🔍 Model file exists: {os.path.exists(model_file)}")
            
            if os.path.exists(model_file):
                print("📦 Loading Keras model...")
                self.model = keras_load_model(model_file)
                print("✅ Keras model loaded successfully!")
                print(f"📊 Model input shape: {self.model.input_shape}")
            else:
                print(f"❌ Model file not found at: {model_file}")
                self.model = None
            
            # Load the scaler
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')
            print(f"🔍 Looking for scaler at: {os.path.abspath(scaler_file)}")
            print(f"🔍 Scaler file exists: {os.path.exists(scaler_file)}")
            
            if os.path.exists(scaler_file):
                print("📦 Loading scaler...")
                self.scaler = joblib.load(scaler_file)
                print("✅ Scaler loaded successfully")
            else:
                print(f"❌ Scaler file not found at: {scaler_file}")
                self.scaler = None
            
            # Load feature names
            features_file = os.path.join(self.model_path, 'feature_names.pkl')
            print(f"🔍 Looking for features at: {os.path.abspath(features_file)}")
            print(f"🔍 Features file exists: {os.path.exists(features_file)}")
            
            if os.path.exists(features_file):
                print("📦 Loading feature names...")
                self.feature_names = joblib.load(features_file)
                print("✅ Feature names loaded successfully")
                print(f"📊 Number of features: {len(self.feature_names)}")
                print(f"📊 Feature names: {self.feature_names}")
            else:
                print(f"❌ Feature names file not found at: {features_file}")
                # Use default feature names if file doesn't exist
                self.feature_names = [
                    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud',
                    'balance_diff_orig', 'balance_diff_dest',
                    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                    'type_PAYMENT', 'type_TRANSFER'
                ]
                print("⚠️ Using default feature names")
            
            # Mark as loaded if model exists
            self.is_loaded = (self.model is not None)
            print(f"🎯 Final loading status: {self.is_loaded}")
            
            if self.is_loaded:
                print("🎉 Deep Neural Network model ready for predictions!")
            else:
                print("⚠️ Falling back to rule-based prediction system")
            
        except Exception as e:
            print(f"💥 CRITICAL ERROR during model loading:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # Print full traceback for debugging
            import traceback
            print("📋 Full error traceback:")
            traceback.print_exc()
            
            self.is_loaded = False
            self.model = None
            self.scaler = None
            print("🔄 Falling back to rule-based prediction")
    
    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction for prediction
        
        Args:
            transaction_data (dict): Transaction data with keys:
                - type: Transaction type
                - amount: Transaction amount
                - oldbalanceOrg: Origin old balance
                - newbalanceOrig: Origin new balance
                - oldbalanceDest: Destination old balance
                - newbalanceDest: Destination new balance
                - isFlaggedFraud: Whether flagged by system
                - step: Transaction step/time
        
        Returns:
            np.array: Preprocessed features ready for model
        """
        try:
            print(f"🔄 Preprocessing transaction: {transaction_data.get('type', 'Unknown')} ${transaction_data.get('amount', 0):,}")
            
            # Create feature vector
            features = {}
            
            # Basic features
            features['step'] = transaction_data.get('step', 1)
            features['amount'] = transaction_data.get('amount', 0)
            features['oldbalanceOrg'] = transaction_data.get('oldbalanceOrg', 0)
            features['newbalanceOrig'] = transaction_data.get('newbalanceOrig', 0)
            features['oldbalanceDest'] = transaction_data.get('oldbalanceDest', 0)
            features['newbalanceDest'] = transaction_data.get('newbalanceDest', 0)
            features['isFlaggedFraud'] = transaction_data.get('isFlaggedFraud', 0)
            
            # Engineered features
            features['balance_diff_orig'] = features['oldbalanceOrg'] - features['newbalanceOrig']
            features['balance_diff_dest'] = features['newbalanceDest'] - features['oldbalanceDest']
            
            # One-hot encoding for transaction type
            transaction_type = transaction_data.get('type', 'PAYMENT')
            features['type_CASH_IN'] = 1 if transaction_type == 'CASH_IN' else 0
            features['type_CASH_OUT'] = 1 if transaction_type == 'CASH_OUT' else 0
            features['type_DEBIT'] = 1 if transaction_type == 'DEBIT' else 0
            features['type_PAYMENT'] = 1 if transaction_type == 'PAYMENT' else 0
            features['type_TRANSFER'] = 1 if transaction_type == 'TRANSFER' else 0
            
            # Convert to DataFrame for consistent processing
            df = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            print(f"📊 Feature vector shape: {df.shape}")
            
            # Scale numerical features if scaler is available
            if self.scaler:
                print("⚖️ Applying feature scaling...")
                numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                    'oldbalanceDest', 'newbalanceDest', 'balance_diff_orig', 'balance_diff_dest']
                numerical_features = [col for col in numerical_features if col in df.columns]
                df[numerical_features] = self.scaler.transform(df[numerical_features])
                print("✅ Feature scaling applied")
            else:
                print("⚠️ No scaler available - using raw features")
            
            return df.values
            
        except Exception as e:
            print(f"❌ Error preprocessing transaction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_fraud_probability(self, transaction_data):
        """
        Predict fraud probability for a transaction
        
        Args:
            transaction_data (dict): Transaction data
            
        Returns:
            dict: Prediction results with probability, classification, and confidence
        """
        print(f"\n🎯 Making fraud prediction for {transaction_data.get('type', 'Unknown')} transaction")
        
        if not self.is_loaded:
            print("🔄 Model not loaded, attempting to load...")
            self.load_model()
        
        try:
            # Preprocess the transaction
            processed_features = self.preprocess_transaction(transaction_data)
            
            if processed_features is None:
                print("❌ Preprocessing failed, using fallback prediction")
                return self._fallback_prediction(transaction_data)
            
            # Make prediction using the model
            if self.model and self.is_loaded:
                print("🧠 Using Deep Neural Network for prediction...")
                probability = self.model.predict(processed_features, verbose=0)[0][0]
                confidence = min(0.99, 0.85 + abs(probability - 0.5) * 0.28)  # Higher confidence for extreme predictions
                
                print(f"📊 Neural network probability: {probability:.4f}")
                print(f"📊 Confidence score: {confidence:.4f}")
                
                # Classify based on probability
                if probability >= 0.5:
                    classification = 'FRAUD'
                    risk_level = 'High'
                elif probability >= 0.3:
                    classification = 'SUSPICIOUS' 
                    risk_level = 'Medium'
                else:
                    classification = 'LEGITIMATE'
                    risk_level = 'Low'
                
                result = {
                    'probability': float(probability),
                    'classification': classification,
                    'risk_level': risk_level,
                    'confidence': float(confidence),
                    'model_used': 'Deep Neural Network',
                    'features_used': self.feature_names
                }
                
                print(f"🎯 Prediction result: {classification} (probability: {probability:.3f})")
                return result
                
            else:
                print("⚠️ Neural network model not available, using fallback")
                return self._fallback_prediction(transaction_data)
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to rule-based prediction")
            return self._fallback_prediction(transaction_data)
    
    def _fallback_prediction(self, transaction_data):
        """
        Fallback prediction when model is not available
        Uses rule-based approach based on research findings
        """
        print("🔄 Using rule-based fallback prediction...")
        
        try:
            # Rule-based prediction based on research insights
            risk_score = 0.0
            
            transaction_type = transaction_data.get('type', 'PAYMENT')
            amount = transaction_data.get('amount', 0)
            
            print(f"📊 Analyzing: {transaction_type} transaction of ${amount:,}")
            
            # High-risk transaction types (CASH_OUT, TRANSFER)
            if transaction_type in ['CASH_OUT', 'TRANSFER']:
                risk_score += 0.4
                print("⚠️ High-risk transaction type detected")
            
            # Amount in high-fraud range (100k - 400k based on research)
            if 100000 <= amount <= 400000:
                risk_score += 0.3
                print("⚠️ Amount in high-fraud range")
            elif amount > 1000000:  # Very high amounts
                risk_score += 0.2
                print("⚠️ Very high amount detected")
            
            # System flagged
            if transaction_data.get('isFlaggedFraud', 0) == 1:
                risk_score += 0.2
                print("⚠️ Transaction flagged by system")
            
            # Balance inconsistencies
            old_bal_orig = transaction_data.get('oldbalanceOrg', 0)
            new_bal_orig = transaction_data.get('newbalanceOrig', 0)
            
            if old_bal_orig > 0 and new_bal_orig == 0:  # Account drained
                risk_score += 0.3
                print("🚨 Account drainage detected")
            
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
            
            result = {
                'probability': risk_score,
                'classification': classification,
                'risk_level': risk_level,
                'confidence': 0.75,  # Lower confidence for rule-based
                'model_used': 'rule_based_fallback',
                'message': 'Using business rules based on research findings'
            }
            
            print(f"🎯 Rule-based result: {classification} (score: {risk_score:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ Error in fallback prediction: {e}")
            return {
                'probability': 0.5,
                'classification': 'UNKNOWN',
                'risk_level': 'Medium',
                'confidence': 0.5,
                'model_used': 'error_fallback',
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model and self.is_loaded:
            return {
                'model_type': 'Deep Neural Network',
                'input_features': len(self.feature_names) if self.feature_names else 'Unknown',
                'architecture': '128→64→32→1',
                'activation': 'ReLU + Sigmoid',
                'is_loaded': self.is_loaded,
                'features': self.feature_names,
                'model_summary': f"Input shape: {self.model.input_shape if self.model else 'N/A'}"
            }
        else:
            return {
                'model_type': 'Rule-based Fallback',
                'input_features': len(self.feature_names) if self.feature_names else 14,
                'is_loaded': False,
                'message': 'Using business rules based on research findings',
                'fallback_features': [
                    'Transaction type analysis',
                    'Amount range detection', 
                    'Balance consistency check',
                    'System flag integration'
                ]
            }

# Global model instance
fraud_model = FraudDetectionModel()

# Test function
def test_model():
    """Test the model with sample data"""
    print("🧪 Testing Fraud Detection Model...")
    
    # Load model
    fraud_model.load_model()
    
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
        result = fraud_model.predict_fraud_probability(transaction)
        print(f"   Type: {transaction['type']}")
        print(f"   Amount: ${transaction['amount']:,}")
        print(f"   Result: {result['classification']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model Used: {result.get('model_used', 'Unknown')}")

if __name__ == "__main__":
    test_model()