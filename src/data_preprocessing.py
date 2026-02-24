import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_data(self, officers_path, cases_path, assignments_path):
        """Load data from CSV files"""
        self.officers_df = pd.read_csv(officers_path)
        self.cases_df = pd.read_csv(cases_path)
        self.assignments_df = pd.read_csv(assignments_path)
        
        print(f"✅ Data Loaded Successfully")
        print(f"Officers: {len(self.officers_df)}, Cases: {len(self.cases_df)}, Assignments: {len(self.assignments_df)}")
        
        return self.officers_df, self.cases_df, self.assignments_df
    
    def explore_data(self):
        """Display data statistics"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\n📊 OFFICERS DATA:")
        print(self.officers_df.info())
        print(self.officers_df.describe())
        
        print("\n📊 CASES DATA:")
        print(self.cases_df.info())
        print(self.cases_df.describe())
        
        print("\n📊 ASSIGNMENTS DATA:")
        print(self.assignments_df.info())
        print(f"Resolution Rate: {(self.assignments_df['resolved'].sum() / len(self.assignments_df) * 100):.2f}%")
    
    def create_training_dataset(self):
        """Create training dataset from assignments"""
        training_data = self.assignments_df.copy()
        
        # Encode categorical variables
        categorical_cols = ['crime_type', 'case_urgency', 'case_location', 'officer_region', 'outcome']
        
        for col in categorical_cols:
            le = LabelEncoder()
            training_data[col + '_encoded'] = le.fit_transform(training_data[col])
            self.label_encoders[col] = le
        
        # Select features for training
        feature_cols = [
            'officer_experience', 'case_complexity', 'location_match', 
            'language_match', 'crime_type_encoded', 'case_urgency_encoded',
            'case_location_encoded', 'officer_region_encoded'
        ]
        
        X = training_data[feature_cols]
        y = training_data['resolved'].astype(int)
        
        self.feature_columns = feature_cols
        
        print(f"\n✅ Training Dataset Created")
        print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        print(f"Target Distribution:\n{y.value_counts()}")
        
        return X, y, training_data
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save preprocessing objects"""
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"✅ Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load preprocessing objects"""
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"✅ Preprocessor loaded from {path}")