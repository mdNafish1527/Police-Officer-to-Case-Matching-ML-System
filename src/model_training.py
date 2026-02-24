import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✅ Data Split Successfully")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Positive class ratio in train: {(self.y_train.sum() / len(self.y_train) * 100):.2f}%")
        print(f"Positive class ratio in test: {(self.y_test.sum() / len(self.y_test) * 100):.2f}%")
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("TRAINING: LOGISTIC REGRESSION")
        print("="*60)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        print("✅ Logistic Regression trained")
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING: RANDOM FOREST")
        print("="*60)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        
        print("🔍 Performing Grid Search...")
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['Random Forest'] = best_model
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV F1 Score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "="*60)
        print("TRAINING: GRADIENT BOOSTING")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = model
        
        print("✅ Gradient Boosting trained")
        
        return model
    
    def evaluate_model(self, model_name, model):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Probabilities for ROC-AUC
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        precision = precision_score(self.y_test, y_pred_test)
        recall = recall_score(self.y_test, y_pred_test)
        f1 = f1_score(self.y_test, y_pred_test)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n📊 Accuracy:")
        print(f"  Train: {train_acc:.4f}")
        print(f"  Test:  {test_acc:.4f}")
        
        print(f"\n📊 Test Set Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred_test, target_names=['Unresolved', 'Resolved']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        metrics = {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def compare_models(self):
        """Compare all models and select the best"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        results = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model_name, model)
            results.append(metrics)
        
        # Find best model by F1-score
        best_idx = np.argmax([r['f1_score'] for r in results])
        self.best_model = self.models[results[best_idx]['model_name']]
        self.best_model_name = results[best_idx]['model_name']
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1-Score: {results[best_idx]['f1_score']:.4f}")
        
        return results
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Feature importance plot saved to models/feature_importance.png")
    
    def save_model(self, path='models/best_model.pkl'):
        """Save trained model"""
        joblib.dump(self.best_model, path)
        print(f"✅ Best model ({self.best_model_name}) saved to {path}")
    
    def load_model(self, path='models/best_model.pkl'):
        """Load trained model"""
        self.best_model = joblib.load(path)
        print(f"✅ Model loaded from {path}")