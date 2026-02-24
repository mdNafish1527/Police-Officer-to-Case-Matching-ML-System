"""
Police Officer-to-Case Matching ML System
Main execution script
"""


import os
import sys
from data.generate_data import generate_officers, generate_cases, generate_assignments
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator


def create_directories():
    """Create necessary directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    print("✅ Directories created/verified")


def interactive_recommendation(feature_engineer, cases_df, officers_df):
    """Interactive officer recommendation based on user input"""

    print("\n" + "="*80)
    print("        🚔 POLICE OFFICER RECOMMENDATION SYSTEM 🚔")
    print("="*80)

    # Show available options to guide the user
    print("\n📋 Available Case Types :", ', '.join(sorted(cases_df['crime_type'].unique())))
    print("📍 Available Locations  :", ', '.join(sorted(officers_df['region'].unique())))
    print("⚡ Available Priorities : Low, Medium, High, Critical")

    while True:
        print("\n" + "-"*80)

        # --- Input: Case Type ---
        while True:
            case_type = input("\n🔍 Enter Case Type  : ").strip()
            if case_type:
                break
            print("   ⚠️  Case type cannot be empty. Please try again.")

        # --- Input: Location ---
        while True:
            location = input("📍 Enter Location   : ").strip()
            if location:
                break
            print("   ⚠️  Location cannot be empty. Please try again.")

        # --- Input: Priority ---
        valid_priorities = ['low', 'medium', 'high', 'critical']
        while True:
            priority = input("⚡ Enter Priority   : ").strip()
            if priority.lower() in valid_priorities:
                break
            print("   ⚠️  Invalid priority. Please choose from: Low, Medium, High, Critical")

        # Map priority to complexity score automatically
        priority_to_complexity = {
            'low'     : 2,
            'medium'  : 5,
            'high'    : 8,
            'critical': 10
        }
        complexity = priority_to_complexity[priority.lower()]

        # Build the case dictionary
        new_case = {
            'case_id'          : 'INPUT001',
            'crime_type'       : case_type,
            'location'         : location,
            'complexity_score' : complexity,
            'language_required': 'English',
            'urgency'          : priority.capitalize()
        }

        # Display case summary
        print("\n" + "="*80)
        print("📋 CASE SUMMARY")
        print("="*80)
        print(f"   Case Type  : {case_type}")
        print(f"   Location   : {location}")
        print(f"   Priority   : {priority.capitalize()}")
        print(f"   Complexity : {complexity}/10")

        # Get ranked officers
        try:
            ranked_officers = feature_engineer.rank_officers_for_case(new_case, top_n=5)
            print(f"\n🏆 TOP 5 RECOMMENDED OFFICERS")
            print("="*80)
            print(ranked_officers.to_string())
            print("="*80)
        except Exception as e:
            print(f"\n⚠️  Could not rank officers: {e}")
            print("   Please check that the case type and location are valid.")

        # Ask if user wants to search again
        print()
        again = input("🔄 Search for another case? (yes / no) : ").strip().lower()
        if again not in ['yes', 'y']:
            print("\n" + "="*80)
            print("   ✅ Thank you for using the Officer Recommendation System!")
            print("="*80 + "\n")
            break


def main():
    print("\n" + "="*80)
    print("       POLICE OFFICER-TO-CASE MATCHING ML SYSTEM")
    print("="*80)

    # Step 1: Create directories
    print("\n📁 STEP 1: Creating directories...")
    create_directories()

    # Step 2: Generate synthetic data
    print("\n📊 STEP 2: Generating synthetic dataset...")
    print("-"*80)

    officers_df    = generate_officers(100)
    cases_df       = generate_cases(500)
    assignments_df = generate_assignments(officers_df, cases_df, 400)

    officers_df.to_csv('data/officers.csv', index=False)
    cases_df.to_csv('data/cases.csv', index=False)
    assignments_df.to_csv('data/assignments_history.csv', index=False)

    # Step 3: Data Preprocessing
    print("\n🔧 STEP 3: Preprocessing data...")
    print("-"*80)

    preprocessor = DataPreprocessor()
    officers_df, cases_df, assignments_df = preprocessor.load_data(
        'data/officers.csv',
        'data/cases.csv',
        'data/assignments_history.csv'
    )

    preprocessor.explore_data()

    X, y, training_data = preprocessor.create_training_dataset()
    X_scaled = preprocessor.scale_features(X, fit=True)

    preprocessor.save_preprocessor()

    # Step 4: Feature Engineering
    print("\n⚙️  STEP 4: Engineering features...")
    print("-"*80)

    feature_engineer  = FeatureEngineer(officers_df, cases_df)
    officers_featured = feature_engineer.create_officer_features()
    cases_featured    = feature_engineer.create_case_features()

    print(f"✅ Officer features engineered: {len(officers_featured.columns)} features")
    print(f"✅ Case features engineered: {len(cases_featured.columns)} features")

    # Step 5: Model Training
    print("\n🤖 STEP 5: Training models...")
    print("-"*80)

    trainer = ModelTrainer()
    trainer.split_data(X_scaled, y)

    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()

    # Step 6: Model Comparison and Selection
    print("\n📊 STEP 6: Comparing and selecting best model...")
    print("-"*80)

    results = trainer.compare_models()

    # Step 7: Evaluation
    print("\n📈 STEP 7: Generating evaluation plots...")
    print("-"*80)

    evaluator = ModelEvaluator(trainer.best_model, trainer.X_test, trainer.y_test)
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.plot_prediction_distribution()
    evaluator.generate_report()

    # Step 8: Feature Importance
    if hasattr(trainer.best_model, 'feature_importances_'):
        feature_names = preprocessor.feature_columns
        trainer.plot_feature_importance(trainer.best_model, feature_names)

    # Step 9: Save Model
    print("\n💾 STEP 9: Saving trained model...")
    print("-"*80)

    trainer.save_model()

    # Summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Output Files Generated:")
    print("   - data/officers.csv")
    print("   - data/cases.csv")
    print("   - data/assignments_history.csv")
    print("   - models/best_model.pkl")
    print("   - models/preprocessor.pkl")
    print("   - models/confusion_matrix.png")
    print("   - models/roc_curve.png")
    print("   - models/prediction_distribution.png")
    print("   - models/feature_importance.png")
    print("   - models/evaluation_report.txt")

    
    # Step 10: Interactive Recommendation
    interactive_recommendation(feature_engineer, cases_df, officers_df)


if __name__ == "__main__":
    main()
