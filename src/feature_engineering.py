import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, officers_df, cases_df):
        self.officers_df = officers_df
        self.cases_df = cases_df
    
    def create_officer_features(self):
        """Engineer features for officers"""
        officer_features = self.officers_df.copy()
        
        # Experience level categories
        officer_features['experience_level'] = pd.cut(
            officer_features['years_experience'],
            bins=[0, 5, 10, 20, 30],
            labels=['Junior', 'Mid-Level', 'Senior', 'Expert']
        )
        
        # Success rate categories
        officer_features['success_category'] = pd.cut(
            officer_features['success_rate'],
            bins=[0, 0.5, 0.7, 0.9, 1.0],
            labels=['Low', 'Medium', 'High', 'Expert']
        )
        
        # Workload score
        officer_features['workload_score'] = (officer_features['cases_assigned'] / 
                                              officer_features['cases_assigned'].max()) * 10
        
        # Officer strength score
        officer_features['officer_strength_score'] = (
            (officer_features['years_experience'] / officer_features['years_experience'].max() * 0.3) +
            (officer_features['success_rate'] * 0.4) +
            ((10 - officer_features['workload_score']) / 10 * 0.3)
        ) * 100
        
        # Count of special units and languages
        officer_features['num_special_units'] = officer_features['special_units'].str.split('|').str.len()
        officer_features['num_languages'] = officer_features['languages'].str.split('|').str.len()
        
        # Availability score
        officer_features['availability_score'] = officer_features['availability'].astype(int) * 10
        
        return officer_features
    
    def create_case_features(self):
        """Engineer features for cases"""
        case_features = self.cases_df.copy()
        
        # Urgency score
        urgency_mapping = {'Low': 1, 'Medium': 5, 'High': 8, 'Critical': 10}
        case_features['urgency_score'] = case_features['urgency'].map(urgency_mapping)
        
        # Case difficulty score
        case_features['difficulty_score'] = (
            (case_features['complexity_score'] / 10 * 0.5) +
            (case_features['urgency_score'] / 10 * 0.5)
        ) * 10
        
        # Description complexity indicator
        case_features['description_complexity'] = pd.cut(
            case_features['description_length'],
            bins=[0, 100, 250, 500],
            labels=['Simple', 'Moderate', 'Complex']
        )
        
        return case_features
    
    def calculate_compatibility_score(self, officer, case):
        """Calculate compatibility between officer and case"""
        score = 0
        
        # Experience match (officer should have enough experience)
        if case['complexity_score'] <= 3 and officer['years_experience'] >= 1:
            score += 25
        elif case['complexity_score'] <= 7 and officer['years_experience'] >= 5:
            score += 25
        elif case['complexity_score'] > 7 and officer['years_experience'] >= 15:
            score += 25
        
        # Location match
        if case['location'] == officer['region']:
            score += 20
        
        # Language match
        if case['language_required'] in officer['languages']:
            score += 15
        
        # Success rate
        score += officer['success_rate'] * 20
        
        # Availability
        if officer['availability']:
            score += 10
        
        # Workload consideration
        workload_factor = max(0, 5 - (officer['cases_assigned'] / 5))
        score += min(5, workload_factor)
        
        return score
    
    def rank_officers_for_case(self, case, top_n=5):
        """Rank officers for a given case"""
        officers = self.officers_df.copy()
        
        # Calculate compatibility score for each officer
        officers['compatibility_score'] = officers.apply(
            lambda row: self.calculate_compatibility_score(row, case), axis=1
        )
        
        # Rank officers
        ranked_officers = officers.nlargest(top_n, 'compatibility_score')
        
        return ranked_officers[['officer_id', 'years_experience', 'success_rate', 
                                'region', 'availability', 'compatibility_score']]