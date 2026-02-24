import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Define data ranges
CRIME_TYPES = ['Theft', 'Robbery', 'Assault', 'Fraud', 'Drug Dealing', 'Burglary', 'Homicide', 'Cybercrime']
UNITS = ['General Patrol', 'Homicide', 'Narcotics', 'Cyber', 'Fraud Investigation', 'Gang Division']
LANGUAGES = ['English', 'Spanish', 'French', 'Mandarin', 'Arabic']
REGIONS = ['North', 'South', 'East', 'West', 'Downtown', 'Suburban']
COMPLEXITY_LEVELS = ['Low', 'Medium', 'High', 'Critical']

# Generate Officers Data
def generate_officers(n_officers=100):
    officers = []
    for i in range(n_officers):
        officer_id = f"OFF{i+1:04d}"
        years_experience = np.random.randint(1, 30)
        special_units = random.sample(UNITS, k=random.randint(1, 3))
        languages = random.sample(LANGUAGES, k=random.randint(1, 3))
        region = random.choice(REGIONS)
        
        # Success rate increases with experience
        base_success_rate = 0.5 + (years_experience / 60)
        success_rate = min(0.95, base_success_rate + np.random.normal(0, 0.1))
        success_rate = max(0.3, success_rate)
        
        solved_cases = int(years_experience * 3 * success_rate)
        availability = np.random.choice([True, False], p=[0.85, 0.15])
        
        officers.append({
            'officer_id': officer_id,
            'years_experience': years_experience,
            'special_units': '|'.join(special_units),
            'languages': '|'.join(languages),
            'region': region,
            'solved_cases': solved_cases,
            'success_rate': round(success_rate, 2),
            'availability': availability,
            'cases_assigned': np.random.randint(0, 20)
        })
    
    return pd.DataFrame(officers)

# Generate Cases Data
def generate_cases(n_cases=500):
    cases = []
    for i in range(n_cases):
        case_id = f"CASE{i+1:05d}"
        crime_type = random.choice(CRIME_TYPES)
        location = random.choice(REGIONS)
        complexity = random.choice(COMPLEXITY_LEVELS)
        language_required = random.choice(LANGUAGES)
        urgency = np.random.choice(['Low', 'Medium', 'High', 'Critical'], p=[0.3, 0.3, 0.25, 0.15])
        
        # Create date within last 2 years
        days_ago = random.randint(0, 730)
        case_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Complexity score (1-10)
        complexity_score = {'Low': np.random.randint(1, 3), 
                           'Medium': np.random.randint(4, 7), 
                           'High': np.random.randint(8, 10),
                           'Critical': 10}[complexity]
        
        cases.append({
            'case_id': case_id,
            'crime_type': crime_type,
            'location': location,
            'complexity': complexity,
            'complexity_score': complexity_score,
            'language_required': language_required,
            'urgency': urgency,
            'case_date': case_date,
            'description_length': np.random.randint(50, 500)
        })
    
    return pd.DataFrame(cases)

# Generate Historical Assignments
def generate_assignments(officers_df, cases_df, n_assignments=400):
    assignments = []
    
    for _ in range(n_assignments):
        officer = officers_df.sample(1).iloc[0]
        case = cases_df.sample(1).iloc[0]
        
        # Resolution probability based on officer success rate and case complexity
        resolution_prob = officer['success_rate'] * (1 - (case['complexity_score'] / 20))
        resolution_prob = max(0.2, min(0.95, resolution_prob))
        
        resolved = np.random.choice([True, False], p=[resolution_prob, 1 - resolution_prob])
        
        if resolved:
            resolution_days = np.random.randint(1, 120)
        else:
            resolution_days = None
        
        assignments.append({
            'assignment_id': f"ASS{len(assignments)+1:05d}",
            'officer_id': officer['officer_id'],
            'case_id': case['case_id'],
            'crime_type': case['crime_type'],
            'case_complexity': case['complexity_score'],
            'officer_experience': officer['years_experience'],
            'case_urgency': case['urgency'],
            'case_location': case['location'],
            'officer_region': officer['region'],
            'location_match': 1 if case['location'] == officer['region'] else 0,
            'language_match': 1 if case['language_required'] in officer['languages'].split('|') else 0,
            'resolved': resolved,
            'resolution_days': resolution_days,
            'outcome': 'Resolved' if resolved else 'Unresolved'
        })
    
    return pd.DataFrame(assignments)

# Generate data
print("Generating synthetic dataset...")
officers_df = generate_officers(100)
cases_df = generate_cases(500)
assignments_df = generate_assignments(officers_df, cases_df, 400)

# Save to CSV
officers_df.to_csv('data/officers.csv', index=False)
cases_df.to_csv('data/cases.csv', index=False)
assignments_df.to_csv('data/assignments_history.csv', index=False)

print(f"\n✅ Data Generated Successfully!")
print(f"\nOfficers: {len(officers_df)}")
print(f"Cases: {len(cases_df)}")
print(f"Historical Assignments: {len(assignments_df)}")
print(f"\nFiles saved to data/ directory")
print(f"\nSample Officers Data:\n{officers_df.head()}")
print(f"\nSample Cases Data:\n{cases_df.head()}")
print(f"\nSample Assignments Data:\n{assignments_df.head()}")