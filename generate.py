import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

def generate_sample_data(num_departments=5, num_employees=50):
    fake = Faker()
    
    # Generate Departments
    departments = {
        'dept_id': range(1, num_departments + 1),
        'dept_name': [
            'Engineering',
            'Marketing',
            'Human Resources',
            'Finance',
            'Operations'
        ][:num_departments],  # Handle cases where fewer departments are needed
        'location': [
            'New York',
            'San Francisco',
            'Chicago',
            'Boston',
            'Austin'
        ][:num_departments],
        'budget': [
            random.randint(500_000, 2_000_000) 
            for _ in range(num_departments)
        ],
        'dept_head': [
            fake.name() 
            for _ in range(num_departments)
        ]
    }
    
    df_departments = pd.DataFrame(departments)
    
    # Generate Employees
    employees = {
        'emp_id': range(1, num_employees + 1),
        'first_name': [fake.first_name() for _ in range(num_employees)],
        'last_name': [fake.last_name() for _ in range(num_employees)],
        'dept_id': [random.randint(1, num_departments) for _ in range(num_employees)],
        'date_hire': [
            (datetime.now() - timedelta(days=random.randint(0, 3650))).strftime('%Y-%m-%d')
            for _ in range(num_employees)
        ],
        'salary': [
            random.randint(50_000, 150_000) 
            for _ in range(num_employees)
        ],
        'performance_rating': [
            round(random.uniform(3.0, 5.0), 2) 
            for _ in range(num_employees)
        ],
        'bonus_percentage': [
            round(random.uniform(0.05, 0.15), 2) 
            for _ in range(num_employees)
        ],
        'vacation_days': [
            random.randint(10, 25) 
            for _ in range(num_employees)
        ],
        'remote_work': [
            random.choice(['Full Remote', 'Hybrid', 'Office']) 
            for _ in range(num_employees)
        ]
    }
    
    df_employees = pd.DataFrame(employees)
    
    # Positions based on departments
    positions = {
        'Engineering': ['Software Engineer', 'DevOps Engineer', 'QA Engineer', 'Tech Lead', 'System Architect'],
        'Marketing': ['Marketing Specialist', 'Content Writer', 'SEO Specialist', 'Marketing Manager', 'Brand Manager'],
        'Human Resources': ['HR Specialist', 'Recruiter', 'HR Manager', 'Training Coordinator', 'HR Assistant'],
        'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'Auditor', 'Financial Controller'],
        'Operations': ['Operations Manager', 'Project Manager', 'Business Analyst', 'Operations Coordinator', 'Process Specialist']
    }
    
    # Email, Username, and Position Assignment
    for index, row in df_employees.iterrows():
        first = row['first_name'].lower()
        last = row['last_name'].lower()
        email = f"{first[0]}{last}@company.com"
        username = f"{first[0]}{last[:7]}"
        df_employees.at[index, 'email'] = email
        df_employees.at[index, 'username'] = username
        
        dept_name = df_departments.loc[df_departments['dept_id'] == row['dept_id'], 'dept_name'].iloc[0]
        position = random.choice(positions.get(dept_name, ['General Employee']))
        df_employees.at[index, 'position'] = position
    
    # Assign Managers
    potential_managers = df_employees.sample(n=max(1, num_employees // 5))['emp_id'].tolist()
    for index, row in df_employees.iterrows():
        if row['emp_id'] not in potential_managers:
            same_dept_managers = df_employees[
                (df_employees['emp_id'].isin(potential_managers)) & 
                (df_employees['dept_id'] == row['dept_id'])
            ]['emp_id'].tolist()
            if same_dept_managers:
                df_employees.at[index, 'manager_id'] = random.choice(same_dept_managers)
            else:
                df_employees.at[index, 'manager_id'] = random.choice(potential_managers)
    
    # Save to CSV
    df_departments.to_csv('departments.csv', index=False)
    df_employees.to_csv('employees.csv', index=False)
    
    return df_departments, df_employees

if __name__ == "__main__":
    dept_df, emp_df = generate_sample_data()
    print("\nDepartments Sample:")
    print(dept_df.head())
    print("\nEmployees Sample:")
    print(emp_df.head())
