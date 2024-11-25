import pandas as pd
from typing import List, Dict
from pathlib import Path
from app.utils.logger import logger

def create_sample_data(data_dir: Path):
    """Create sample CSV files if they don't exist"""
    
    # Sample data
    employees_data = {
        'id': range(1, 11),
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 
                     'David', 'Eve', 'Frank', 'Grace', 'Henry'],
        'last_name': ['Smith', 'Doe', 'Johnson', 'Williams', 'Brown',
                     'Davis', 'Wilson', 'Moore', 'Taylor', 'Anderson'],
        'department_id': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
        'position': ['Lead Engineer', 'Sales Director', 'Marketing Head', 'HR Director',
                    'Senior Engineer', 'Sales Manager', 'Marketing Manager', 'HR Manager',
                    'Engineer', 'Sales Representative'],
        'salary': [120000, 150000, 140000, 130000, 100000,
                  90000, 95000, 85000, 80000, 70000],
        'hire_date': ['2020-01-15', '2019-03-20', '2019-06-10', '2019-09-05',
                     '2020-04-15', '2020-07-20', '2020-10-10', '2021-01-05',
                     '2021-04-15', '2021-07-20'],
        'manager_id': [None, None, None, None, 1, 2, 3, 4, 1, 2]
    }
    
    departments_data = {
        'id': range(1, 5),
        'name': ['Engineering', 'Sales', 'Marketing', 'HR'],
        'head_id': [1, 2, 3, 4],
        'budget': [1000000, 800000, 600000, 400000],
        'location': ['Floor 3', 'Floor 2', 'Floor 2', 'Floor 1']
    }
    
    financials_data = {
        'id': range(1, 13),
        'department_id': [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4],
        'year': [2023] * 12,
        'quarter': [1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 3],
        'revenue': [300000, 350000, 400000, 450000, 200000, 250000, 100000, 120000,
                   380000, 480000, 280000, 140000],
        'expenses': [250000, 280000, 300000, 320000, 150000, 180000, 80000, 90000,
                    300000, 350000, 200000, 100000],
        'profit': [50000, 70000, 100000, 130000, 50000, 70000, 20000, 30000,
                  80000, 130000, 80000, 40000]
    }
    
    # Create DataFrames
    employees_df = pd.DataFrame(employees_data)
    departments_df = pd.DataFrame(departments_data)
    financials_df = pd.DataFrame(financials_data)
    
    # Save to CSV
    data_dir.mkdir(exist_ok=True)
    employees_df.to_csv(data_dir / 'employees.csv', index=False)
    departments_df.to_csv(data_dir / 'departments.csv', index=False)
    financials_df.to_csv(data_dir / 'financials.csv', index=False)
    
    logger.info("Sample data created successfully")

def process_data_for_rag(data_dir: Path) -> List[str]:
    """Process CSV data into documents for RAG"""
    try:
        employees = pd.read_csv(data_dir / "employees.csv")
        departments = pd.read_csv(data_dir / "departments.csv")
        financials = pd.read_csv(data_dir / "financials.csv")
        
        documents = []
        
        # Process employees with departments
        emp_dept = pd.merge(
            employees,
            departments,
            left_on='department_id',
            right_on='id',
            suffixes=('_emp', '_dept')
        )
        
        # Group by department
        for dept_name, group in emp_dept.groupby('name'):
            emp_list = []
            for _, emp in group.iterrows():
                emp_list.append(f"""
                    {emp['first_name']} {emp['last_name']}:
                    - Position: {emp['position']}
                    - Salary: ${emp['salary']:,}
                    - Hire Date: {emp['hire_date']}
                    """)
            
            doc = f"""
            Department: {dept_name}
            Location: {group.iloc[0]['location']}
            Employees:
            {''.join(emp_list)}
            """
            documents.append(doc)
        
        # Process financials by department and quarter
        fin_dept = pd.merge(
            financials,
            departments,
            left_on='department_id',
            right_on='id'
        )
        
        for dept_name, group in fin_dept.groupby('name'):
            for year in group['year'].unique():
                year_data = group[group['year'] == year]
                quarters_info = []
                for _, quarter in year_data.iterrows():
                    quarters_info.append(f"""
                    Q{quarter['quarter']}:
                    - Revenue: ${quarter['revenue']:,}
                    - Expenses: ${quarter['expenses']:,}
                    - Profit: ${quarter['profit']:,}
                    """)
                
                doc = f"""
                Department: {dept_name}
                Year: {year}
                Financial Results:
                {''.join(quarters_info)}
                """
                documents.append(doc)
        
        return documents
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise