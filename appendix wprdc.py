# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:21:32 2023

@author: breeb
"""
##set up
##load packages
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency  
from scipy.stats  import spearmanr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from collections import Counter
import numpy as np
from scipy import stats

# Load the dataset
file_path = 'updated_final_merged_dataset_feb24.csv'   
data = pd.read_csv(file_path)


## Define relevant subsets
##all data
subset_all_geographies = data.copy()

##All Data, Republican and Other
subset_republican_other = data[(data['Political_Party_grouped'] == 'Republican') | (data['Political_Party_grouped'] == 'Other')]


##All Data, Republican

subset_republican = data[(data['Political_Party_grouped'] == 'Republican')]

##All Data, Other

subset_other = data[(data['Political_Party_grouped'] == 'Other')]

##All Data, Democrat


subset_democrat= data[(data['Political_Party_grouped'] == 'Democrat')]

##Allegheny, all parties


subset_allegheny = data[data['county'] == 'ALLEGHENY']

##Allegheny, Republican and Other

subset_allegheny_republican_other = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Republican') | (data['Political_Party_grouped'] == 'Other')]


##Allegheny, Republican
subset_allegheny_republican = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Republican')]

##Allegheny, Other
subset_allegheny_other = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Other')]


##Allegheny, Democrat
subset_allegheny_democrat = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Democrat')]


##Not Allegheny, all parties
##define other counties
other_counties = ['Butler', 'Beaver', 'Washington', 'Westmoreland', 'Fayette', 'Armstrong']
subset_other_counties = data[data['county'] != 'ALLEGHENY']


##Not Allegheny, Republican and Other
subset_other_counties_republican_other = data[(data['county'] != 'ALLEGHENY') & ((data['Political_Party_grouped'] == 'Republican') | (data['Political_Party_grouped'] == 'Other'))]


##Not Allegheny, Republican
subset_other_counties_republican = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Republican')]



##Not ALlegheny, Other
subset_other_counties_other = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Other')]


##Not Allegheny, Democrat
subset_other_counties_democrat = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped'] == 'Democrat')]



##Section 1: Factors correlated with confidence in the election (Chi Square Test)

# Function to run weighted chi square on subsets and make table

def generate_latex_table_from_subset(subset, relevant_columns, weight_column):
    """
    Perform weighted chi-square tests on a given subset and generate a LaTeX table of results.

    :param subset: DataFrame, the subset of data to analyze.
    :param relevant_columns: List[str], list of columns to perform the chi-square test on.
    :param weight_column: str, the name of the column containing weights.
    :return: str, LaTeX table as a string.
    """
    def perform_weighted_chi_square_test(data, variable, weight_column):
        contingency_table = data.groupby(['election_trust', variable]).apply(lambda x: np.sum(x[weight_column]))
        contingency_table = contingency_table.unstack(fill_value=0)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p, dof

    # Dropping rows with missing values in the subset
    cleaned_subset = subset[relevant_columns].dropna()
    
    relevant_columns = [
        'election_trust', 'AgeGroup', 'Race_grouped', 'Income_grouped', 
        'Education_grouped', 'Health_Group', 'Gender', 
        'Local Facebook Group', 'Number of Groups', 'combined_weight_rescaled'
    ]

    # Performing weighted chi-square test for each variable
    results = {}
    for variable in relevant_columns[1:-1]:  # Excluding 'election_trust' and the weight column
        results[variable] = perform_weighted_chi_square_test(cleaned_subset, variable, weight_column)

    # Generating the LaTeX table
    latex_table = "\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
    latex_table += "Variable & Chi-square & p-value & Degrees of Freedom \\\\\n\\hline\n"
    for variable, (chi2, p, dof) in results.items():
        latex_table += f"{variable} & {chi2:.2f} & {p:.3f} & {dof} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}"

    return latex_table



# Make tables
latex_table1 = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
print(latex_table1)
latex_table2= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table2)
latex_table3 = generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table3)
latex_table4 = generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table4)
latex_table5 = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table5)
latex_table6 = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7 = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table7)
latex_table8 = generate_latex_table_from_subset(subset_allegheny_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table8)
latex_table9 = generate_latex_table_from_subset(subset_allegheny_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table9)
latex_table10 = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table10)
latex_table11 = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12 = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table12)
latex_table13 = generate_latex_table_from_subset(subset_other_counties_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table13)
latex_table14 = generate_latex_table_from_subset(subset_other_counties_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table14)
latex_table15 = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table15)

##unweighted analysis
def generate_latex_table_from_subset_unweighted(subset, relevant_columns):
    """
    Perform unweighted chi-square tests on a given subset and generate a LaTeX table of results.

    :param subset: DataFrame, the subset of data to analyze.
    :param relevant_columns: List[str], list of columns to perform the chi-square test on.
    :return: str, LaTeX table as a string.
    """
    def perform_unweighted_chi_square_test(data, variable):
        contingency_table = data.groupby(['election_trust', variable])
        contingency_table = contingency_table.unstack(fill_value=0)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p, dof

    # Dropping rows with missing values in the subset
    cleaned_subset = subset[relevant_columns].dropna()

    # Performing weighted chi-square test for each variable
    results = {}
    for variable in relevant_columns[1:-1]:  # Excluding 'election_trust' 
        results[variable] = perform_unweighted_chi_square_test(cleaned_subset, variable)

    # Generating the LaTeX table
    latex_table = "\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
    latex_table += "Variable & Chi-square & p-value & Degrees of Freedom \\\\\n\\hline\n"
    for variable, (chi2, p, dof) in results.items():
        latex_table += f"{variable} & {chi2:.2f} & {p:.3f} & {dof} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}"

    return latex_table

relevant_columns = [
    'election_trust', 'AgeGroup', 'Race_grouped', 'Income_grouped', 
    'Education_grouped', 'Health_Group', 'Gender', 
    'Local Facebook Group', 'Number of Groups'
]

# make tables
latex_table16 = generate_latex_table_from_subset_unweighted(subset_all_geographies, relevant_columns)
latex_table17= generate_latex_table_from_subset_unweighted(subset_republican_other, relevant_columns)
latex_table18 = generate_latex_table_from_subset_unweighted(subset_democrat, relevant_columns)
latex_table19 = generate_latex_table_from_subset_unweighted(subset_allegheny, relevant_columns)
latex_table20 = generate_latex_table_from_subset_unweighted(subset_allegheny_republican_other, relevant_columns)
latex_table21 = generate_latex_table_from_subset_unweighted(subset_allegheny_democrat, relevant_columns)
latex_table22 = generate_latex_table_from_subset_unweighted(subset_other_counties, relevant_columns)
latex_table23 = generate_latex_table_from_subset_unweighted(subset_other_counties_republican_other, relevant_columns)
latex_table24 = generate_latex_table_from_subset_unweighted(subset_other_counties_democrat, relevant_columns)

##Section 2: Logit on Election Denial Results

#Data Preparation



# Step 1: Convert categorical variables to dummy variables
# List of categorical variables you want to convert (adjust this list based on your analysis needs)
categorical_vars = ['Gender', 'AgeGroup', 'Race_grouped', 'Income_grouped', 'Education_grouped', 'Health_Group', 'Political_Party_grouped', 'Local Facebook Group', 'social_media']

# Convert these variables into dummy variables
data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
# Create a dummy variable for the county being Allegheny or not
data['County_Allegheny'] = (data['county'] == 'ALLEGHENY').astype(int)

# Ensure the dependent variable 'election_trust' is numeric
# Assuming 'election_trust' needs to be converted from textual representation to binary
# Note: Adjust the lambda function based on the actual values in 'election_trust'
data['election_trust'] = data['election_trust'].apply(lambda x: 1 if x == 'High' else 0)

# Step 2: Ensure all predictors are numeric (already achieved by converting to dummies and initial transformations)

# Step 3: Check for and handle NaNs in predictors and the dependent variable
# Assuming 'combined_weight_rescaled' is your weight column and it's already numeric and without NaNs
# Drop rows with NaNs in any of the model variables
model_vars = [var for var in data.columns if var not in ['election_trust', 'combined_weight_rescaled']]  # Adjust based on actual model variables




# Define relevant subsets
subset_all_geographies = data.copy()


# All Data, Republican and Other
subset_republican_other = data[(data['Political_Party_grouped_Republican'] == 1) | (data['Political_Party_grouped_Other'] == 1)]

# All Data, Republican
subset_republican = data[data['Political_Party_grouped_Republican'] == 1]

# All Data, Other
subset_other = data[data['Political_Party_grouped_Other'] == 1]

# All Data, Democrat
# Assuming Democrat individuals are those not flagged as Republican or Other
subset_democrat = data[(data['Political_Party_grouped_Republican'] == 0) & (data['Political_Party_grouped_Other'] == 0)]

# Allegheny, all parties
subset_allegheny = data[data['county'] == 'ALLEGHENY']

# Allegheny, Republican and Other
subset_allegheny_republican_other = data[(data['county'] == 'ALLEGHENY') & ((data['Political_Party_grouped_Republican'] == 1) | (data['Political_Party_grouped_Other'] == 1))]

# Allegheny, Republican
subset_allegheny_republican = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped_Republican'] == 1)]

# Allegheny, Other
# Correction: The original condition seems to mistakenly filter for Republicans again. Adjusting it for 'Other'.
subset_allegheny_other = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped_Other'] == 1)]

# Allegheny, Democrat
# Assuming Democrat individuals are those not flagged as Republican or Other
subset_allegheny_democrat = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party_grouped_Republican'] == 0) & (data['Political_Party_grouped_Other'] == 0)]

# Not Allegheny, all parties
subset_other_counties = data[data['county'] != 'ALLEGHENY']

# Not Allegheny, Republican and Other
subset_other_counties_republican_other = data[(data['county'] != 'ALLEGHENY') & ((data['Political_Party_grouped_Republican'] == 1) | (data['Political_Party_grouped_Other'] == 1))]

# Not Allegheny, Republican
subset_other_counties_republican = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped_Republican'] == 1)]

# Not Allegheny, Other
subset_other_counties_other = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped_Other'] == 1)]

# Not Allegheny, Democrat
# Assuming Democrat individuals are those not flagged as Republican or Other
subset_other_counties_democrat = data[(data['county'] != 'ALLEGHENY') & (data['Political_Party_grouped_Republican'] == 0) & (data['Political_Party_grouped_Other'] == 0)]

print(data.columns)

# Define the function for weighted logit regression
def perform_weighted_logit_regression(subset, dependent_var, independent_vars, weight_column):
    """
    Perform a logistic regression on a given subset of data and output results as a LaTeX table.
    Note: This function setup does not directly apply weights in the regression due to method constraints.
    """
    X = subset[independent_vars]
    y = subset[dependent_var]
    X = sm.add_constant(X)  # Adds a constant term to the predictors
    logit_model = sm.Logit(y, X)
    results = logit_model.fit()  # Removed the weights parameter from the call
    latex_table = results.summary().as_latex()
    return latex_table

# Perform weighted logit regression for each subset
dependent_var = 'election_trust'
weight_column = 'combined_weight_rescaled'
independent_vars= ['Gender_Male', 'AgeGroup_Over 65', 'AgeGroup_Under 45', 'Race_grouped_White', 'Income_grouped_Less than 55,000', 'Income_grouped_More than 100,000', 'Education_grouped_Less than BA', 'Education_grouped_More than BA', 
                   'Health_Group_Not good', 'Local Facebook Group_Yes', 'social_media_6 or more hours', 'County_Allegheny']


latex_table1Z = perform_weighted_logit_regression(data, dependent_var, independent_vars, weight_column)
print(latex_table1Z)

# Add similar lines for other subsets e.g., subset_republican_other, subset_democrat, etc.

# Run function for required subsets
latex_table1A = perform_weighted_logit_regression(subset_all_geographies, dependent_var, independent_vars, weight_column)

print(latex_table1A)
latex_table2A = perform_weighted_logit_regression(subset_republican_other, dependent_var, independent_vars, weight_column)
print(latex_table2A)
latex_table3A = perform_weighted_logit_regression(subset_democrat, dependent_var, independent_vars, weight_column)
print(latex_table3A)
latex_table4A = perform_weighted_logit_regression(subset_allegheny, dependent_var, independent_vars, weight_column)
print(latex_table4A)
latex_table5A = perform_weighted_logit_regression(subset_allegheny_republican_other, dependent_var, independent_vars, weight_column)
print(latex_table5A)
latex_table6A = perform_weighted_logit_regression(subset_allegheny_democrat, dependent_var, independent_vars, weight_column)
print(latex_table6A)
latex_table7A = perform_weighted_logit_regression(subset_other_counties, dependent_var, independent_vars, weight_column)
print(latex_table7A)
latex_table8A = perform_weighted_logit_regression(subset_other_counties_republican_other, dependent_var, independent_vars, weight_column)
print(latex_table8A)
latex_table9A = perform_weighted_logit_regression(subset_other_counties_democrat, dependent_var, independent_vars, weight_column)
print(latex_table9A)

##Section  3: Spearman Rank Correlation
def weighted_spearman_corr_table(subset, variables, weight_column):
    """
    Calculate weighted Spearman rank correlations for multiple variable pairs and output as a LaTeX table.

    :param subset: DataFrame, the subset of data to analyze.
    :param variables: List[str], list of variables to calculate correlations.
    :param weight_column: str, the name of the column containing weights.
    :return: LaTeX formatted string of the Spearman correlation results.
    """
    def expand_data(data, weight_col):
        return data.loc[data.index.repeat(data[weight_col].astype(int))]

    # Expand the dataset according to weights
    expanded_data = expand_data(subset, weight_column)

    # Initialize LaTeX table
    latex_table = "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_table += "Variable Pair & Spearman Correlation & p-value \\\\\n\\hline\n"

    # Calculate Spearman rank correlations for each pair of variables
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var1 = variables[i]
            var2 = variables[j]
            corr, p_value = spearmanr(expanded_data[var1], expanded_data[var2])
            latex_table += f"{var1} & {var2} & {corr:.3f} & {p_value:.3f} \\\\\n"

    latex_table += "\\hline\n\\end{tabular}"

    return latex_table

# VARIABLES
variables = ['Q1L', 'Q2L', 'national_trust', 'local_trust', 'Education', 'Income', 'AgeGroup', 'Political_Party_grouped', 'Health', 'social_media', "local facebook group"]  
weight_column = 'combined_weight_rescaled' 

#  subsets
latex_table1B = weighted_spearman_corr_table(subset_all_geographies, variables, weight_column)
latex_table2B = weighted_spearman_corr_table(subset_allegheny, variables, weight_column)
latex_table3B = weighted_spearman_corr_table(subset_other_counties, variables, weight_column)
latex_table4B = weighted_spearman_corr_table(subset_republican_other, variables, weight_column)
latex_table5B = weighted_spearman_corr_table(subset_allegheny_republican_other, variables, weight_column)
latex_table6B = weighted_spearman_corr_table(subset_other_counties_republican_other, variables, weight_column)
latex_table7B = weighted_spearman_corr_table(subset_democrat, variables, weight_column)
latex_table8B = weighted_spearman_corr_table(subset_allegheny_democrat, variables, weight_column)
latex_table9B = weighted_spearman_corr_table(subset_other_counties_republican_other, variables, weight_column)

#reorganized the tables a bit and combined them for concision


##Section 4: Election Trust and Difference in Means Tests
##Gender


def weighted_t_test(subset, test_var, group_var, weight_var):
    """
    Perform a weighted t-test between two groups in a given subset.

    :param subset: DataFrame, the subset of data to analyze.
    :param test_var: str, the variable to test.
    :param group_var: str, the grouping variable (binary).
    :param weight_var: str, the name of the column containing weights.
    :return: t-statistic, p-value, degrees of freedom
    """
    # Split the data into two groups
    group1 = subset[subset[group_var] == subset[group_var].unique()[0]]
    group2 = subset[subset[group_var] == subset[group_var].unique()[1]]

    # Calculate weighted means
    mean1 = np.average(group1[test_var], weights=group1[weight_var])
    mean2 = np.average(group2[test_var], weights=group2[weight_var])

    # Calculate weighted variances
    var1 = np.average((group1[test_var] - mean1)**2, weights=group1[weight_var])
    var2 = np.average((group2[test_var] - mean2)**2, weights=group2[weight_var])

    # Calculate weighted sample sizes
    n1 = group1[weight_var].sum()
    n2 = group2[weight_var].sum()

    # Calculate the t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)

    # Degrees of freedom
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # Calculate the p-value
    p_value = stats.t.sf(np.abs(t_stat), df)*2  # Two-tailed test

    return t_stat, p_value, df

# plug in vars
test_var = 'election_trust'
group_var = 'Gender_Male'
weight_var = 'combined_weight_rescaled'  

# Perform the weighted t-test for each subset
t_stat_all, p_value_all, df_all = weighted_t_test(subset_all_geographies, test_var, group_var, weight_var)
print(t_stat_all, p_value_all, df_all)
t_stat_all_RO, p_value_all_RO, df_all_RO= weighted_t_test(subset_republican_other, test_var, group_var, weight_var)
print(t_stat_all_RO, p_value_all_RO, df_all_RO)
t_stat_all_D, p_value_all_D, df_all_D = weighted_t_test(subset_democrat, test_var, group_var, weight_var)
print(t_stat_all_D, p_value_all_D, df_all_D )
t_stat_A, p_value_A, df_A = weighted_t_test(subset_allegheny, test_var, group_var, weight_var)
print(t_stat_A, p_value_A, df_A)
t_stat_A_RO, p_value_A_RO, df_A_RO = weighted_t_test(subset_allegheny_republican_other, test_var, group_var, weight_var)
print(t_stat_A_RO, p_value_A_RO, df_A_RO)
t_stat_A_D, p_value_A_D, df_A_D = weighted_t_test(subset_allegheny_democrat, test_var, group_var, weight_var)
print(t_stat_A_D, p_value_A_D, df_A_D)
t_stat_6, p_value_6, df_6 = weighted_t_test(subset_other_counties, test_var, group_var, weight_var)
print(t_stat_6, p_value_6, df_6)
t_stat_6_RO, p_value_6_RO, df_6_RO = weighted_t_test(subset_other_counties_republican_other, test_var, group_var, weight_var)
print(t_stat_6_RO, p_value_6_RO, df_6_RO)
t_stat_6_D, p_value_6_D, df_6_D = weighted_t_test(subset_other_counties_democrat, test_var, group_var, weight_var)
print(t_stat_6_D, p_value_6_D, df_6_D)

##repeat for Race-- race is binarized to white/ non-white
# Assuming 'data' is your pandas DataFrame
print(subset_all_geographies.columns.tolist())

group_var= 'Race_grouped_White'



# Perform the weighted t-test for each subset
t_stat_all2, p_value_all2, df_all2 = weighted_t_test(subset_all_geographies, test_var, group_var, weight_var)
print(t_stat_all2, p_value_all2, df_all2)
t_stat_all_RO2, p_value_all_RO2, df_all_RO2= weighted_t_test(subset_republican_other, test_var, group_var, weight_var)
print(t_stat_all_RO2, p_value_all_RO2, df_all_RO2)
t_stat_all_D2, p_value_all_D2, df_all_D2 = weighted_t_test(subset_democrat, test_var, group_var, weight_var)
print(t_stat_all_D2, p_value_all_D2, df_all_D2)
t_stat_A2, p_value_A2, df_A2 = weighted_t_test(subset_allegheny, test_var, group_var, weight_var)
print(t_stat_A2, p_value_A2, df_A2)
t_stat_A_RO2, p_value_A_RO2, df_A_RO2 = weighted_t_test(subset_allegheny_republican_other, test_var, group_var, weight_var)
print(t_stat_A_RO2, p_value_A_RO2, df_A_RO2 )
t_stat_A_D2, p_value_A_D2, df_A_D2 = weighted_t_test(subset_allegheny_democrat, test_var, group_var, weight_var)
print(t_stat_A_D2, p_value_A_D2, df_A_D2)
t_stat_62, p_value_62, df_62 = weighted_t_test(subset_other_counties, test_var, group_var, weight_var)
print(t_stat_62, p_value_62, df_62)
t_stat_6_RO2, p_value_6_RO2, df_6_RO2 = weighted_t_test(subset_other_counties_republican_other, test_var, group_var, weight_var)
print(t_stat_6_RO2, p_value_6_RO2, df_6_RO2)
t_stat_6_D2, p_value_6_D2, df_6_D2 = weighted_t_test(subset_other_counties_democrat, test_var, group_var, weight_var)
print(t_stat_6_D2, p_value_6_D2, df_6_D2 )

##Section 5: Understanding Distrust
##Correlation Matrix


def weighted_correlation_matrix(data, variables, weight_column):
    """
    Calculate a weighted correlation matrix for specified variables, ensuring all data is numeric.
    
    :param data: DataFrame, the dataset containing the variables.
    :param variables: List[str], list of variables to include in the correlation matrix.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, weighted correlation matrix.
    """
    def expand_data(data, weight_col):
        return data.loc[data.index.repeat(data[weight_col].astype(int))]
    
    # Ensure all variables are numeric. Convert or drop non-numeric.
    data = data.copy()
    for var in variables + [weight_column]:
        data[var] = pd.to_numeric(data[var], errors='coerce')  # Convert to numeric, set errors to 'coerce' to handle non-convertibles
    
    # Drop rows with NaNs in any of the specified variables or weight column after conversion
    data.dropna(subset=variables + [weight_column], inplace=True)
    
    # Expand the dataset according to weights
    expanded_data = expand_data(data, weight_column)
    
    # Initialize the correlation matrix
    corr_matrix = pd.DataFrame(index=variables, columns=variables)
    
    # Calculate pairwise weighted correlations
    for var1 in variables:
        for var2 in variables:
            if var1 == var2:
                corr_matrix.loc[var1, var2] = 1.0
            else:
                corr, _ = spearmanr(expanded_data[var1], expanded_data[var2])
                corr_matrix.loc[var1, var2] = corr
    
    return corr_matrix.astype(float)  # Ensure the matrix is of float type

# Example usage remains the same


def correlation_matrix_to_latex(corr_matrix):
    """
    Convert a correlation matrix to a LaTeX formatted table.

    :param corr_matrix: DataFrame, the correlation matrix to format.
    :return: str, LaTeX formatted table.
    """
    latex_table = corr_matrix.to_latex(float_format="%.2f")
    return latex_table

# Plug in variables
variables = ['Q1L', 'Q2L', 'local_trust', 'national_trust']  
weight_column = 'combined_weight_rescaled'   # Replace with your actual weight column

# Calculate the weighted correlation matrix
corr_matrix = weighted_correlation_matrix(data, variables, weight_column)

# Convert the correlation matrix to a LaTeX table
latex_table = correlation_matrix_to_latex(corr_matrix)

# Print the LaTeX table
print(latex_table)

## Make  a  Heat Map
# Creating a mask for the upper triangle
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Setting the mask for the diagonal as well (to make diagonal white)
np.fill_diagonal(mask, True)

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, mask=mask, cmap='Blues', annot=True, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .5})

# Adding titles and labels (modify as needed)
plt.title('Weighted Correlation Matrix Heatmap')
plt.xlabel('')
plt.ylabel('')

# Show the plot
plt.show()

##Section 6-Comparing Trust in Local Government and the 2020 election (Q3L, Q1L)
##use chi square function


def generate_latex_table_from_subset(subset, relevant_columns, weight_column):
    """
    Perform weighted chi-square tests on a given subset and generate a LaTeX table of results.

    :param subset: DataFrame, the subset of data to analyze.
    :param relevant_columns: List[str], list of columns to perform the chi-square test on.
    :param weight_column: str, the name of the column containing weights.
    :return: str, LaTeX table as a string.
    """

    results = {}
    for variable in relevant_columns[1:-1]:  # Excluding 'election_trust' and the weight column
        # Creating a contingency table
        contingency_table = subset.groupby(['election_trust', variable]).apply(lambda x: np.sum(x[weight_column]))
        contingency_table = contingency_table.unstack(fill_value=0)
        
        # Performing the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        results[variable] = (chi2, p, dof)
    
    # Generating the LaTeX table
    latex_table = "\\begin{tabular}{|l|r|r|r|}\n\\hline\n"
    latex_table += "Variable & Chi-square & p-value & Degrees of Freedom \\\\\n\\hline\n"
    for variable, (chi2, p, dof) in results.items():
        latex_table += f"{variable} & {chi2:.2f} & {p:.3f} & {dof} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}"
    
    return latex_table

# Specify the relevant columns and the weight column
relevant_columns = ['election_trust', 'local_trust', 'combined_weight_rescaled']
weight_column = 'combined_weight_rescaled'




# Make tables
latex_table1C = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
print(latex_table1C)
latex_table2C= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table2C)
latex_table5C = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table5C)
latex_table6C = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
print(latex_table6C)
latex_table7C = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table7C)
latex_table10C = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table10C)
latex_table11C = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
print(latex_table11C)
latex_table12C = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table12C)
latex_table15C = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table15C)

##combined tables into 1


##Section 7: Perceptions of Misinformation by Party 
##Group Q11 for better plotting

# Define the mapping dictionary for the new groups
response_mapping = {
    'almost all true': 'More True than False',
    'more true than false': 'More True than False',
    'half and half': 'half and half',
    'almost all false': 'More False than True',
    'more false than true': 'More False than True'
}

def create_weighted_histogram(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset.
    
    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Define the order of Q11 responses
    q11_order = ['Almost all false', 'More false than true', 'About half false and half true', 'More true than false', 'Almost all true']
    
    # Calculate the weighted percentages within each political party
    party_weighted_counts = subset.groupby(['Q11', 'Political_Party_grouped']).apply(lambda x: np.sum(x[weight_column]) / np.sum(subset.loc[subset['Political_Party_grouped'] == x['Political_Party_grouped'].iloc[0], weight_column]))
    party_weighted_counts = party_weighted_counts.reset_index(name='Weighted_Percentage')
    
    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}
    
    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Q11', y='Weighted_Percentage', hue='Political_Party_grouped', data=party_weighted_counts, palette=colors, order=q11_order)
    
    # Adding labels and title
    plt.xlabel('Q11 Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)
    
    # Annotate each bar with its percentage value
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
    # Show the plot
    plt.show()

# Work with subsets
create_weighted_histogram(subset_all_geographies, 'combined_weight_rescaled', 'All Geographies')
create_weighted_histogram(subset_allegheny, 'combined_weight_rescaled', 'Allegheny County')
create_weighted_histogram(subset_other_counties, 'combined_weight_rescaled', 'Other Counties')

##friends and family, Q14

def create_weighted_histogram(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset.
    
    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Define the order of Q14 responses
    q14_order = ['Almost all false', 'More false than true', 'About half false half true', 'More true than false', 'Almost all true']
    
    # Calculate the weighted percentages within each political party for each response in Q11
    party_weighted_counts = subset.groupby(['Q14', 'Political_Party_grouped']).apply(lambda x: np.sum(x[weight_column]) / np.sum(subset.loc[subset['Political_Party_grouped'] == x['Political_Party_grouped'].iloc[0], weight_column]))
    party_weighted_counts = party_weighted_counts.reset_index(name='Weighted_Percentage')
    
    # Reorder the Q14 responses
    party_weighted_counts['Q14'] = pd.Categorical(party_weighted_counts['Q14'], categories=q14_order, ordered=True)
    
    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}
    
    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Q14', y='Weighted_Percentage', hue='Political_Party_grouped', data=party_weighted_counts, palette=colors)
    
    # Adding labels and title
    plt.xlabel('Q14 Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)
    
    # Annotate each bar with its percentage value
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
    # Show the plot
    plt.show()

# Work with subsets
create_weighted_histogram(subset_all_geographies, 'combined_weight_rescaled', 'All Geographies')
create_weighted_histogram(subset_allegheny, 'combined_weight_rescaled', 'Allegheny County')
create_weighted_histogram(subset_other_counties, 'combined_weight_rescaled', 'Other Counties')

###Section 9: Perceptions of Misinformation Consumption by people respondent disagrees with

def create_weighted_histogram_with_percentages(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset with percentages displayed on top of each bar.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Calculate the weighted percentages
    weighted_counts = subset.groupby(['Q17', 'Political_Party_grouped']).apply(lambda x: np.sum(x[weight_column]) / np.sum(subset.loc[subset['Political_Party_grouped'] == x['Political_Party_grouped'].iloc[0], weight_column]))
    weighted_counts = weighted_counts.reset_index(name='Weighted_Percentage')

    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Q17', y='Weighted_Percentage', hue='Political_Party_grouped', data=weighted_counts, palette=colors)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Q17 Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Subsets
create_weighted_histogram_with_percentages(subset_all_geographies, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')
create_weighted_histogram_with_percentages(subset_allegheny, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')
create_weighted_histogram_with_percentages(subset_other_counties, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')

##Section 10: Comparing Perceptions of Misinformation

def create_comparison_histogram(subset, weight_column, title):
    """
    Create a histogram comparing Q14 and Q17 with weighted percentages calculated within each party.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data
    subset_q14 = subset[['Q14', 'Political_Party_grouped', weight_column]].copy()
    subset_q14['Question'] = 'Friends'
    subset_q14.rename(columns={'Q14': 'Response'}, inplace=True)

    subset_q17 = subset[['Q17', 'Political_Party_grouped', weight_column]].copy()
    subset_q17['Question'] = 'Opponents'
    subset_q17.rename(columns={'Q17': 'Response'}, inplace=True)

    combined_data = pd.concat([subset_q14, subset_q17])

    # Calculate weighted percentages
    combined_data['Weighted_Count'] = combined_data.groupby(['Question', 'Response', 'Political_Party_grouped'])['combined_weight_rescaled'].transform('sum')
    combined_data['Total_Weight'] = combined_data.groupby(['Question', 'Political_Party_grouped'])['combined_weight_rescaled'].transform('sum')
    combined_data['Weighted_Percentage'] = combined_data['Weighted_Count'] / combined_data['Total_Weight']

    # Drop duplicates
    combined_data.drop_duplicates(subset=['Question', 'Response', 'Political_Party_grouped'], inplace=True)

    # Colors for each question
    colors = {'Friends': 'purple', 'Opponents': 'green'}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Response', y='Weighted_Percentage', hue='Question', data=combined_data, palette=colors)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Subsets
create_comparison_histogram(subset_all_geographies, 'combined_weight_rescaled', 'Distribution Comparision of Q14 and Q17 (all data)')
create_comparison_histogram(subset_allegheny, 'combined_weight_rescaled', 'Distribution Comparision of Q14 and Q17 (Allegheny)')
create_comparison_histogram(subset_other_counties, 'combined_weight_rescaled', 'Distribution Comparision of Q14 and Q17 (other counties)')


##Section 11: Statistics on conservative-leaning TV news consumption
##Fox News, OANN and Newsmax consumption by party

def create_media_histogram(subset, weight_column, title):
    """
    Create a histogram with weighted percentages for Fox_viewers and OANN/Newsmax, 
    displayed separately but next to each other, and colored by political party.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data for Fox_viewers and OANN/Newsmax
    subset['Media_Outlet'] = np.nan
    subset.loc[subset['Fox_viewers'].notna(), 'Media_Outlet'] = 'Fox Viewers'
    subset.loc[subset['OANN/Newsmax'].notna(), 'Media_Outlet'] = 'OANN/Newsmax'
    subset['Response'] = subset['Fox_viewers'].fillna(subset['OANN/Newsmax'])

    # Filter out rows where both media outlet responses are NaN
    combined_data = subset.dropna(subset=['Media_Outlet', 'Response'])

    # Calculate weighted counts
    combined_data['Weighted_Count'] = combined_data['combined_weight_rescaled']
    combined_grouped = combined_data.groupby(['Media_Outlet', 'Response', 'Political_Party_grouped'])['Weighted_Count'].sum().reset_index()

    # Calculate total weight for each group
    total_weight = combined_grouped.groupby('Media_Outlet')['Weighted_Count'].transform('sum')
    combined_grouped['Weighted_Percentage'] = combined_grouped['Weighted_Count'] / total_weight

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Media_Outlet', y='Weighted_Percentage', hue='Political_Party_grouped', data=combined_grouped)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Media Outlets')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Subsets
create_media_histogram(subset_all_geographies, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')
create_media_histogram(subset_allegheny, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')
create_media_histogram(subset_other_counties, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')


## Fox News and election trust by party
relevant_columns = ['election_trust', 'Fox_viewers', 'combined_weight_rescaled']

# Make tables
latex_table1D = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
print(latex_table1D)
latex_table2D= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table5D = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table5D)
latex_table6D = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7D = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table10D = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table11D = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12D = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table15D = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table16D = generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table16D)
latex_table17D= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table17D)
##Section 12: Conservative-leaning radio consumption
##Conservative-leaning radio consumption by party


def create_radio_histogram(subset, weight_column, title):
    """
    Create a histogram with weighted percentages for KDKA AM, Word 101.5 FM, and Fox News /WJAS.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data for each radio station
    subset_kdka = subset[['KDKA AM', 'combined_weight_rescaled']].copy()
    subset_kdka['Media_Outlet'] = 'KDKA AM'

    subset_word = subset[['Word 101.5 FM', 'combined_weight_rescaled']].copy()
    subset_word['Media_Outlet'] = 'Word 101.5 FM'

    subset_fox_wjas = subset[['WJAS/Fox', 'combined_weight_rescaled']].copy()
    subset_fox_wjas['Media_Outlet'] = 'WJAS/Fox'

    combined_data = pd.concat([subset_kdka, subset_word, subset_fox_wjas])
    combined_data.rename(columns={'KDKA AM': 'Response', 'Word 101.5 FM': 'Response', 'WJAS/Fox': 'Response'}, inplace=True)

    # Filter out rows where both media outlet responses are NaN
    combined_data = combined_data.dropna(subset=['Media_Outlet', 'Response'])

    # Calculate weighted counts
    combined_data['Weighted_Count'] = combined_data['combined_weight_rescaled']
    combined_grouped = combined_data.groupby(['Media_Outlet', 'Response'])['Weighted_Count'].sum().reset_index()

    # Calculate total weight for each group
    total_weight = combined_grouped.groupby('Media_Outlet')['Weighted_Count'].transform('sum')
    combined_grouped['Weighted_Percentage'] = combined_grouped['Weighted_Count'] / total_weight

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Media_Outlet', y='Weighted_Percentage', data=combined_grouped)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Media Outlets')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Subsets
create_radio_histogram(subset_all_geographies, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')
create_radio_histogram(subset_allegheny, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')
create_radio_histogram(subset_other_counties, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')

##Conservative Radio and Election Trust Among Republicans
relevant_columns = ['election_trust', 'WJAS/Fox', 'KDKA AM', 'Word 101.5 FM', 'combined_weight_rescaled']

# Make table
latex_table1E = generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table1E)

##Conservative Radio and Election Trust Among Other
relevant_columns = ['election_trust', 'Fox News/WJAS', 'KDKA AM', 'Word 101.5 FM' 'combined_weight_rescaled']

# Make table
latex_table1F = generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
print(_table1F)

##Conservative Radio and Election Trust Among Democrats
relevant_columns = ['election_trust', 'Fox News/WJAS', 'KDKA AM', 'Word 101.5 FM' 'combined_weight_rescaled']

# Make table
latex_table1G = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')

##Section 13: Statistics on Fox News Consumption


def calculate_media_categories(subset, weight_column):
    """
    Calculate categories based on media consumption.
    
    :param subset: DataFrame, the subset of data to analyze.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, categorized data with weighted percentages.
    """
    # Adjusting conditions to check for viewership based on numeric values
    conditions = [
        (subset['Fox_viewers'] > 0) & (subset['WJAS/Fox'] <= 0),  # Fox_viewers Only
        (subset['Fox_viewers'] <= 0) & (subset['WJAS/Fox'] > 0),  # WJAS/Fox Only
        (subset['Fox_viewers'] > 0) & (subset['WJAS/Fox'] > 0),   # Both
        (subset['Fox_viewers'] <= 0) & (subset['WJAS/Fox'] <= 0)  # Neither
    ]
    choices = ['Fox_viewers Only', 'WJAS/Fox Only', 'Both', 'Neither']
    subset['Category'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate weighted counts
    category_weights = subset.groupby('Category')[weight_column].sum()
    total_weight = category_weights.sum()
    category_percentages = (category_weights / total_weight).reset_index(name='Weighted_Percentage')
    
    return category_percentages

def category_table_to_latex(df):
    """
    Convert a DataFrame to a LaTeX formatted table.
    
    :param df: DataFrame, the DataFrame to format.
    :return: str, LaTeX formatted table.
    """
    return df.to_latex(index=False, float_format="%.2f")

# Example usage
# Replace 'data' with your actual DataFrame and ensure 'combined_weight_rescaled' is your actual weight column name
subset = data  # This should be your DataFrame
weight_column = 'combined_weight_rescaled'

# Calculate the weighted categories and generate LaTeX table
category_percentages = calculate_media_categories(subset, weight_column)
latex_table = category_table_to_latex(category_percentages)

# Printing the LaTeX table
print(latex_table)

##Section 14: Statistics on liberal-leaning TV news consumption
##CNN and MSNBC consumption by party


def create_media_histogram(subset, weight_column, title):
    """
    Create a histogram with weighted percentages for CNN_viewers and MSNBC_viewers, 
    displayed separately but next to each other, and colored by political party.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data for Fox_viewers and OANN/Newsmax
    subset['Media_Outlet'] = np.nan
    subset.loc[subset['CNN_viewers'].notna(), 'Media_Outlet'] = 'CNN_viewers'
    subset.loc[subset['MSNBC_viewers'].notna(), 'Media_Outlet'] = 'MSNBC_viewers'
    subset['Response'] = subset['CNN_viewers'].fillna(subset['MSNBC_viewers'])

    # Filter out rows where both media outlet responses are NaN
    combined_data = subset.dropna(subset=['Media_Outlet', 'Response'])

    # Calculate weighted counts
    combined_data['Weighted_Count'] = combined_data['combined_weight_rescaled']
    combined_grouped = combined_data.groupby(['Media_Outlet', 'Response', 'Political_Party_grouped'])['Weighted_Count'].sum().reset_index()

    # Calculate total weight for each group
    total_weight = combined_grouped.groupby('Media_Outlet')['Weighted_Count'].transform('sum')
    combined_grouped['Weighted_Percentage'] = combined_grouped['Weighted_Count'] / total_weight

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Media_Outlet', y='Weighted_Percentage', hue='Political_Party_grouped', data=combined_grouped)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Media Outlets')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Make subsets
create_media_histogram(subset_all_geographies, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')
create_media_histogram(subset_allegheny, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')
create_media_histogram(subset_other_counties, 'combined_weight_rescaled', 'Percentage of Viewers by Media Outlet and Political Party Group')

##MSNBC viewership and election trust
relevant_columns = ['election_trust', 'MSNBC_viewers', 'combined_weight_rescaled']

# Make tables
latex_table1I = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table3I= generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
latex_table4I= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
latex_table5I = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table1I)
print(latex_table3I)

##CNN viewership and election trust
relevant_columns = ['election_trust', 'CNN_viewers', 'combined_weight_rescaled']

# Make tables
latex_table1J = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table3J= generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
latex_table4J= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
latex_table5J = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')


##Liberal leaning radio consumption
##Liberal-leaning radio consumption by party


def create_radio_histogram(subset, weight_column, title):
    """
    Create a histogram with weighted percentages for MSNBC_radio and NPR.
    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data for each radio station
    subset_kdka = subset[['MSNBC_radio', 'combined_weight_rescaled']].copy()
    subset_kdka['Media_Outlet'] = 'MSNBC_radio'

    subset_word = subset[['NPR', 'combined_weight_rescaled']].copy()
    subset_word['Media_Outlet'] = 'NPR'

    combined_data = pd.concat([subset_kdka, subset_word, subset_fox_wjas])
    combined_data.rename(columns={'MSNBC_radio': 'Response', 'NPR': 'Response'}, inplace=True)

    # Filter out rows where both media outlet responses are NaN
    combined_data = combined_data.dropna(subset=['Media_Outlet', 'Response'])

    # Calculate weighted counts
    combined_data['Weighted_Count'] = combined_data['combined_weight_rescaled']
    combined_grouped = combined_data.groupby(['Media_Outlet', 'Response'])['Weighted_Count'].sum().reset_index()

    # Calculate total weight for each group
    total_weight = combined_grouped.groupby('Media_Outlet')['Weighted_Count'].transform('sum')
    combined_grouped['Weighted_Percentage'] = combined_grouped['Weighted_Count'] / total_weight

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Media_Outlet', y='Weighted_Percentage', data=combined_grouped)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Media Outlets')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Subsets
create_radio_histogram(subset_all_geographies, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')
create_radio_histogram(subset_allegheny, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')
create_radio_histogram(subset_other_counties, 'combined_weight_rescaled', 'Percentages of Listeners by Political Party')


##Liberal leaning radio and election trust among Republicans
##Liberal leaning radio and election trust among Other
##Liberal leaning radio and election trust among Democrats

##MSNBC viewership and election trust
relevant_columns = ['election_trust', 'NPR', 'MSNBC_radio', 'combined_weight_rescaled']

# Make tables
latex_table1K = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table3K= generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
latex_table4K= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
latex_table5K = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')

print(latex_table1K)
print(latex_table3K)
print(latex_table4K)
print(latex_table5K)
##Concern about misinformation and trust in the 2020 election
## Concern about misinformation by party and location

def calculate_weighted_percentages(subset, question, weight_column):
    """
    Calculate weighted percentages for responses to a specific question.

    :param subset: DataFrame, the subset of data to analyze.
    :param question: str, the question column to analyze.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, weighted percentages for each response.
    """
    # Group by the question and calculate weighted counts
    weighted_counts = subset.groupby(question)[weight_column].sum()
    total_weight = weighted_counts.sum()
    weighted_percentages = (weighted_counts / total_weight).reset_index(name='Weighted_Percentage')

    return weighted_percentages

def weighted_percentages_to_latex(df, question):
    """
    Convert weighted percentages DataFrame to a LaTeX formatted table.

    :param df: DataFrame, the DataFrame to format.
    :param question: str, the question column to display in the table.
    :return: str, LaTeX formatted table.
    """
    df.columns = [question, 'Weighted Percentage']
    return df.to_latex(index=False, float_format="%.2f")

#  subsets
subsets = [subset_all_geographies, subset_allegheny, subset_other_counties, subset_republican, subset_other, subset_democrat]  # Replace with your actual subsets
subset_names = ['All Geographies', 'Allegheny', 'Other Counties', 'Republicans', 'Other', 'Democrat']  # Names for each subset

# Iterate through subsets and create LaTeX tables
for subset, name in zip(subsets, subset_names):
    weighted_percentages = calculate_weighted_percentages(subset, 'Q17', 'combined_weight_rescaled')  
    latex_table = weighted_percentages_to_latex(weighted_percentages, 'Q17 Responses - ')
    print(latex_table)


## How are high, medium, and low levels of concern about misinformation associated with distrust in the 2020 election

relevant_columns = ['election_trust', 'Q17', 'combined_weight_rescaled']

# Make tables
latex_table1L = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table3L= generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
latex_table4L= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
latex_table5L = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table1L)
print(latex_table3L)
print(latex_table4L)
print(latex_table5L)

##Combined into one table manually for better formatting

##Different age grouping for trust in elections by party
##Republicans
##Other
##Democrats

def calculate_weighted_percentages_by(subset, question, by_question, weight_column):
    """
    Calculate weighted percentages for responses to a question, grouped by another question.

    :param subset: DataFrame, the subset of data to analyze.
    :param question: str, the main question column to analyze.
    :param by_question: str, the column by which to group the data.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, weighted percentages for each response grouped by the by_question.
    """
    # Group by the questions and calculate weighted counts
    group = subset.groupby([by_question, question])[weight_column].sum().reset_index()
    total_weight = subset.groupby(by_question)[weight_column].sum().reset_index()
    weighted_data = pd.merge(group, total_weight, on=by_question, how='left')
    weighted_data['Weighted_Percentage'] = weighted_data[weight_column + '_x'] / weighted_data[weight_column + '_y']

    # Pivot table to format for Q1L responses by Q25
    pivot_table = weighted_data.pivot_table(index=by_question, columns=question, values='Weighted_Percentage').fillna(0)

    return pivot_table

def pivot_table_to_latex(pivot_table, title):
    """
    Convert a pivot table DataFrame to a LaTeX formatted table.

    :param pivot_table: DataFrame, the pivot table to format.
    :param title: str, the title to include in the table.
    :return: str, LaTeX formatted table.
    """
    latex_table = pivot_table.to_latex(float_format="%.2f")
    return f"{title}\n\n{latex_table}"

#  subsets
subsets = [subset_all_geographies, subset_republican, subset_other, subset_democrat]  
subset_names = ['All Geographies', 'Republicans', 'Other', 'Democrat']  # Names for each subset

# Iterate through subsets and create LaTeX tables
for subset, name in zip(subsets, subset_names):
    pivot_table = calculate_weighted_percentages_by(subset, 'Q1L', 'Q25', 'combined_weight_rescaled') 
    latex_table = pivot_table_to_latex(pivot_table, 'Q1L Responses by Q25 - ')
    print(latex_table)


##Section 19: Trust in Election Results


def calculate_weighted_percentages_for_question(subset, question, weight_column):
    """
    Calculate weighted percentages for responses to a specific question.

    :param subset: DataFrame, the subset of data to analyze.
    :param question: str, the question column to analyze.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, weighted percentages for each response.
    """
    # Group by the question and calculate weighted counts
    weighted_counts = subset.groupby(question)[weight_column].sum()
    total_weight = weighted_counts.sum()
    weighted_percentages = (weighted_counts / total_weight).reset_index(name='Weighted_Percentage')

    return weighted_percentages

def weighted_percentages_to_latex(df, title):
    """
    Convert weighted percentages DataFrame to a LaTeX formatted table.

    :param df: DataFrame, the DataFrame to format.
    :param title: str, the title of the table.
    :return: str, LaTeX formatted table.
    """
    latex_table = df.to_latex(index=False, float_format="%.2f")
    return f"{title}\n\n{latex_table}"

#  subsets
subsets = [subset_republican, subset_allegheny_republican, subset_other_counties_republican, subset_other, subset_allegheny_other, subset_other_counties_other, subset_democrat, subset_allegheny_democrat, subset_other_counties_democrat]  # Replace with your actual subsets
subset_names = ['Republicans in whole MSA', 'Republicans in Allegheny', 'Republicans outside of Allegheny', 'Other in whole MSA', 'Other in Allegheny', 'Other outside of Allegheny', 'Democrats in whole MSA', 'Democrats in Allegheny', 'Democrats outside of Allegheny']  # Names for each subset

# Iterate through subsets and create LaTeX tables
for subset, name in zip(subsets, subset_names):
    weighted_percentages = calculate_weighted_percentages_for_question(subset, 'Q1L', 'combined_weight_rescaled')  
    latex_table = weighted_percentages_to_latex(weighted_percentages, 'Q1L Responses - ')
    print(latex_table)



##related visuals

def create_subset_histogram(subset, question, weight_column, title):
    """
    Create a histogram with weighted percentages for a specific question.

    :param subset: DataFrame, the subset of data to plot.
    :param question: str, the question column to analyze.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Calculate weighted counts
    subset['Weighted'] = subset[weight_column]
    subset_grouped = subset.groupby(['county', question])['Weighted'].sum().reset_index()

    # Calculate total weight for each county category
    total_weight = subset_grouped.groupby('county')['Weighted'].transform('sum')
    subset_grouped['Weighted_Percentage'] = subset_grouped['Weighted'] / total_weight

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=question, y='Weighted_Percentage', hue='county', data=subset_grouped, palette=['blue', 'yellow'])

    # Adding labels and title
    plt.xlabel(question)
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# subsets
create_subset_histogram(subset_democrat, 'Q1L', 'combined_weight_rescaled', 'Democrat Responses to Q1L')
create_subset_histogram(subset_other, 'Q1L', 'combined_weight_rescaled', 'Other Responses to Q1L')
create_subset_histogram(subset_republican, 'Q1L', 'combined_weight_rescaled', 'Republican Responses to Q1L')


##Section 20: Predictors of Election Trust
print(data.columns)

##process for categorical variables
# Example of handling categorical variables (One-Hot Encoding)
if 'county' in independent_vars:
    subset = pd.get_dummies(data, columns=['county'], drop_first=True)
if 'AgeGroup' in independent_vars:
    subset = pd.get_dummies(data, columns=['AgeGroup'], drop_first=True)
if 'Political_Party_grouped' in independent_vars:
    subset = pd.get_dummies(data, columns=['Political_Party_grouped'], drop_first=True)

# Ensure the dependent variable is binary and numeric
data[dependent_var] = data[dependent_var].apply(lambda x: 1 if x == 'Yes' else 0)


# Ensure all independent variables are numeric
for var in independent_vars:
    data[var] = pd.to_numeric(data[var], errors='coerce')

# Drop any rows with NaN values that may disrupt the regression analysis
#data.dropna(data= independent_vars + [dependent_var, weight_column], inplace=True)

##run logit
def perform_weighted_logit_regression(data, dependent_var, independent_vars, weight_column):
    """
    Perform a weighted logit regression on a given subset of data and output results as a LaTeX table.

    :param subset: DataFrame, the subset of data to analyze.
    :param dependent_var: str, the name of the dependent variable.
    :param independent_vars: List[str], list of independent variables.
    :param weight_column: str, the name of the column containing weights.
    :return: LaTeX formatted string of the logit regression results.
    """
    # Selecting the dependent and independent variables
    X = data[independent_vars]
    y = data[dependent_var]
    weights = data[weight_column]

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Performing the weighted logit regression
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(weights=weights)

    # Converting the summary to a LaTeX table
    latex_table = results.summary().as_latex()

    return latex_table

# inputs
dependent_var = 'election_trust'  
independent_vars = ['Fox_viewers', 'county', 'AgeGroup', 'Political_Party_grouped', 'interpersonal_trust']  # Replace with your actual independent variables
weight_column = 'combined_weight_rescaled'  

# Run function for required subsets
latex_table1M = perform_weighted_logit_regression(data, dependent_var, independent_vars, weight_column)


##Section 21: Facebook Group Behavior by Political Party
##Local Facebook Group participation by political party

def calculate_weighted_percentages_by_group(subset, group_col, target_col, weight_col):
    """
    Calculate weighted percentages for target_col by group_col.

    :param subset: DataFrame, the subset of data to analyze.
    :param group_col: str, the column to group by.
    :param target_col: str, the target column to analyze.
    :param weight_col: str, the name of the column containing weights.
    :return: DataFrame, weighted percentages for each combination of group_col and target_col.
    """
    # Group by the specified columns and calculate weighted counts
    group_data = subset.groupby([group_col, target_col])[weight_col].sum().reset_index()
    total_weight = group_data.groupby(group_col)[weight_col].sum().reset_index(name='Total_Weight')
    weighted_data = pd.merge(group_data, total_weight, on=group_col)
    weighted_data['Weighted_Percentage'] = weighted_data[weight_col] / weighted_data['Total_Weight']

    # Pivot table for better formatting
    pivot_table = weighted_data.pivot(index=group_col, columns=target_col, values='Weighted_Percentage').fillna(0)

    return pivot_table

def pivot_table_to_latex(pivot_table, title):
    """
    Convert a pivot table DataFrame to a LaTeX formatted table.

    :param pivot_table: DataFrame, the pivot table to format.
    :param title: str, the title of the table.
    :return: str, LaTeX formatted table.
    """
    return f"{title}\n\n{pivot_table.to_latex(float_format='%.2f')}"

# plug in vars
subset = data  
weighted_pivot = calculate_weighted_percentages_by_group(subset, 'Political_Party_grouped', 'Local Facebook Group', 'combined_weight_rescaled')
latex_table = pivot_table_to_latex(weighted_pivot, 'Distribution of In Local Facebook Group by Political Party')

# Print the LaTeX table
print(latex_table)

###Number of Facebook Groups by Party
def calculate_weighted_percentages_by_group(subset, group_col, target_col, weight_col):
    """
    Calculate weighted percentages for target_col by group_col.

    :param subset: DataFrame, the subset of data to analyze.
    :param group_col: str, the column to group by.
    :param target_col: str, the target column to analyze.
    :param weight_col: str, the name of the column containing weights.
    :return: DataFrame, weighted percentages for each combination of group_col and target_col.
    """
    # Group by the specified columns and calculate weighted counts
    group_data = subset.groupby([group_col, target_col])[weight_col].sum().reset_index()
    total_weight = group_data.groupby(group_col)[weight_col].sum().reset_index(name='Total_Weight')
    weighted_data = pd.merge(group_data, total_weight, on=group_col)
    weighted_data['Weighted_Percentage'] = weighted_data[weight_col] / weighted_data['Total_Weight']

    # Pivot table for better formatting
    pivot_table = weighted_data.pivot(index=group_col, columns=target_col, values='Weighted_Percentage').fillna(0)

    return pivot_table

def pivot_table_to_latex(pivot_table, title):
    """
    Convert a pivot table DataFrame to a LaTeX formatted table.

    :param pivot_table: DataFrame, the pivot table to format.
    :param title: str, the title of the table.
    :return: str, LaTeX formatted table.
    """
    return f"{title}\n\n{pivot_table.to_latex(float_format='%.2f')}"

# Example usage
subset = data  # Replace with your actual DataFrame
weighted_pivot = calculate_weighted_percentages_by_group(subset, 'Political_Party_grouped', 'Number of Groups', 'combined_weight_rescaled')
latex_table = pivot_table_to_latex(weighted_pivot, 'Number of Facebook Groups by Political Party')

# Print the LaTeX table
print(latex_table)

## Participation in Local Facebook Groups and Trust in the 2020 Presidential Election
def generate_latex_table_from_subset(subset, relevant_columns, weight_column):
    """
    Perform weighted chi-square tests on a given subset and generate a LaTeX table of results.

    :param subset: DataFrame, the subset of data to analyze.
    :param relevant_columns: List[str], list of columns to perform the chi-square test on.
    :param weight_column: str, the name of the column containing weights.
    :return: str, LaTeX table as a string.
    """
    def perform_weighted_chi_square_test(data, variable, weight_column):
        contingency_table = data.groupby(['election_trust', variable]).apply(lambda x: np.sum(x[weight_column]))
        contingency_table = contingency_table.unstack(fill_value=0)
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p, dof

    # Dropping rows with missing values in the subset
    cleaned_subset = subset[relevant_columns].dropna()
    
 
    relevant_columns = ['election_trust',  'Local Facebook Group', 'combined_weight_rescaled']
    # Performing weighted chi-square test for each variable
    results = {}
    for variable in relevant_columns[1:-1]:  # Excluding 'election_trust' and the weight column
        results[variable] = perform_weighted_chi_square_test(cleaned_subset, variable, weight_column)

    # Generating the LaTeX table
    latex_table = "\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
    latex_table += "Variable & Chi-square & p-value & Degrees of Freedom \\\\\n\\hline\n"
    for variable, (chi2, p, dof) in results.items():
        latex_table += f"{variable} & {chi2:.2f} & {p:.3f} & {dof} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}"

    return latex_table



# Make tables
latex_table1N = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
print(latex_table1N)
latex_table2N= generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table2N)
latex_table3N= generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table3N)
latex_table5N = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table5N)
latex_table6N = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7N = generate_latex_table_from_subset(subset_allegheny_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table7N)
latex_table8N = generate_latex_table_from_subset(subset_allegheny_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table8N)
latex_table10N = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table10N)
latex_table11N = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12N = generate_latex_table_from_subset(subset_other_counties_republican, relevant_columns, 'combined_weight_rescaled')
print(latex_table12N)
latex_table14N = generate_latex_table_from_subset(subset_other_counties_other, relevant_columns, 'combined_weight_rescaled')
print(latex_table14N)
latex_table15N = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')
print(latex_table15N)

##Section 22: News sources
##cleaning so a comma can serve as a our split (ie removal of commas which prevent that)
def replace_comma_in_phrase(s):
    specific_phrase_with_comma = "A news app or news website, like the Pittsburgh Post-Gazette"
    specific_phrase_without_comma = specific_phrase_with_comma.replace(",", "")
    return s.replace(specific_phrase_with_comma, specific_phrase_without_comma)

# Apply the function to Q9 and Q10 columns
data['Q9_cleaned'] = data['Q9'].astype(str).apply(replace_comma_in_phrase)
\
# Verify the replacement by displaying a sample of the cleaned responses
data[['Q9_cleaned']].head()

def replace_comma_in_phrase2(s):
    specific_phrase_with_comma = "A news app or news website, like the New York Times"
    specific_phrase_without_comma = specific_phrase_with_comma.replace(",", "")
    return s.replace(specific_phrase_with_comma, specific_phrase_without_comma)

# Apply the function to Q9 and Q10 columns
data['Q10_cleaned'] = data['Q10'].astype(str).apply(replace_comma_in_phrase2)

# Verify the replacement by displaying a sample of the cleaned responses
data[['Q10_cleaned']].head()

## Function to split the responses into lists and count the frequencies of each item
def count_rankings(column):
    # Split the responses into lists
    split_responses = column.str.split(',')
    # Flatten the list of lists and count occurrences
    flat_list = [item.strip() for sublist in split_responses for item in sublist if item]
    return Counter(flat_list)

# Count the frequencies for Q9 and Q10 cleaned responses
q9_rankings = count_rankings(data['Q9_cleaned'])
q10_rankings = count_rankings(data['Q10_cleaned'])

# Convert the Counter objects to DataFrames for easier analysis and display
q9_rankings_df = pd.DataFrame(q9_rankings.items(), columns=['News Source', 'Frequency']).sort_values(by='Frequency', ascending=False)
q10_rankings_df = pd.DataFrame(q10_rankings.items(), columns=['News Source', 'Frequency']).sort_values(by='Frequency', ascending=False)

q9_rankings_df.head(), q10_rankings_df.head()

##calculate weighted top news sources for all data
def calculate_weighted_percentages(column, weights):
    # Initialize a dictionary to hold cumulative weights for each news source
    cumulative_weights = {}
    
    # Iterate over each response and its corresponding weight
    for response, weight in zip(column, weights):
        # Check if the response is not NaN
        if pd.notna(response):
            # Split the response into individual news sources
            news_sources = response.split(',')
            # Calculate the weight contribution for each news source
            weight_per_source = weight / len(news_sources)
            for source in news_sources:
                source = source.strip()  # Remove any leading/trailing whitespace
                # Update the cumulative weight for this news source
                if source in cumulative_weights:
                    cumulative_weights[source] += weight_per_source
                else:
                    cumulative_weights[source] = weight_per_source
    
    # Calculate the total weight for normalization
    total_weight = sum(cumulative_weights.values())
    
    # Convert cumulative weights to percentages
    percentages = {source: (weight / total_weight * 100) for source, weight in cumulative_weights.items()}
    
    # Sort the percentages dictionary by value in descending order and convert to a DataFrame
    percentages_sorted = sorted(percentages.items(), key=lambda item: item[1], reverse=True)
    percentages_df = pd.DataFrame(percentages_sorted, columns=['News Source', 'Percentage'])
    
    return percentages_df

# Calculate weighted percentages for Q9 and Q10
q9_weighted_percentages_simple = calculate_weighted_percentages(data['Q9_cleaned'], data['combined_weight_rescaled_new'])
q10_weighted_percentages_simple = calculate_weighted_percentages(data['Q10_cleaned'], data['combined_weight_rescaled_new'])

q9_weighted_percentages_simple.head(), q10_weighted_percentages_simple.head()

##iterate through subsets
# Filter the dataset for responses from 'county' equals to "ALLEGHENY"
alleg_county_data_corrected = data[data['county'] == 'ALLEGHENY']

# Recalculate weighted percentages for Q9 and Q10 within the corrected subset
q9_alleg_percentages_corrected = calculate_weighted_percentages(alleg_county_data_corrected['Q9_cleaned'], alleg_county_data_corrected['combined_weight_rescaled_new'])
q10_alleg_percentages_corrected = calculate_weighted_percentages(alleg_county_data_corrected['Q10_cleaned'], alleg_county_data_corrected['combined_weight_rescaled_new'])

q9_alleg_percentages_corrected.head(), q10_alleg_percentages_corrected.head()

##non-allegheny
# Filter the dataset to exclude responses from 'county' equals to "ALLEGHENY"
non_alleg_county_data = data[data['county'] != 'ALLEGHENY']

# Recalculate weighted percentages for Q9 and Q10 within this new subset
q9_non_alleg_percentages = calculate_weighted_percentages(non_alleg_county_data['Q9_cleaned'], non_alleg_county_data['combined_weight_rescaled_new'])
q10_non_alleg_percentages = calculate_weighted_percentages(non_alleg_county_data['Q10_cleaned'], non_alleg_county_data['combined_weight_rescaled_new'])

q9_non_alleg_percentages.head(), q10_non_alleg_percentages.head()

##segment by age, party, iterate through subsets
# Re-run the analysis for all counties across different political party and age group segments
for party in political_parties:
    for age_group in age_groups:
        segment_name = f"All County, {party}, {age_group}"
        subset = data[(data['Political_Party_grouped'] == party) & (data['AgeGroup'] == age_group)]
        results[segment_name] = analyze_subset_for_q9_q10(subset)

# Displaying the results for the first segment as an example
results_key = list(results.keys())[0]
results[results_key]



##make segments

def create_segments(data):
    # Get unique values for each attribute
    unique_counties = data['County'].unique().tolist() + ['!ALLEGHENY']
    age_groups = data['AgeGroup'].unique().tolist()
    political_parties = data['Political_Party_grouped'].unique().tolist()

    # Generate permutations of these unique values
    all_combinations = list(itertools.product(unique_counties, age_groups, political_parties))

    # Assign a unique number to each segment
    segments = {}
    for i, combination in enumerate(all_combinations, 1):
        segments[i] = combination

    return segments

def sort_data_into_segments(data, segments):
    sorted_data = {}
    for segment_number, (county, age_group, political_party) in segments.items():
        # Apply filters based on the segment definition
        df_filtered = dataframe.copy()

        if county != '!ALLEGHENY':
            df_filtered = df_filtered[df_filtered['County'] == county]
        else:
            df_filtered = df_filtered[df_filtered['County'] != 'ALLEGHENY']

        df_filtered = df_filtered[df_filtered['AgeGroup'] == age_group]
        df_filtered = df_filtered[df_filtered['Political_Party_grouped'] == political_party]

        sorted_data[segment_number] = df_filtered

    return sorted_data



segments = create_segments(data)
sorted_data = sort_data_into_segments(data, segments)

##make tables
# Function to process a batch of segments and return their LaTeX tables
def process_segments_to_latex(results_dict, start=1, end=54):
    latex_tables = {}
    for i, (segment_name, (q9_df, q10_df)) in enumerate(results_dict.items(), start=1):
        if start <= i <= end:  # Process only a subset for demonstration
            q9_caption = f"Weighted Percentages for Q9 Responses ({segment_name})"
            q10_caption = f"Weighted Percentages for Q10 Responses ({segment_name})"
            q9_label = f"tab:q9_{segment_name.lower().replace(' ', '_').replace(',', '')}"
            q10_label = f"tab:q10_{segment_name.lower().replace(' ', '_').replace(',', '')}"
            
            # Generate LaTeX tables
            q9_latex = format_latex_table(q9_df, q9_caption, q9_label)
            q10_latex = format_latex_table(q10_df, q10_caption, q10_label)
            
            latex_tables[segment_name] = (q9_latex, q10_latex)
            
    return latex_tables


##Section 23: Election trust and time spent online versus listening to radio
##cleaning media data for analysis
# Renaming the specified columns in the dataset
data = pd.DataFrame(columns={
    'Q1_1': 'television',
    'Q1_2': 'radio',
    'social_media': 'phone',
    'Q1_4': 'computer',
    'Q1_5': 'print materials'
})

# Confirm the columns have been renamed by displaying the updated column names
data_columns = data.columns.tolist()
data_columns


# Adjust the mapping to match the data format
value_mapping_adjusted = {
    'Less than 1 hour': 0.5,
    '1-2 hours': 1.5,
    '2-4 hours': 3,
    '4-6 hours': 5,
    '6 or more hours': 7,
    'Never': 0
}


#Reapply the value conversions to all columns with the adjusted mapping
for column in data.columns:
    data[column] = data[column].map(value_mapping_adjusted).fillna(data[column])



# Convert the specified columns to numeric, setting errors='coerce' to handle non-numeric values
columns_to_convert = ['computer', 'phone', 'radio', 'television', 'print materials']
for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Recalculate the 'online' and 'offline' columns after ensuring all values are numeric
# Recalculate 'online' and 'offline' treating NaN values as 0
data['online'] = data['computer'].fillna(0) + data['phone'].fillna(0)
data['offline'] = data['radio'].fillna(0) + data['television'].fillna(0) + data['print materials'].fillna(0)


##make comparision columns
# Define a function to compare two columns and return 'more', 'same', or 'less'
def compare_columns(a, b):
    if a > b:
        return 'more'
    elif a < b:
        return 'less'
    else:
        return 'same'

# Apply the comparison function to create the new comparison columns
data['Online vs Radio'] = data.apply(lambda row: compare_columns(row['online'], row['radio']), axis=1)
data['Online vs TV'] = data.apply(lambda row: compare_columns(row['online'], row['television']), axis=1)
data['Radio vs TV'] = data.apply(lambda row: compare_columns(row['radio'], row['television']), axis=1)

# Display a sample of the dataset to confirm the new comparison columns are added correctly
data[['online', 'radio', 'Online vs Radio', 'television', 'Online vs TV', 'Radio vs TV']].sample(5)

##make tables
# Prepare the data for cross-tabulation using 'Q1L' for Trust in Elections
crosstab_data_q1l = pd.crosstab(data['Online vs Radio'], data['Q1L'],
                                values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True)

# Perform the chi-square test on the adjusted data
chi2_q1l, p_q1l, dof_q1l, expected_q1l = chi2_contingency(crosstab_data_q1l.fillna(0))

# Round the cross-tabulation data to whole numbers
crosstab_data_q1l_rounded = crosstab_data_q1l.round()

# Convert the rounded data to LaTeX format
crosstab_latex = crosstab_data_q1l_rounded.to_latex(caption="Cross-tabulation of Trust in Elections (Q1L) vs. Online vs Radio",
                                                    label="tab:trust_elections_online_radio",
                                                    column_format='lcccc')

#add party
# Create a new column combining "Political_Party_grouped" and "Online vs Radio" for cross-tabulation
data['Party_Online_Radio'] = data['Political_Party_grouped'] + ' - ' + data['Online vs Radio']

# Cross-tabulate the new combined column with "Q1L" and calculate weighted counts
crosstab_party_online_radio_q1l = pd.crosstab(data['Party_Online_Radio'], data['Q1L'],
                                              values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation
chi2_party_online_radio_q1l, p_party_online_radio_q1l, dof_party_online_radio_q1l, expected_party_online_radio_q1l = chi2_contingency(crosstab_party_online_radio_q1l.fillna(0))

# Convert the cross-tabulation to LaTeX format with chi-square results in the caption
crosstab_party_online_radio_q1l_latex = crosstab_party_online_radio_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs Radio Comparison. Chi-square: {chi2_party_online_radio_q1l:.2f}, p-value: {p_party_online_radio_q1l:.2e}",
                                                                                label="tab:trust_elections_party_online_radio",
                                                                                column_format='lcccc')

crosstab_party_online_radio_q1l_latex

##Section 24: online vs televison
# Cross-tabulate "Q1L" with "Online vs TV" and perform chi-square test
crosstab_online_tv = pd.crosstab(data['Online vs TV'], data['Q1L'],
                                 values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_online_tv, p_online_tv, dof_online_tv, expected_online_tv = chi2_contingency(crosstab_online_tv.fillna(0))

# Convert the rounded data to LaTeX for "Online vs TV"
crosstab_online_tv_latex = crosstab_online_tv.to_latex(caption="Cross-tabulation of Trust in Elections (Q1L) vs. Online vs TV",
                                                       label="tab:trust_elections_online_tv",
                                                       column_format='lcccc')

##add party
# Update the combined column to include "Online vs TV" comparisons for each political party group
data['Party_Online_TV'] = data['Political_Party_grouped'] + ' - ' + data['Online vs TV']

# Cross-tabulate the new combined column with "Q1L" and calculate weighted counts for "Online vs TV"
crosstab_party_online_tv_q1l = pd.crosstab(data['Party_Online_TV'], data['Q1L'],
                                           values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation for "Online vs TV"
chi2_party_online_tv_q1l, p_party_online_tv_q1l, dof_party_online_tv_q1l, expected_party_online_tv_q1l = chi2_contingency(crosstab_party_online_tv_q1l.fillna(0))


# Convert the cross-tabulations to LaTeX format with chi-square results in the caption for both comparisons
crosstab_party_online_tv_q1l_latex = crosstab_party_online_tv_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs TV Comparison. Chi-square: {chi2_party_online_tv_q1l:.2f}, p-value: {p_party_online_tv_q1l:.2e}",
                                                                          label="tab:trust_elections_party_online_tv",
                                                                          column_format='lcccc')                                                                      label="tab:trust_elections_party_radio_tv",
                                                                          column_format='lcccc')



##section 25: radio vs television
# Repeat the process for "Radio vs TV"
crosstab_radio_tv = pd.crosstab(data['Radio vs TV'], data['Q1L'],
                                values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_radio_tv, p_radio_tv, dof_radio_tv, expected_radio_tv = chi2_contingency(crosstab_radio_tv.fillna(0))

# Convert the rounded data to LaTeX for "Radio vs TV"
crosstab_radio_tv_latex = crosstab_radio_tv.to_latex(caption="Cross-tabulation of Trust in Elections (Q1L) vs. Radio vs TV",
                                                     label="tab:trust_elections_radio_tv",
                                                     column_format='lcccc')

# Repeat the process for "Radio vs TV"
data['Party_Radio_TV'] = data['Political_Party_grouped'] + ' - ' + data['Radio vs TV']
crosstab_party_radio_tv_q1l = pd.crosstab(data['Party_Radio_TV'], data['Q1L'],
                                          values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_party_radio_tv_q1l, p_party_radio_tv_q1l, dof_party_radio_tv_q1l, expected_party_radio_tv_q1l = chi2_contingency(crosstab_party_radio_tv_q1l.fillna(0))


##make tables
crosstab_party_radio_tv_q1l_latex = crosstab_party_radio_tv_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Political Party and Radio vs TV Comparison. Chi-square: {chi2_party_radio_tv_q1l:.2f}, p-value: {p_party_radio_tv_q1l:.2e}",
                                                                                   label="tab:trust_elections_radio_tv",
                                                                                   column_format='lcccc')

   

##media comparisions for allegheny county
##Filter the dataset for ALLEGHENY county
allegheny_data = data[data['county'] == 'ALLEGHENY']

# Update the combined columns for the subset
allegheny_data['Party_Online_TV'] = allegheny_data['Political_Party_grouped'] + ' - ' + allegheny_data['Online vs TV']
allegheny_data['Party_Radio_TV'] = allegheny_data['Political_Party_grouped'] + ' - ' + allegheny_data['Radio vs TV']
allegheny_data['Party_Online_Radio'] = allegheny_data['Political_Party_grouped'] + ' - ' + allegheny_data['Online vs Radio']



# Cross-tabulate and perform chi-square test for "Online vs TV" in ALLEGHENY county
allegheny_crosstab_online_tv = pd.crosstab(allegheny_data['Party_Online_TV'], allegheny_data['Q1L'],
                                           values=allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_allegheny_online_tv, p_allegheny_online_tv, _, _ = chi2_contingency(allegheny_crosstab_online_tv.fillna(0))

# Cross-tabulate and perform chi-square test for "Radio vs TV" in ALLEGHENY county
allegheny_crosstab_radio_tv = pd.crosstab(allegheny_data['Party_Radio_TV'], allegheny_data['Q1L'],
                                          values=allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_allegheny_radio_tv, p_allegheny_radio_tv, _, _ = chi2_contingency(allegheny_crosstab_radio_tv.fillna(0))

# Cross-tabulate and perform chi-square test for "Online vs Radio" in ALLEGHENY county
allegheny_crosstab_online_radio = pd.crosstab(allegheny_data['Party_Online_Radio'], allegheny_data['Q1L'],
                                              values=allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_allegheny_online_radio, p_allegheny_online_radio, _, _ = chi2_contingency(allegheny_crosstab_online_radio.fillna(0))



# Format the cross-tabulations for LaTeX
allegheny_online_tv_latex = allegheny_crosstab_online_tv.to_latex(caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs TV. Chi-square: {chi2_allegheny_online_tv:.2f}, p-value: {p_allegheny_online_tv:.2e}",
                                                                  label="tab:allegheny_trust_elections_party_online_tv",
                                                                  column_format='lcccc')
allegheny_radio_tv_latex = allegheny_crosstab_radio_tv.to_latex(caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Radio vs TV. Chi-square: {chi2_allegheny_radio_tv:.2f}, p-value: {p_allegheny_radio_tv:.2e}",
                                                                label="tab:allegheny_trust_elections_party_radio_tv",
                                                                column_format='lcccc')
allegheny_online_radio_latex = allegheny_crosstab_online_radio.to_latex(caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs Radio. Chi-square: {chi2_allegheny_online_radio:.2f}, p-value: {p_allegheny_online_radio:.2e}",
     
##repeat for !ALLEGHENY
# Filter the dataset to exclude ALLEGHENY county
not_allegheny_data = data[data['county'] != 'ALLEGHENY']

# Update the combined columns for the subset excluding ALLEGHENY
not_allegheny_data['Party_Online_TV'] = not_allegheny_data['Political_Party_grouped'] + ' - ' + not_allegheny_data['Online vs TV']
not_allegheny_data['Party_Radio_TV'] = not_allegheny_data['Political_Party_grouped'] + ' - ' + not_allegheny_data['Radio vs TV']
not_allegheny_data['Party_Online_Radio'] = not_allegheny_data['Political_Party_grouped'] + ' - ' + not_allegheny_data['Online vs Radio']

# Cross-tabulate and perform chi-square test for "Online vs TV" excluding ALLEGHENY
not_allegheny_crosstab_online_tv = pd.crosstab(not_allegheny_data['Party_Online_TV'], not_allegheny_data['Q1L'],
                                               values=not_allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_not_allegheny_online_tv, p_not_allegheny_online_tv, _, _ = chi2_contingency(not_allegheny_crosstab_online_tv.fillna(0))

# Cross-tabulate and perform chi-square test for "Radio vs TV" excluding ALLEGHENY
not_allegheny_crosstab_radio_tv = pd.crosstab(not_allegheny_data['Party_Radio_TV'], not_allegheny_data['Q1L'],
                                              values=not_allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_not_allegheny_radio_tv, p_not_allegheny_radio_tv, _, _ = chi2_contingency(not_allegheny_crosstab_radio_tv.fillna(0))

# Cross-tabulate and perform chi-square test for "Online vs Radio" excluding ALLEGHENY
not_allegheny_crosstab_online_radio = pd.crosstab(not_allegheny_data['Party_Online_Radio'], not_allegheny_data['Q1L'],
                                                  values=not_allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()
chi2_not_allegheny_online_radio, p_not_allegheny_online_radio, _, _ = chi2_contingency(not_allegheny_crosstab_online_radio.fillna(0))

# Format the cross-tabulations for LaTeX with chi-square results in the caption for all three comparisons
not_allegheny_online_tv_latex = not_allegheny_crosstab_online_tv.to_latex(caption=f"Excluding ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs TV. Chi-square: {chi2_not_allegheny_online_tv:.2f}, p-value: {p_not_allegheny_online_tv:.2e}",
                                                                          label="tab:not_allegheny_trust_elections_party_online_tv",
                                                                          column_format='lcccc')
not_allegheny_radio_tv_latex = not_allegheny_crosstab_radio_tv.to_latex(caption=f"Excluding ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Radio vs TV. Chi-square: {chi2_not_allegheny_radio_tv:.2f}, p-value: {p_not_allegheny_radio_tv:.2e}",
                                                                         label="tab:not_allegheny_trust_elections_party_radio_tv",
                                                                         column_format='lcccc')
not_allegheny_online_radio_latex = not_allegheny_crosstab_online_radio.to_latex(caption=f"Excluding ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Online vs Radio. Chi-square: {chi2_not_allegheny_online_radio:.2f}, p-value: {p_not_allegheny_online_radio:.2e}",
                                                                                label="tab:not_allegheny_trust_elections_party_online_radio"
  
 ##Section 26: Number of Facebook groups
# Convert "Number of Groups" to numeric, treating non-numeric values as NaN
data['Number of Groups Numeric'] = pd.to_numeric(data['Number of Groups'], errors='coerce')

# Re-categorize the "Number of Groups" column with numeric conversion
data['Groups_Categorized'] = data['Number of Groups Numeric'].apply(
    lambda x: 'More than 5' if x >= 5 else '5 or less' if pd.notnull(x) else np.nan
)

# Display a sample of the dataset to confirm the new categorization
data[['Number of Groups', 'Number of Groups Numeric', 'Groups_Categorized']].sample(5)
                                                                  
##make cross tabs
# Cross-tabulate "Q1L" with "Groups_Categorized" and calculate weighted counts
crosstab_groups_q1l = pd.crosstab(data['Groups_Categorized'], data['Q1L'],
                                  values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation
chi2_groups_q1l, p_groups_q1l, dof_groups_q1l, expected_groups_q1l = chi2_contingency(crosstab_groups_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption
groups_q1l_latex = crosstab_groups_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Number of Groups. Chi-square: {chi2_groups_q1l:.2f}, p-value: {p_groups_q1l:.2e}",
                                                label="tab:trust_elections_groups",
                                                column_format='lcccc')

##add party
# Create a new column combining "Political_Party_Grouped" and "Groups_Categorized"
data['Party_Groups'] = data['Political_Party_grouped'] + ' - ' + data['Groups_Categorized']

# Cross-tabulate "Q1L" with the new combined column and calculate weighted counts
crosstab_party_groups_q1l = pd.crosstab(data['Party_Groups'], data['Q1L'],
                                        values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation
chi2_party_groups_q1l, p_party_groups_q1l, dof_party_groups_q1l, expected_party_groups_q1l = chi2_contingency(crosstab_party_groups_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption
party_groups_q1l_latex = crosstab_party_groups_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Political Party and Number of Groups. Chi-square: {chi2_party_groups_q1l:.2f}, p-value: {p_party_groups_q1l:.2e}",
                                                            label="tab:trust_elections_party_groups",
                                                            column_format='lcccc')
##look at allegheny
# Filter the dataset for ALLEGHENY county 
allegheny_data = data_new_renamed[data['county'] == 'ALLEGHENY']

# Create a new combined column for the ALLEGHENY subset
allegheny_data['Party_Groups'] = allegheny_data['Political_Party_grouped'] + ' - ' + allegheny_data['Groups_Categorized']

# Cross-tabulate "Q1L" with the new combined column for ALLEGHENY and calculate weighted counts
allegheny_crosstab_party_groups_q1l = pd.crosstab(allegheny_data['Party_Groups'], allegheny_data['Q1L'],
                                                   values=allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation for ALLEGHENY
chi2_allegheny_party_groups_q1l, p_allegheny_party_groups_q1l, _, _ = chi2_contingency(allegheny_crosstab_party_groups_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption for ALLEGHENY
allegheny_party_groups_q1l_latex = allegheny_crosstab_party_groups_q1l.to_latex(
    caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Number of Groups. Chi-square: {chi2_allegheny_party_groups_q1l:.2f}, p-value: {p_allegheny_party_groups_q1l:.2e}",
    label="tab:allegheny_trust_elections_party_groups",
    column_format='lcccc'
)

#look at not allegheny
# Filter the dataset to exclude ALLEGHENY county again
not_allegheny_data = data_new_renamed[data_new_renamed['county'] != 'ALLEGHENY']

# Create a new combined column for the subset excluding ALLEGHENY
not_allegheny_data['Party_Groups'] = not_allegheny_data['Political_Party_grouped'] + ' - ' + not_allegheny_data['Groups_Categorized']

# Cross-tabulate "Q1L" with the new combined column for the subset excluding ALLEGHENY and calculate weighted counts
not_allegheny_crosstab_party_groups_q1l = pd.crosstab(not_allegheny_data['Party_Groups'], not_allegheny_data['Q1L'],
                                                       values=not_allegheny_data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation for the subset excluding ALLEGHENY
chi2_not_allegheny_party_groups_q1l, p_not_allegheny_party_groups_q1l, _, _ = chi2_contingency(not_allegheny_crosstab_party_groups_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption for the subset excluding ALLEGHENY
not_allegheny_party_groups_q1l_latex = not_allegheny_crosstab_party_groups_q1l.to_latex(
    caption=f"Excluding ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Number of Groups. Chi-square: {chi2_not_allegheny_party_groups_q1l:.2f}, p-value: {p_not_allegheny_party_groups_q1l:.2e}",
    label="tab:not_allegheny_trust_elections_party_groups",
    column_format='lcccc'
)

##Section 27: Local Facebook group

# Cross-tabulate "Q1L" with "Local Facebook Group" and calculate weighted counts for the entire dataset
crosstab_fb_group_q1l = pd.crosstab(data['Local Facebook Group'], data['Q1L'],
                                    values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation
chi2_fb_group_q1l, p_fb_group_q1l, dof_fb_group_q1l, expected_fb_group_q1l = chi2_contingency(crosstab_fb_group_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption for the entire dataset
fb_group_q1l_latex = crosstab_fb_group_q1l.to_latex(caption=f"Cross-tabulation of Trust in Elections (Q1L) by Local Facebook Group. Chi-square: {chi2_fb_group_q1l:.2f}, p-value: {p_fb_group_q1l:.2e}",
                                                    label="tab:trust_elections_fb_group",
                                                    column_format='lcccc')
##add political party
# Create a new combined column for the entire dataset using "Local Facebook Group" and "Political_Party_grouped"
data['Party_LocalFB'] = data['Political_Party_grouped'] + ' - ' + data['Local Facebook Group'].astype(str)

# Cross-tabulate "Q1L" with the new combined column for the entire dataset and calculate weighted counts
crosstab_party_localfb_q1l = pd.crosstab(data['Party_LocalFB'], data['Q1L'],
                                         values=data['combined_weight_rescaled_new'], aggfunc='sum', dropna=True).round()

# Perform chi-square test on the cross-tabulation for the entire dataset
chi2_party_localfb_q1l, p_party_localfb_q1l, _, _ = chi2_contingency(crosstab_party_localfb_q1l.fillna(0))

# Format the cross-tabulation to LaTeX with chi-square results in the caption for the entire dataset
party_localfb_q1l_latex = crosstab_party_localfb_q1l.to_latex(
    caption=f"Cross-tabulation of Trust in Elections (Q1L) by Political Party and Local Facebook Group. Chi-square: {chi2_party_localfb_q1l:.2f}, p-value: {p_party_localfb_q1l:.2e}",
    label="tab:trust_elections_party_localfb",
    column_format='lcccc'
)

#politcal party and allegheny
# Format the cross-tabulation to LaTeX with chi-square results in the caption for ALLEGHENY
allegheny_party_localfb_q1l_latex = allegheny_crosstab_party_localfb_q1l.to_latex(
    caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Local Facebook Group. Chi-square: {chi2_allegheny_party_localfb_q1l:.2f}, p-value: {p_allegheny_party_localfb_q1l:.2e}",
    label="tab:allegheny_trust_elections_party_localfb",
    column_format='lcccc'
)

##political party and not allegheny
# Format the cross-tabulation to LaTeX with chi-square results in the caption for ALLEGHENY
allegheny_party_localfb_q1l_latex = allegheny_crosstab_party_localfb_q1l.to_latex(
    caption=f"ALLEGHENY: Cross-tabulation of Trust in Elections (Q1L) by Political Party and Local Facebook Group. Chi-square: {chi2_allegheny_party_localfb_q1l:.2f}, p-value: {p_allegheny_party_localfb_q1l:.2e}",
    label="tab:allegheny_trust_elections_party_localfb",
    column_format='lcccc'
)



##Section 28: Raw Count of Respondents by County
# Count the occurrences of each category in the 'county' column
county_counts = data['county'].value_counts()

# Display the counts
print(county_counts)