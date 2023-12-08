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
from scipy.stats import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'updated_final_merged_dataset_media.csv'  
data = pd.read_csv(file_path)

## Define relevant subsets
##all data
subset_all_geographies = data.copy()

##All Data, Republican and Other
subset_republican_other = data[(data['Political_Party'] == 'Republican') | (data['Political_Party'] == 'Other')]

##All Data, Republican

subset_republican = data[(data['Political_Party'] == 'Republican')]

##All Data, Other

subset_other = data[(data['Political_Party'] == 'Other')]

##All Data, Democrat


subset_democrat= data[(data['Political_Party'] == 'Democrat')]

##Allegheny, all parties


subset_allegheny = data[data['county'] == 'ALLEGHENY']

##Allegheny, Republican and Other

subset_allegheny_republican_other = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party'] == 'Republican') | (data['Political_Party'] == 'Other')]


##Allegheny, Republican
subset_allegheny_republican = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party'] == 'Republican')]

##Allegheny, Other
subset_allegheny_other = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party'] == 'Republican')]


##Allegheny, Democrat
subset_allegheny_democrat = data[(data['county'] == 'ALLEGHENY') & (data['Political_Party'] == 'Democrat')]


##Not Allegheny, all parties
##define other counties
other_counties = ['Butler', 'Beaver', 'Washington', 'Westmoreland', 'Fayette', 'Armstrong']
subset_other_counties = data[data['county'].isin(other_counties)]

##Not Allegheny, Republican and Other
subset_other_counties_republican_other = data[data['county'].isin(other_counties) &(data['Political_Party'] == 'Republican') | (data['Political_Party'] == 'Other')]

##Not Allegheny, Republican
subset_other_counties_republican = data[data['county'].isin(other_counties) &(data['Political_Party'] == 'Republican')]


##Not ALlegheny, Other
subset_other_counties_other = data[data['county'].isin(other_counties)& (data['Political_Party'] == 'Other')]


##Not Alleghenry, Democrat
subset_other_counties_democrat= data[data['county'].isin(other_counties)& (data['Political_Party'] == 'Democrat')]



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

relevant_columns = [
    'election_trust', 'AgeGroup', 'Race_grouped', 'Income_grouped', 
    'Education_grouped', 'Health_Group', 'Gender', 
    'Local Facebook Group', 'Number of Groups Categorized', 'combined_weight_rescaled'
]

# Make tables
latex_table1 = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table2= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table3 = generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')
latex_table4 = generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')
latex_table5 = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table6 = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7 = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table8 = generate_latex_table_from_subset(subset_allegheny_republican, relevant_columns, 'combined_weight_rescaled')
latex_table9 = generate_latex_table_from_subset(subset_allegheny_other, relevant_columns, 'combined_weight_rescaled')
latex_table10 = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table11 = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12 = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table13 = generate_latex_table_from_subset(subset_other_counties_republican, relevant_columns, 'combined_weight_rescaled')
latex_table14 = generate_latex_table_from_subset(subset_other_counties_other, relevant_columns, 'combined_weight_rescaled')
latex_table15 = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')


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
    'Local Facebook Group', 'Number of Groups Categorized'
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


def perform_weighted_logit_regression(subset, dependent_var, independent_vars, weight_column):
    """
    Perform a weighted logit regression on a given subset of data and output results as a LaTeX table.

    :param subset: DataFrame, the subset of data to analyze.
    :param dependent_var: str, the name of the dependent variable.
    :param independent_vars: List[str], list of independent variables.
    :param weight_column: str, the name of the column containing weights.
    :return: LaTeX formatted string of the logit regression results.
    """
    # Selecting the dependent and independent variables
    X = subset[independent_vars]
    y = subset[dependent_var]
    weights = subset[weight_column]

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
independent_vars = ['Gender', 'AgeGroup', 'Race_grouped', 'Income_grouped', 'Education_grouped', 'Health_Group',
'Local Facebook Group', 'social_media']  # Replace with your actual independent variables
weight_column = 'combined_weight_rescaled'  

# Run function for required subsets
latex_table1A = perform_weighted_logit_regression(subset_all_geographies, dependent_var, independent_vars, weight_column)
latex_table2A = perform_weighted_logit_regression(subset_republican_other, dependent_var, independent_vars, weight_column)
latex_table3A = perform_weighted_logit_regression(subset_democrat, dependent_var, independent_vars, weight_column)
latex_table4A = perform_weighted_logit_regression(subset_allegheny, dependent_var, independent_vars, weight_column)
latex_table5A = perform_weighted_logit_regression(subset_allegheny_republican_other, dependent_var, independent_vars, weight_column)
latex_table6A = perform_weighted_logit_regression(subset_allegheny_democrat, dependent_var, independent_vars, weight_column)
latex_table7A = perform_weighted_logit_regression(subset_other_counties, dependent_var, independent_vars, weight_column)
latex_table8A = perform_weighted_logit_regression(subset_other_counties_republican_other, dependent_var, independent_vars, weight_column)
latex_table9A = perform_weighted_logit_regression(subset_other_counties_democrat, dependent_var, independent_vars, weight_column)


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
group_var = 'Gender'
weight_var = 'combined_weight_rescaled'  

# Perform the weighted t-test for each subset
t_stat_all, p_value_all, df_all = weighted_t_test(subset_all_geographies, test_var, group_var, weight_var)
t_stat_all_RO, p_value_all_RO, df_all_RO= weighted_t_test(subset_republican_other, test_var, group_var, weight_var)
t_stat_all_D, p_value_all_D, df_all_D = weighted_t_test(subset_democrat, test_var, group_var, weight_var)
t_stat_A, p_value_A, df_A = weighted_t_test(subset_allegheny, test_var, group_var, weight_var)
t_stat_A_RO, p_value_A_RO, df_A_RO = weighted_t_test(subset_allegheny_republican_other, test_var, group_var, weight_var)
t_stat_A_D, p_value_A_D, df_A_D = weighted_t_test(subset_allegheny_democrat, test_var, group_var, weight_var)
t_stat_6, p_value_6, df_6 = weighted_t_test(subset_other_counties, test_var, group_var, weight_var)
t_stat_6_RO, p_value_6_RO, df_6_RO = weighted_t_test(subset_other_counties_republican_other, test_var, group_var, weight_var)
t_stat_6_D, p_value_6_D, df_6_D = weighted_t_test(subset_other_counties_democrat, test_var, group_var, weight_var)

##repeat for Race-- race is binarized to white/ non-white
group_var= 'Gender'




# Perform the weighted t-test for each subset
t_stat_all2, p_value_all2, df_all2 = weighted_t_test(subset_all_geographies, test_var, group_var, weight_var)
t_stat_all_RO2, p_value_all_RO2, df_all_RO2= weighted_t_test(subset_republican_other, test_var, group_var, weight_var)
t_stat_all_D2, p_value_all_D2, df_all_D2 = weighted_t_test(subset_democrat, test_var, group_var, weight_var)
t_stat_A2, p_value_A2, df_A2 = weighted_t_test(subset_allegheny, test_var, group_var, weight_var)
t_stat_A_RO2, p_value_A_RO2, df_A_RO2 = weighted_t_test(subset_allegheny_republican_other, test_var, group_var, weight_var)
t_stat_A_D2, p_value_A_D2, df_A_D2 = weighted_t_test(subset_allegheny_democrat, test_var, group_var, weight_var)
t_stat_62, p_value_62, df_62 = weighted_t_test(subset_other_counties, test_var, group_var, weight_var)
t_stat_6_RO2, p_value_6_RO2, df_6_RO2 = weighted_t_test(subset_other_counties_republican_other, test_var, group_var, weight_var)
t_stat_6_D2, p_value_6_D2, df_6_D2 = weighted_t_test(subset_other_counties_democrat, test_var, group_var, weight_var)


##Section 5: Understanding Distrust
##Correlation Matrix
def weighted_correlation_matrix(data, variables, weight_column):
    """
    Calculate a weighted correlation matrix for specified variables.

    :param data: DataFrame, the dataset containing the variables.
    :param variables: List[str], list of variables to include in the correlation matrix.
    :param weight_column: str, the name of the column containing weights.
    :return: DataFrame, weighted correlation matrix.
    """
    def expand_data(data, weight_col):
        return data.loc[data.index.repeat(data[weight_col].astype(int))]

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

    return corr_matrix

def correlation_matrix_to_latex(corr_matrix):
    """
    Convert a correlation matrix to a LaTeX formatted table.

    :param corr_matrix: DataFrame, the correlation matrix to format.
    :return: str, LaTeX formatted table.
    """
    latex_table = corr_matrix.to_latex(float_format="%.2f")
    return latex_table

# Plug in variables
variables = ['Q1L', 'Q2L', 'Q3L', 'Q4L']  
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

relevant_columns = ['election_trust', 'Q3L', 'combined_weight_rescaled']

# Make tables
latex_table1C = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table2C= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table5C = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table6C = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7C = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table10C = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table11C = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12C = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table15C = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')


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

# Apply the mapping to the Q11 column
data['Q11_grouped'] = data['Q11'].map(response_mapping)

# Checking the first few rows to verify the transformation
print(data[['Q11', 'Q11_grouped']].head())

##Make the plot
def create_weighted_histogram(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Calculate the weighted percentages
    weighted_counts = subset.groupby(['Q11_grouped', 'Political_Party_grouped']).apply(lambda x: np.sum(x['combined_weight_rescaled']) / np.sum(subset['combined_weight_rescaled']))
    weighted_counts = weighted_counts.reset_index(name='Weighted_Percentage')

    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Q11_grouped', y='Weighted_Percentage', hue='Political_Party_grouped', data=weighted_counts, palette=colors)

    # Adding labels and title
    plt.xlabel('Q11 Grouped Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

# Work with subsets
create_weighted_histogram(subset_all_geographies, 'combined_weight_rescaled', 'All Geographies')
create_weighted_histogram(subset_allegheny, 'combined_weight_rescaled', 'Allegheny County')
create_weighted_histogram(subset_other_counties, 'combined_weight_rescaled', 'Other Counties')


##Perceptions of Misinformation Consumption by Friends and Family

def create_weighted_histogram_with_percentages(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset with percentages displayed on top of each bar.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Calculate the weighted percentages
    weighted_counts = subset.groupby(['Q14', 'Political_Party_grouped']).apply(lambda x: np.sum(x['combined_weight_rescaled']) / np.sum(subset['combined_weight_rescaled']))
    weighted_counts = weighted_counts.reset_index(name='Weighted_Percentage')

    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Q14', y='Weighted_Percentage', hue='Political_Party_grouped', data=weighted_counts, palette=colors)

    # Adding percentages on top of each bar
    for p in sns_barplot.patches:
        sns_barplot.annotate(format(p.get_height(), '.1%'), 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points')

    # Adding labels and title
    plt.xlabel('Q14 Responses')
    plt.ylabel('Weighted Percentage')
    plt.title(title)

    # Show the plot
    plt.show()

#Subsets
create_weighted_histogram_with_percentages(subset_all_geographies, 'combined_weight_rescaled', 'Concern about Friend`s Exposure to Misinformation')
create_weighted_histogram_with_percentages(subset_allegheny, 'combined_weight_rescaled', 'Concern about Friend`s Exposure to Misinformation')
create_weighted_histogram_with_percentages(subset_other_counties, 'combined_weight_rescaled', 'Concern about Friend`s Exposure to Misinformation')

###Section 9: Perceptions of Misinformation Consumption by people respondent disagrees with

def create_weighted_histogram_with_percentages(subset, weight_column, title):
    """
    Create a weighted histogram for a given subset with percentages displayed on top of each bar.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Calculate the weighted percentages
    weighted_counts = subset.groupby(['Q17', 'Political_Party_grouped']).apply(lambda x: np.sum(x['combined_weight_rescaled']) / np.sum(subset['combined_weight_rescaled']))
    weighted_counts = weighted_counts.reset_index(name='Weighted_Percentage')

    # Colors for each political party
    colors = {'Democrat': 'blue', 'Other': 'purple', 'Republican': 'red'}

    # Plotting
    plt.figure(figsize=(10, 6))
    sns_barplot = sns.barplot(x='Q14', y='Weighted_Percentage', hue='Political_Party_grouped', data=weighted_counts, palette=colors)

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
create_weighted_histogram_with_percentages(subset_all_geographies, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')
create_weighted_histogram_with_percentages(subset_allegheny, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')
create_weighted_histogram_with_percentages(subset_other_counties, 'combined_weight_rescaled', 'Perceptions of Misinformation Consumption by people respondent disagrees with')

##Section 10: Comparing Perceptions of Misinformation

def create_comparison_histogram(subset, weight_column, title):
    """
    Create a histogram comparing Q14 and Q17 with weighted percentages.

    :param subset: DataFrame, the subset of data to plot.
    :param weight_column: str, the name of the column containing weights.
    :param title: str, the title of the plot.
    """
    # Preparing the data
    subset_q14 = subset[['Q14', weight_column]].copy()
    subset_q14['Question'] = 'Friends'
    subset_q14.rename(columns={'Q14': 'Response'}, inplace=True)

    subset_q17 = subset[['Q17', weight_column]].copy()
    subset_q17['Question'] = 'Opponents'
    subset_q17.rename(columns={'Q17': 'Response'}, inplace=True)

    combined_data = pd.concat([subset_q14, subset_q17])

    # Calculate weighted percentages
    combined_data['Weighted_Count'] = combined_data.groupby(['Question', 'Response'])['combined_weight_rescaled'].transform('sum')
    combined_data['Total_Weight'] = combined_data['combined_weight_rescaled'].sum()
    combined_data['Weighted_Percentage'] = combined_data['Weighted_Count'] / combined_data['Total_Weight']

    # Drop duplicates
    combined_data.drop_duplicates(subset=['Question', 'Response'], inplace=True)

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
create_comparison_histogram(subset_other_counties, 'combined_weight_rescaled', 'Distribution Comparision of Q14 and Q17 (not_allegheny)')


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
latex_table2D= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table5D = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table6D = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7D = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table10D = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table11D = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12D = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table15D = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')


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

    subset_fox_wjas = subset[['Fox News /WJAS', 'combined_weight_rescaled']].copy()
    subset_fox_wjas['Media_Outlet'] = 'Fox News /WJAS'

    combined_data = pd.concat([subset_kdka, subset_word, subset_fox_wjas])
    combined_data.rename(columns={'KDKA AM': 'Response', 'Word 101.5 FM': 'Response', 'Fox News /WJAS': 'Response'}, inplace=True)

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
relevant_columns = ['election_trust', 'Fox News/WJAS', 'KDKA AM', 'Word 101.5 FM' 'combined_weight_rescaled']

# Make table
latex_table1E = generate_latex_table_from_subset(subset_republican, relevant_columns, 'combined_weight_rescaled')


##Conservative Radio and Election Trust Among Other
relevant_columns = ['election_trust', 'Fox News/WJAS', 'KDKA AM', 'Word 101.5 FM' 'combined_weight_rescaled']

# Make table
latex_table1F = generate_latex_table_from_subset(subset_other, relevant_columns, 'combined_weight_rescaled')


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
    # Categorizing respondents
    conditions = [
        (subset['Fox_viewers'].notna()) & (subset['WJAS/Fox'].isna()),
        (subset['Fox_viewers'].isna()) & (subset['WJAS/Fox'].notna()),
        (subset['Fox_viewers'].notna()) & (subset['WJAS/Fox'].notna()),
        (subset['Fox_viewers'].isna()) & (subset['WJAS/Fox'].isna())
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

# plug in 
subset = data  # Replace with your actual DataFrame
weight_column = 'combined_weight_rescaled'  # Replace with your actual weight column

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
subsets = [subset_all_geographies, subset_allegheny, subset_other_counties, subset_republian, subset_other, subset_democrat]  # Replace with your actual subsets
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
subsets = [subset_republicans, subset_allegheny_republicans, subset_other_counties_republicans, subset_other, subset_allegheny_other, subset_other_counties_other, subset_democrat, subset_allegheny_democrat, subset_other_counties_democrat]  # Replace with your actual subsets
subset_names = ['Republicans in whole MSA', 'Republicans in Allegheny', 'Republicans outside of Allegheny', 'Other in whole MSA', 'Other in Allegheny', 'Other outside of Allegheny', 'Democrats in whole MSA', 'Democrats in Allegheny', 'Democrats outside of Allegheny']  # Names for each subset

# Iterate through subsets and create LaTeX tables
for subset, name in zip(subsets, subset_names):
    weighted_percentages = calculate_weighted_percentages_for_question(subset, '1QL', 'combined_weight_rescaled')  
    latex_table = weighted_percentages_to_latex(weighted_percentages, '1QL Responses - ')
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
def perform_weighted_logit_regression(subset, dependent_var, independent_vars, weight_column):
    """
    Perform a weighted logit regression on a given subset of data and output results as a LaTeX table.

    :param subset: DataFrame, the subset of data to analyze.
    :param dependent_var: str, the name of the dependent variable.
    :param independent_vars: List[str], list of independent variables.
    :param weight_column: str, the name of the column containing weights.
    :return: LaTeX formatted string of the logit regression results.
    """
    # Selecting the dependent and independent variables
    X = subset[independent_vars]
    y = subset[dependent_var]
    weights = subset[weight_column]

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
independent_vars = ['fox_viewers', 'county', 'AgeGroup', 'Political_Party_grouped', 'Q2l']  # Replace with your actual independent variables
weight_column = 'combined_weight_rescaled'  

# Run function for required subsets
latex_table1M = perform_weighted_logit_regression(subset_all_geographies, dependent_var, independent_vars, weight_column)


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
weighted_pivot = calculate_weighted_percentages_by_group(subset, 'Political_Party_grouped', 'In Local Facebook Group', 'combined_weight_rescaled')
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
relevant_columns = ['election_trust',  'Local Facebook Group', 'combined_weight_rescaled']

# Make tables
latex_table1N = generate_latex_table_from_subset(subset_all_geographies, relevant_columns, 'combined_weight_rescaled')
latex_table2N= generate_latex_table_from_subset(subset_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table5N = generate_latex_table_from_subset(subset_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table6N = generate_latex_table_from_subset(subset_allegheny, relevant_columns, 'combined_weight_rescaled')
latex_table7N = generate_latex_table_from_subset(subset_allegheny_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table10N = generate_latex_table_from_subset(subset_allegheny_democrat, relevant_columns, 'combined_weight_rescaled')
latex_table11N = generate_latex_table_from_subset(subset_other_counties, relevant_columns, 'combined_weight_rescaled')
latex_table12N = generate_latex_table_from_subset(subset_other_counties_republican_other, relevant_columns, 'combined_weight_rescaled')
latex_table15N = generate_latex_table_from_subset(subset_other_counties_democrat, relevant_columns, 'combined_weight_rescaled')



##Section 22: Raw Count of Respondents by County
# Count the occurrences of each category in the 'county' column
county_counts = data['county'].value_counts()

# Display the counts
print(county_counts)