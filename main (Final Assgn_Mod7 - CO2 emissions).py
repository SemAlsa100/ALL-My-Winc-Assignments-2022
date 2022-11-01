
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import copy
from copy import deepcopy
import sys

import sklearn as sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy.polynomial.polynomial as poly
from pingouin import corr as pingouin_corr
import seaborn as sns  # Convention alias for Seaborn
from IPython import get_ipython


pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 500)


# Collect datasets to study the subject of CO2 Emissions
# ------------------------------------------------------
df_energy_data = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\owid-energy-data.csv", sep=",")
df_energy_data.head(20)
df_energy_data.shape
# OUTPUT:   (22343, 128)
df_energy_data.columns
# OUTPUT:  ['country', 'year', 'iso_code', 'population', 'gdp',
#    'biofuel_cons_change_pct', 'biofuel_cons_change_twh',
#    'biofuel_cons_per_capita', 'biofuel_consumption',
#    'biofuel_elec_per_capita',
#    ...
#    'solar_share_elec', 'solar_share_energy', 'wind_cons_change_pct',
#    'wind_cons_change_twh', 'wind_consumption', 'wind_elec_per_capita',
#    'wind_electricity', 'wind_energy_per_capita', 'wind_share_elec',
#    'wind_share_energy']


df_energy_codebook = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\owid-energy-codebook.csv", sep='|')
df_energy_codebook.head(20)
df_energy_codebook.shape
# OUTPUT:   (128, 3)
df_energy_codebook.columns
# OUTPUT:   ['column', 'description', 'source']


# *************************************************************


# get an object name from globals (any object)
def get_obj_name(obj):
    name = [x for x in globals() if globals()[x] is obj][0]
    return name


# analze columns dataframes
def analyzeDataframes(myAnalysisDF, myAnalysisColumns):
    print(' ')
    print('*' * 60)
    print(f'RESULTS ANALYSIS DATAFRAME "{get_obj_name(myAnalysisDF)}"')
    print(f"AND ITS COLUMNS:")
    print(f"{myAnalysisColumns}")
    print('*' * 60)
    print(f"{get_obj_name(myAnalysisDF):}")
    print(f"{myAnalysisDF.shape= }")
    print('*' * 40)
    print("myAnalysisDF.head(50)=")
    print(f"{myAnalysisDF.head(50)}")
    print("myAnalysisDF.tail(50)=")
    print(f"{myAnalysisDF.tail(50)}")
    print('*' * 40)

    print("myAnalysisDF.value_counts()=")
    print(f"{myAnalysisDF.value_counts()}")
    print('*' * 40)

    for myCol in myAnalysisColumns:
        try:
            print(f"myAnalysisDF.{myCol}.value_counts().sum()=")
            print(f"{myAnalysisDF[myCol].value_counts().sum()}")
            print('*' * 40)
        except AttributeError as error:     # Errors:   AttributeError, IndexError, ValueError, TypeError
            print(f"The {myCol} sum Error is: {error}.")
        # finally:
            print('*' * 40)

        try:
            print(f"myAnalysisDF.{myCol}.unique().shape=")
            print(f"{myAnalysisDF[myCol].unique().shape}")
            print('*' * 40)
        except AttributeError as error:
            print(f"The {myCol} shape Error is: {error}.")
        # finally:
            print('*' * 40)
            # return

        print(f"myAnalysisDF.{myCol}.value_counts()")
        print(f"{myAnalysisDF[myCol].value_counts()}")
        print('*' * 40)
        print('End*' * 10)
        print('*' * 40)
        print(' ')
        print('*' * 40)


# ***************************************************************************
# 7. for each column:
#    1.	find any non-default missing values (there may be more than one)
#    2.	decide what default NULL values to replace them with
#    3.	replace the missing values
#    4.	calculate the percentage of missing values
#
# Tactics:
# -------
#    A. looking at all the unique values
#    B. sorting and looking at the edges
#    C. casting to a type
#    D. looking at the frequency
#
def find_missing_values(myDF, column_name):

    def print_separator(sep, num, msg):
        print("\n")
        print(sep * num)
        print(f"{msg}")
        print(sep * num)

    def look_at_unique_values(column):
        unique_values_cutoff = 160  # bv. 50 or 160
        unique_values = column.unique()
        num_unique_values = len(unique_values)
        if num_unique_values == len(column):
            print(
                f"Each value in the column is unique (total: {num_unique_values})")
        elif num_unique_values < unique_values_cutoff:
            print(f"Less than {unique_values_cutoff} unique values:")
            # We may get an error when sorting
            try:
                sorted = np.sort(unique_values)
                print("Values are sorted")
                print(list(sorted))
            except:
                print("Could not sort values")
                print(list(unique_values))
        else:
            print(
                f"More than {unique_values_cutoff} unique values (total: {num_unique_values})")

    def look_at_edges(df, column_name):
        # inner function
        def show_head_and_tail(values):
            num_items_to_slice = 10
            print(list(values)[:num_items_to_slice])
            print(list(values)[-num_items_to_slice:])

        column = df[column_name]
        unique_values = column.unique()
        try:
            sorted = np.sort(unique_values)
            print("Unique values sorted, head and tail:")
            print("\n")
            show_head_and_tail(sorted)
        except TypeError as error:
            print(f"Could not sort values: {error}")
            print("..so let's try filtering NULL values and then sorting")
            non_null_uniques = df.loc[~df[column_name].isnull(
            ), column_name].unique()
            sorted = np.sort(non_null_uniques)
            print("\n")
            show_head_and_tail(sorted)

    def cast_to_type(column):
        myDataTypes = ["datetime64", "int64", "float64", "string", "bool"]

        # stopLoop = False
        for maybe_type in myDataTypes:
            try:
                column.astype(maybe_type)
                print(f"Casting to {maybe_type} was successful")
                print('*' * 40)
                break
            except ValueError as error:
                # stopLoop = True
                print(f"Could not cast to {maybe_type}: {error}")
                print('*' * 40)
            # finally:
            #     print(" ")

    def find_non_default_missing_values(df, column_name):
        long_separator_amount = 80
        short_separator_amount = 40

        print_separator("*", long_separator_amount,
                        f"Finding non default missing values for column \"{column_name}\"")

        print(
            f"Column \"{column_name}\" has datatype: {df.dtypes[column_name]}")

        column = df[column_name]

        # Tactic A
        print_separator("-", short_separator_amount,
                        "A: Looking at unique values")
        look_at_unique_values(column)

        # Tactic B
        print_separator("-", short_separator_amount,
                        "B: Sorting and looking at the edges")
        look_at_edges(df, column_name)

        # Tactic C
        print_separator("-", short_separator_amount,
                        f'C: Casting to type: ["datetime64", "int64", "float64", "string", "bool"]')   # {maybe_type}")
        cast_to_type(column)

        # Tactic D
        print_separator("-", short_separator_amount, "D: Looking at frequency")
        print(column.value_counts(dropna=False))

        print("\n")

    find_non_default_missing_values(myDF, column_name)  # , maybe_type)


# Generate statistics for two columns at a time
def generateStatistics(myAllDF, twoAnalysisColumns):
    nameDF = get_obj_name(myAllDF)

    col1 = twoAnalysisColumns[0]
    col2 = twoAnalysisColumns[1]

    print(' ')
    print('*' * 40)
    print("            << Statistics >>            ")
    print(f"using dataset '{nameDF}' and columns= {twoAnalysisColumns}")
    print('*' * 40)

    df_ALL = myAllDF[[col1, col2]].dropna()
    print(f"ALL Dataframe: 'df_ALL'")
    print(df_ALL.head())
    print(df_ALL.columns)
    print('*' * 40)

    # Let us calculate the Correlation Coefficients:
    # ---------------------------------------------

    # 1) Pandas internal Correlation Coefficient:
    df_ALL.corr(method='pearson')
    # OUTPUT: example
    """
            GDP      Gini
    GDP   1.000000 -0.429966
    Gini -0.429966  1.000000
    """
    df_corr_1 = df_ALL.corr(method='pearson')
    print(type(df_corr_1))
    print(df_corr_1)
    print('*' * 40)

    corr1 = round(df_corr_1.iloc[1][col1], 6)
    print(type(corr1))
    print(f"{corr1= }")
    print('*' * 40)

    # 2) Correlation Coefficient using pingouin library:
    pingouin_corr(df_ALL[col1], df_ALL[col2])
    # OUTPUT:  example
    """
                n         r           CI95%         p-val       BF10  power
    pearson  1811 -0.429966  [-0.47, -0.39]  2.203046e-82  4.744e+78    1.0
    """

    df_corr_2 = pingouin_corr(df_ALL[col1], df_ALL[col2])
    print(type(df_corr_2))
    print(df_corr_2)
    print('*' * 40)

    corr2_r = round(df_corr_2.iloc[0]['r'], 6)
    corr2_p_val = round(df_corr_2.iloc[0]['p-val'], 6)
    print(f"{corr2_r= }")
    print(f"{corr2_p_val= }")

    print('*' * 40)
    print('*' * 40)


#
#
# **********************************************************************************
# **********************************************************************************
# **********************************************************************************
#                             << ANSWER ALL QUESTIONS >>
# **********************************************************************************
# **********************************************************************************
# **********************************************************************************
#
#
# *********************************************************************
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Q1: Biggest predictor of CO2 output per capita of a country
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Collect data for Q1
# -------------------
df_co2_gdp = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\co2-emissions-and-gdp.csv", sep=",")  # , encoding="ISO-8859-1")
df_co2_gdp.head(20)
df_co2_gdp.shape
# OUTPUT:   (7795, 6)
df_co2_gdp.columns
# OUTPUT:   ['Entity', 'Code', 'Year', 'Annual CO2 emissions', 'Annual consumption-based CO2 emissions', 'GDP per capita']

# df_co2_gdp.dropna(inplace=True)
# df_co2_gdp.shape
# OUTPUT:   (3470, 6)

df_co2_gdp.rename(columns={'Entity': 'country', 'Year': 'year', 'Annual CO2 emissions': 'annual_co2_emissions',
                  'GDP per capita': 'gdp_per_capita'}, inplace=True)
df_co2_gdp = df_co2_gdp[['country', 'year', 'annual_co2_emissions',
                         'gdp_per_capita']]


# Collect data for Q1
# -------------------
df_co2_kaya = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\kaya-identity-co2.csv", sep=",")  # , encoding="ISO-8859-1")
df_co2_kaya.head(20)
df_co2_kaya.shape
# OUTPUT:   (7795, 6)
df_co2_kaya.columns
# OUTPUT:   ['Entity', 'Code', 'Year', 'Annual CO2 emissions',
#    'Annual CO2 emissions per unit energy (kg per kilowatt-hour)',
#    'Annual CO2 emissions per GDP (kg per $PPP)',
#    'Primary energy consumption per GDP (kWh/$)', 'GDP per capita',
#    '417485-annotations', 'Population (historical estimates)']

# df_co2_kaya.dropna(inplace=True)
df_co2_kaya.shape
# OUTPUT:   (14925, 10)

df_co2_kaya.rename(columns={
                   'Entity': 'country', 'Year': 'year', 'Annual CO2 emissions': 'annual_co2_emissions', 'Annual CO2 emissions per unit energy (kg per kilowatt-hour)': 'co2_per_unit_energy',
                   'Annual CO2 emissions per GDP (kg per $PPP)': 'co2_per_gdp', 'Primary energy consumption per GDP (kWh/$)':
                   'energy_per_gdp', 'GDP per capita': 'gdp_per_capita', 'Population (historical estimates)': 'population'}, inplace=True)
df_co2_kaya = df_co2_kaya[['country', 'year', 'annual_co2_emissions',
                           'co2_per_unit_energy', 'co2_per_gdp', 'energy_per_gdp', 'gdp_per_capita', 'population']]

df_co2_kaya.head(20)
df_co2_kaya.shape
# OUTPUT:   (14925, 8)
df_co2_kaya.columns
# OUTPUT:   ['country', 'year', 'annual_co2_emissions', 'co2_per_unit_energy',
#    'co2_per_gdp', 'energy_per_gdp', 'gdp_per_capita', 'population']


'''
Q1 Analysis: 'CO2 output per capita' versus --> [Candidate Predictors]
Candidates I:  GDP per capita, diets, number of cars per capita, various energy source, mobility and other factors
Candidates II: Population, GDP per capita, 'energy_per_gdp', 'co2_per_unit_energy', 'co2_per_gdp'

For my analysis, I have chosen the second group (Candidates II).

'''

# Analyze dataframe 'df_co2_gdp'
analyzeDataframes(df_co2_gdp, df_co2_gdp.columns)

# To generate statistics, determine those columns which can be a good predictor of CO2 output per capita of a country
col_under_analysis = 'annual_co2_emissions'
col_candidate_predictors = ['population', 'gdp_per_capita',
                            'co2_per_gdp', 'energy_per_gdp', 'co2_per_unit_energy']

# Generate Statitics using dataset "df_co2_gdp"
generateStatistics(df_co2_gdp, ['gdp_per_capita', 'annual_co2_emissions'])
# OUTPUT:
"""
1) Correlation Coefficient using Pandas itself:
                      gdp_per_capita  annual_co2_emissions
gdp_per_capita              1.000000              0.984548
annual_co2_emissions        0.984548              1.000000

2) Correlation Coefficient using pingouin library:
            n         r         CI95%  p-val BF10  power
pearson  5690  0.984548  [0.98, 0.99]    0.0  nan    1.0
"""


# Generate Statitics for all candidate predictors using dataset "df_co2_kaya"
for col in col_candidate_predictors:
    generateStatistics(df_co2_kaya, [col, 'annual_co2_emissions'])
    print('*' * 40)
# OUTPUT:
"""
1) Correlation Coefficient using Pandas itself:
                      co2_per_unit_energy  annual_co2_emissions
co2_per_unit_energy              1.000000              0.051652
annual_co2_emissions             0.051652              1.000000

2) Correlation Coefficient using pingouin library:
            n         r         CI95%         p-val     BF10     power
pearson  9958  0.051652  [0.03, 0.07]           0.0  7470.51  0.999309
"""

# *********************************************************************

'''
And NOW implement an integration analysis of CO2 Emissions Predictors, combined together !

This implies applying of the following two "CO2 Emissions" formulas:

Method I = [ CO2 emissions = Population * (GDP /  Population) * (CO2 emissions per $) ]

Method II = [ CO2 Emissions = Population * (GDP /  Population) * (Energy / GDP) * (CO2 / Energy) ]


The corresponding columns for both methods are as follow:

Method I = [ 'annual_co2_emissions' vs. { 'population' * 'gdp_per_capita' * 'co2_per_gdp' } ]

Method II = [ 'annual_co2_emissions' vs. { 'population' * 'gdp_per_capita' * 'energy_per_gdp' * 'co2_per_unit_energy' } ]

'''

# Calculation of CO2 emissions driver using Method I ['co2_driver_method_1']
df_co2_kaya['co2_driver_method_1'] = df_co2_kaya['population'] * \
    df_co2_kaya['gdp_per_capita'] * df_co2_kaya['co2_per_gdp']


# Calculation of CO2 emissions driver using Method II ['co2_driver_method_2']
df_co2_kaya['co2_driver_method_2'] = df_co2_kaya['population'] * df_co2_kaya['gdp_per_capita'] * \
    df_co2_kaya['energy_per_gdp'] * df_co2_kaya['co2_per_unit_energy']

df_co2_kaya.head(20)
df_co2_kaya.shape
# OUTPUT:   (14925, 8) --> (14925, 10)
df_co2_kaya.columns
# OUTPUT:   ['country', 'year', 'annual_co2_emissions', 'co2_per_unit_energy', 'co2_per_gdp', 'energy_per_gdp', 'gdp_per_capita',
#            'population', 'co2_driver_method_1', 'co2_driver_method_2']


#
# Generate Statitics using dataset "df_co2_kaya", based on [ CO2 Method I ]
generateStatistics(
    df_co2_kaya, ['co2_driver_method_1', 'annual_co2_emissions'])
# OUTPUT:
"""
1) Correlation Coefficient using Pandas itself:
                      co2_driver_method_1  annual_co2_emissions
co2_driver_method_1              1.000000              0.999905
annual_co2_emissions             0.999905              1.000000

2) Correlation Coefficient using pingouin library:
            n         r       CI95%  p-val BF10  power
pearson  8533  0.999905  [1.0, 1.0]    0.0  inf    1.0
"""

#
# Generate Statitics using dataset "df_co2_kaya", based on [ CO2 Method II ]
generateStatistics(
    df_co2_kaya, ['co2_driver_method_2', 'annual_co2_emissions'])
# OUTPUT:
"""
1) Correlation Coefficient using Pandas itself:
                      co2_driver_method_2  annual_co2_emissions
co2_driver_method_2              1.000000              0.999878
annual_co2_emissions             0.999878              1.000000

2) Correlation Coefficient using pingouin library:
            n         r       CI95%  p-val BF10  power
pearson  7109  0.999878  [1.0, 1.0]    0.0  inf    1.0
"""


# *********************************************************************
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Q2: Biggest strides in decreasing CO2 output
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Collect data for Q2
# -------------------
df_co2_emissions = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\owid-co2-data.csv", sep=",")  # , encoding="ISO-8859-1")
df_co2_emissions.head(20)
df_co2_emissions.shape
# OUTPUT:   (26008, 60)
df_co2_emissions.columns
# OUTPUT:   ['country', 'year', 'iso_code', 'population', 'gdp', 'cement_co2',
#    'cement_co2_per_capita', 'co2', 'co2_growth_abs', 'co2_growth_prct',
#    'co2_per_capita', 'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2',
#    'coal_co2_per_capita', 'consumption_co2', 'consumption_co2_per_capita',
#    'consumption_co2_per_gdp', 'cumulative_cement_co2', 'cumulative_co2',
#    'cumulative_coal_co2', 'cumulative_flaring_co2', 'cumulative_gas_co2',
#    'cumulative_oil_co2', 'cumulative_other_co2', 'energy_per_capita',
#    'energy_per_gdp', 'flaring_co2', 'flaring_co2_per_capita', 'gas_co2',
#    'gas_co2_per_capita', 'ghg_excluding_lucf_per_capita', 'ghg_per_capita',
#    'methane', 'methane_per_capita', 'nitrous_oxide',
#    'nitrous_oxide_per_capita', 'oil_co2', 'oil_co2_per_capita',
#    'other_co2_per_capita', 'other_industry_co2',
#    'primary_energy_consumption', 'share_global_cement_co2',
#    'share_global_co2', 'share_global_coal_co2',
#    'share_global_cumulative_cement_co2', 'share_global_cumulative_co2',
#    'share_global_cumulative_coal_co2',
#    'share_global_cumulative_flaring_co2',
#    'share_global_cumulative_gas_co2', 'share_global_cumulative_oil_co2',
#    'share_global_cumulative_other_co2', 'share_global_flaring_co2',
#    'share_global_gas_co2', 'share_global_oil_co2',
#    'share_global_other_co2', 'total_ghg', 'total_ghg_excluding_lucf',
#    'trade_co2', 'trade_co2_share']


# IMPORTANT COLUMNS:
# ['country', 'year', 'iso_code', 'population', 'gdp', 'co2', 'co2_growth_abs', 'co2_growth_prct',
#    'co2_per_capita', 'co2_per_gdp', 'co2_per_unit_energy', 'consumption_co2_per_capita',
#    'consumption_co2_per_gdp', 'energy_per_capita', 'energy_per_gdp',
#    'primary_energy_consumption', 'share_global_co2', 'trade_co2', 'trade_co2_share']

df_co2_emissions2 = df_co2_emissions[['country', 'year', 'iso_code', 'population', 'gdp', 'co2', 'co2_growth_abs', 'co2_growth_prct',
                                      'co2_per_capita', 'co2_per_gdp', 'co2_per_unit_energy', 'consumption_co2_per_capita',
                                     'consumption_co2_per_gdp', 'energy_per_capita', 'energy_per_gdp',
                                      'primary_energy_consumption', 'share_global_co2', 'trade_co2', 'trade_co2_share']
                                     ]
df_co2_emissions2.head(20)
df_co2_emissions2.shape
# OUTPUT:   (26008, 60) --> (26008, 19)
df_co2_emissions2.columns

# df_co2_emissions2.dropna(inplace=True)
df_co2_emissions2.shape
# OUTPUT:   (26008, 60) --> (26008, 19) --> (3397, 19)
df_co2_emissions2.head(50)

df_co2_emissions2.country.value_counts()
len(df_co2_emissions2.country.unique())
# OUTPUT:  118
df_co2_emissions2.year.min()
# OUTPUT:   min=1990
df_co2_emissions2.year.max()
# OUTPUT:   max=2020 ?

df_co2_1990 = df_co2_emissions2.loc[df_co2_emissions2['year'] == 1990]
df_co2_1990
df_co2_1990.shape
# OUTPUT:   (3397, 19) --> (113, 19)

df_co2_emissions2.tail(20)

# Select Relevante columns
df_co2_strides = df_co2_emissions[['country', 'year', 'co2_per_capita']]
df_co2_strides

df_co2_strides.reset_index(drop=True, inplace=True)
df_co2_strides

df_co2_strides.shape
# OUTPUT:   (26008, 3)

# df_co2_strides.dropna(inplace=True)
# df_co2_strides = df_co2_strides.dropna()
# df_co2_strides.shape
# OUTPUT:   (26008, 3) --> (24032, 3)

# Due to the bad quality of data before year 1990, I have excluded that data:
df_co2_strides_2 = df_co2_strides.loc[df_co2_strides['year'] >= 1990]
df_co2_strides_2
df_co2_strides_2.shape
# OUTPUT:   (26008, 3) --> (7627, 3)        OUD: (7253, 3)

df_co2_strides_2 = df_co2_strides_2.dropna()
df_co2_strides_2
df_co2_strides_2.shape
# OUTPUT:   (7627, 3)  --> (7253, 3)

# Sort my basic clean dataset
df_co2_strides_2_sorted = df_co2_strides_2.sort_values(
    by=['country', 'year'], ascending=[True, True])
df_co2_strides_2_sorted


# Take one country as an example and then create a generic function for ALL COUNTRIES
# Test this code:
# df_co2_strides_2_sorted = df_co2_strides_2_sorted.loc[
#     df_co2_strides_2_sorted['country'] == "Afghanistan"]
# df_co2_strides_2_sorted['biggest_stride_decreasing_co2'] = df_co2_strides_2_sorted['co2_per_capita'] - \
#     df_co2_strides_2_sorted['co2_per_capita'].shift(periods=1, fill_value=0)
# df_co2_strides_2_sorted


# Make list of all countries
countryList = df_co2_strides_2_sorted.country.unique().tolist()
countryList.sort()
type(countryList)
len(countryList)
countryList

# Short lists of countries for Testing
countryList2 = []
countryList3 = ["Afghanistan"]
countryList4 = ["Afghanistan", "Turkmenistan", "United States"]


# Add a new column for the "Biggest strides in decreasing CO2 output"
df_co2_strides_2_sorted = df_co2_strides_2_sorted.assign(
    co2_stride=0)


# Process ALL Countries
def processCountries(myCtryList, myDF_AllCountries):

    # Capture an empty country list or dataframe
    if (len(myCtryList) == 0) or (len(myDF_AllCountries) == 0):
        print(
            f"MyError: [the dataframe '{get_obj_name(myDF_AllCountries)}' is EMPTY] OR [the countrylist '{get_obj_name(myCtryList)}' is EMPTY].")
        return myDF_AllCountries

    # Create an empty dataframe as a basic dataframe for collecting statistics of all countries
    myDF = pd.DataFrame(
        data=None, columns=myDF_AllCountries.columns, index=myDF_AllCountries.index)
    myDF.dropna(inplace=True)

    # Loop over countries
    for myCountry in myCtryList:

        myDF_oneCtry = myDF_AllCountries.loc[myDF_AllCountries['country'] == myCountry]

        # Calculate the CO2 strides for all countries and for all years [co2_stride = co2_stride_Yn - co2_stride_Yn-1]
        myDF_oneCtry['co2_stride'] = myDF_oneCtry['co2_per_capita'] - \
            myDF_oneCtry['co2_per_capita'].shift(periods=1, fill_value=0)

        minimum_year = myDF_oneCtry['year'].min()

        # As a BENCHMARK set value of year 1990 to zero
        myDF_oneCtry['co2_stride'] = np.where(
            myDF_oneCtry['year'].eq(minimum_year), 0, myDF_oneCtry['co2_stride'])

        # merge the results with the ALL countries dataframe:
        myDF = pd.concat([myDF, myDF_oneCtry])

    myDF.reset_index(drop=True, inplace=True)

    return myDF


# Calculate the CO2 strides for all countries for all years from year 1990
df_co2_strides_3 = processCountries(countryList, df_co2_strides_2_sorted)

df_co2_strides_3.reset_index(drop=True, inplace=True)

print(' ')
print('*' * 60)
print("[CO2 strides for all countries for all years]:")
print(df_co2_strides_3.shape)
print('*' * 60)
print(df_co2_strides_3.head(60))
print('*' * 60)


print(df_co2_strides_3.columns.to_list())
# OUTPUT:
# ['country', 'year', 'co2_per_capita', 'co2_stride']

# Implement data quality (DQ) analysis
# analysisColumnsList = df_co2_strides_3.columns.to_list()
# analyzeDataframes(df_co2_strides_3, analysisColumnsList)


def calculate_biggest_decrease_co2(myCtryList, myDF_AllCountries):

    # Capture an empty country list or dataframe
    if (len(myCtryList) == 0) or (len(myDF_AllCountries) == 0):
        print(
            f"MyError: [the dataframe '{get_obj_name(myDF_AllCountries)}' is EMPTY] OR [the countrylist '{get_obj_name(myCtryList)}' is EMPTY].")
        return myDF_AllCountries

    # *********************************************************
    # Create "biggest_stride_decreasing_co2" on country basis
    # *********************************************************

    # Create an empty dataframe as a basic dataframe for collecting statistics for all countries
    myDF = pd.DataFrame(
        data=None, columns=myDF_AllCountries.columns, index=myDF_AllCountries.index)
    myDF.dropna(inplace=True)

    # Loop over countries
    for myCountry in myCtryList:

        myDF_oneCtry = myDF_AllCountries.loc[myDF_AllCountries['country'] == myCountry]

        mySeries_biggest_stride = myDF_oneCtry.groupby(
            ['country'])['co2_stride'].min()
        type(mySeries_biggest_stride)
        mySeries_biggest_stride

        # Convert Series to DataFrame
        # mySeries_biggest_stride.unstack(level=1)
        df_biggest_stride = mySeries_biggest_stride.reset_index(drop=False)
        df_biggest_stride
        df_biggest_stride.rename(
            columns={"co2_stride": "biggest_stride_decreasing_co2"}, inplace=True)
        df_biggest_stride

        biggest_stride_decreasing_co2_value = df_biggest_stride[
            'biggest_stride_decreasing_co2'].values[0]
        biggest_stride_decreasing_co2_value

        df_co2_strides_merged = pd.merge(
            myDF_oneCtry, df_biggest_stride, on=['country'])
        # print(df_co2_strides_merged)

        myDF_oneCtry2 = df_co2_strides_merged.loc[(
            (df_co2_strides_merged['co2_stride'] == biggest_stride_decreasing_co2_value) & (df_co2_strides_merged['co2_stride'] != 0))]

        # Merge the results with the ALL countries dataframe:
        myDF = pd.concat([myDF, myDF_oneCtry2])
        # print(f"{myDF.shape= }")

    # ***********************************************************
    # Create "biggest_stride_decreasing_co2_year" on yearly basis
    # ***********************************************************
    myDF_AllCountries_year = deepcopy(myDF_AllCountries)

    myDF_AllCountries_year = myDF_AllCountries_year[[
        'year', 'country', 'co2_per_capita', 'co2_stride']]
    myDF_AllCountries_year = myDF_AllCountries_year.sort_values(
        by=['year', 'country'], ascending=[True, True])

    # Make list of all years
    yearList = myDF_AllCountries_year.year.unique().tolist()
    yearList.sort()
    # len(yearList)
    yearList

    myDF_year = pd.DataFrame(
        data=None, columns=myDF_AllCountries_year.columns, index=myDF_AllCountries_year.index)
    myDF_year.dropna(inplace=True)

    # Loop over countries
    for myYear in yearList:

        myDF_oneYear = myDF_AllCountries_year.loc[myDF_AllCountries_year['year'] == myYear]

        mySeries_biggest_stride_year = myDF_oneYear.groupby(
            ['year'])['co2_stride'].min()
        type(mySeries_biggest_stride_year)
        mySeries_biggest_stride_year

        # Convert Series to DataFrame
        # mySeries_biggest_stride_year.unstack(level=1)
        df_biggest_stride_year = mySeries_biggest_stride_year.reset_index(
            drop=False)
        df_biggest_stride_year
        df_biggest_stride_year.rename(
            columns={"co2_stride": "biggest_stride_decreasing_co2_year"}, inplace=True)
        df_biggest_stride_year

        biggest_stride_decreasing_co2_year_value = df_biggest_stride_year[
            'biggest_stride_decreasing_co2_year'].values[0]
        biggest_stride_decreasing_co2_year_value

        df_co2_strides_merged_year = pd.merge(
            myDF_oneYear, df_biggest_stride_year, on=['year'])
        # print(df_co2_strides_merged_year)

        myDF_oneYear2 = df_co2_strides_merged_year.loc[(
            (df_co2_strides_merged_year['co2_stride'] == biggest_stride_decreasing_co2_year_value) & (df_co2_strides_merged_year['co2_stride'] != 0))]

        # Merge the results with the ALL countries dataframe:
        myDF_year = pd.concat([myDF_year, myDF_oneYear2])

    return (myDF, myDF_year)


# Generate the values of "Biggest strides in decreasing CO2 output" & "biggest_stride_decreasing_co2_year"
df_co2_strides_4,  df_co2_strides_4_year = calculate_biggest_decrease_co2(
    countryList, df_co2_strides_3)


# ****************************************************
# Prepare Countries Result
df_co2_strides_4.reset_index(drop=True, inplace=True)

report_countries_decreasing_co2 = df_co2_strides_4.drop(
    columns=['co2_stride'])
# print(report_countries_decreasing_co2.head(500))

report_countries_decreasing_co2 = report_countries_decreasing_co2.sort_values(
    by=['biggest_stride_decreasing_co2'], ascending=True)

report_countries_decreasing_co2.reset_index(drop=True, inplace=True)

print(' ')
print('*' * 60)
print("[Biggest stride in decreasing CO2 output of each country]:")
print(report_countries_decreasing_co2.shape)
print('*' * 60)
print(report_countries_decreasing_co2.head(500))
print('*' * 60)
print(' ')

biggest_10_strides = report_countries_decreasing_co2.iloc[:10]

print('*' * 60)
print("[Biggest 10 strides in decreasing CO2 output internationally (on country basis)]:")
print(biggest_10_strides.shape)
print('*' * 60)
print(biggest_10_strides)
print('*' * 60)
# print('*' * 60)


# ****************************************************
# Prepare Years Result
df_co2_strides_4_year = df_co2_strides_4_year.sort_values(
    by=['year', 'country'], ascending=[True, True])

report_years_decreasing_co2 = df_co2_strides_4_year.drop(
    columns=['co2_stride'])
# print(report_years_decreasing_co2.head(500))

report_years_decreasing_co2 = report_years_decreasing_co2.sort_values(
    by=['biggest_stride_decreasing_co2_year'], ascending=True)

report_years_decreasing_co2.reset_index(drop=True, inplace=True)

print(' ')
print('*' * 60)
print("[Biggest stride in decreasing CO2 output of each year]:")
print(report_years_decreasing_co2.shape)
print('*' * 60)
print(report_years_decreasing_co2.head(500))
print('*' * 60)
print(' ')

biggest_10_strides = report_years_decreasing_co2.iloc[:10]

print('*' * 60)
print("[Biggest 10 strides in decreasing CO2 output internationally (on yearly basis)]:")
print(biggest_10_strides.shape)
print('*' * 60)
print(biggest_10_strides)
print('*' * 60)


# *************************************************
# Prepare Countries Result starting from year 2015
report_countries_decreasing_co2_recent = report_countries_decreasing_co2.loc[
    report_countries_decreasing_co2['year'] >= 2015]
report_countries_decreasing_co2_recent = report_countries_decreasing_co2_recent.sort_values(
    by=['biggest_stride_decreasing_co2'], ascending=True)

report_countries_decreasing_co2_recent.reset_index(drop=True, inplace=True)

biggest_10_strides = report_countries_decreasing_co2_recent.iloc[:10]

print('*' * 60)
print("[Recent (from 2015), Biggest 10 strides in decreasing CO2 output internationally (on country basis)]:")
print(biggest_10_strides.shape)
print('*' * 60)
print(biggest_10_strides)
print('*' * 60)
# print('*' * 60)


# *************************************************
# Prepare Years Result starting from year 2015
report_years_decreasing_co2_recent = report_years_decreasing_co2.loc[
    report_years_decreasing_co2['year'] >= 2015]
report_years_decreasing_co2_recent = report_years_decreasing_co2_recent.sort_values(
    by=['biggest_stride_decreasing_co2_year'], ascending=True)

report_years_decreasing_co2_recent.reset_index(drop=True, inplace=True)
# print("report_years_decreasing_co2.head(500)=")
# print(report_years_decreasing_co2.head(500))

biggest_10_strides = report_years_decreasing_co2_recent.iloc[:10]

print('*' * 60)
print("[Recent (from 2015), Biggest 10 strides in decreasing CO2 output internationally (on yearly basis)]:")
print(biggest_10_strides.shape)
print('*' * 60)
print(biggest_10_strides)
print('*' * 60)


# *********************************************************************
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Q3: Best future price for non-fossil fuel energy
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Collect data for Q3
# -------------------
df_energy_levelized_cost = pd.read_csv(
    r".\ALL Assignments\Final Assgn_007_Mod7 (CO2 emissions)\1_F_CO2_Mn\data\ready\levelized-cost-of-energy.csv")
df_energy_levelized_cost.head(20)
df_energy_levelized_cost.shape
# OUTPUT:   (413, 10)
df_energy_levelized_cost.columns
# OUTPUT: ['Entity', 'Code', 'Year', 'CSP LCOE (2019 USD/kWh)',
#        'Hydro LCOE (2019 USD/kWh)', 'Solar LCOE (2019 USD/kWh)',
#        'Onshore wind LCOE (2019 USD/kWh)', 'Bioenergy LCOE (2019 USD/kWh)',
#        'Geothermal LCOE (2019 USD/kWh)', 'Offshore wind LCOE (2019 USD/kWh)']


# Analize Columns:
# analyzeDataframes(df_energy_levelized_cost,
#                   df_energy_levelized_cost.columns)
# sys.exit('<<< !! MY FORCING STOP RUNNING !! >>>')


# Analize missing values
for col in df_energy_levelized_cost.columns:
    find_missing_values(df_energy_levelized_cost, col)
    print('*' * 40)


# Rename columns
df_energy_levelized_cost.columns = ['country', 'code', 'year', 'csp_price', 'hydro_price',
                                    'solar_price', 'onshore_wind_price', 'bioenergy_price', 'geothermal_price', 'offshore_wind_price']
df_energy_levelized_cost.head(20)
df_energy_levelized_cost.shape
# OUTPUT:   (413, 10)
df_energy_levelized_cost.columns

# Keep relevant columns
df_energy_levelized_cost = df_energy_levelized_cost[['country', 'year', 'csp_price', 'hydro_price',
                                                     'solar_price', 'onshore_wind_price', 'bioenergy_price', 'geothermal_price', 'offshore_wind_price']]

df_energy_levelized_cost = df_energy_levelized_cost.dropna(subset=[
    'country', 'year'])
df_energy_levelized_cost.shape

df_energy_levelized_cost = df_energy_levelized_cost.dropna(
    subset=['csp_price', 'hydro_price', 'solar_price', 'onshore_wind_price', 'bioenergy_price', 'geothermal_price', 'offshore_wind_price'], how='all')
df_energy_levelized_cost.shape

df_energy_levelized_cost.isnull().mean() * 100


df_energy_levelized_cost_World = df_energy_levelized_cost.loc[
    df_energy_levelized_cost['country'] == 'World']

df_energy_levelized_cost_World = df_energy_levelized_cost_World.sort_values(by=[
    'year'], ascending=True)

df_energy_levelized_cost_World.reset_index(drop=True, inplace=True)

df_energy_levelized_cost_World.head(20)
df_energy_levelized_cost_World.shape
# OUTPUT:   (128, 3)
df_energy_levelized_cost_World.columns
# OUTPUT: ['country', 'year', 'csp_price', 'hydro_price', 'solar_price', 'onshore_wind_price', 'bioenergy_price',
#           'geothermal_price', 'offshore_wind_price']

analysis_columns_all = df_energy_levelized_cost_World.columns
# OUTPUT: ['country', 'year', 'csp_price', 'hydro_price', 'solar_price', 'onshore_wind_price', 'bioenergy_price',
#           'geothermal_price', 'offshore_wind_price']

non_fossil_energy_price_columns = ['csp_price', 'hydro_price', 'solar_price',
                                   'onshore_wind_price', 'bioenergy_price', 'geothermal_price', 'offshore_wind_price']

# In order to predict prices and simultaneously test the model, I began from year 2019 to overlap the future!
yearList_pred = [2019, 2020, 2021, 2022, 2025, 2028, 2030, 2033, 2035]

# Make a deep copy, delete all null-values and reset the index
df_energy_levelized_cost_World_NoNull = df_energy_levelized_cost_World[:].copy(
)
df_energy_levelized_cost_World_NoNull.dropna(inplace=True)
df_energy_levelized_cost_World_NoNull.reset_index(drop=True, inplace=True)

df_energy_levelized_cost_World_NoNull
df_energy_levelized_cost_World_NoNull.shape
df_energy_levelized_cost_World_NoNull.isnull().value_counts()
df_energy_levelized_cost_World_NoNull

df_pred_results = pd.DataFrame(
    data=None, columns=df_energy_levelized_cost_World.columns, index=df_energy_levelized_cost_World.index)

df_pred_results.drop(columns=['csp_price', 'hydro_price', 'solar_price',
                              'onshore_wind_price', 'bioenergy_price', 'geothermal_price', 'offshore_wind_price'], inplace=True)
df_pred_results.dropna(inplace=True)
df_pred_results.reset_index(drop=True, inplace=True)
df_pred_results
df_pred_results.shape

df_pred_results['year'] = yearList_pred
df_pred_results['country'] = 'World'

df_pred_results
df_pred_results.shape


# Predict future prices using a linear regression model
def predict_future_prices(myDF, myCol):

    myPredCol = 'pred_' + myCol

    myX = np.asarray(
        df_energy_levelized_cost_World_NoNull['year'].values.astype('datetime64[D]'))
    type(myX)
    myX
    myX.ndim

    # Convert to 2D array
    myX = myX.reshape(-1, 1)
    myX.ndim
    myX.shape

    # put your y-values in here and cast the data type
    myY = np.asarray(
        df_energy_levelized_cost_World_NoNull[myCol].values.astype('float64'))
    myY = myY.reshape(-1, 1)
    myY
    type(myY)
    myY.shape

    # make scatter plots for all non-fossil energy columns in our World dataset (df_energy_levelized_cost_World_NoNull)
    # Nevertheless, I execute code for running the scatterplot not for every item. Instead, I let the code to run with one item as a demo!
    if myCol == 'onshore_wind_price':
        df_energy_levelized_cost_World_NoNull.plot.scatter(
            x='year', y=myCol, title=f'Scatterplot of years versus {myCol}')
        plt.show()

    model = LinearRegression()
    model.fit(myX, myY)

    # In order to predict prices and simultaneously test the model, I began from year 2019 to overlap the future!
    X_year_pred = np.asarray(
        [2019, 2020, 2021, 2022, 2025, 2028, 2030, 2033, 2035]).astype('float').reshape(-1, 1)
    X_year_pred

    y_pred = model.predict(X_year_pred)
    # Test case: y_pred = model.predict([[2025]])
    print('*' * 40)
    print(f"Predicted prices of '{myCol}', \n{y_pred= }")
    print(f"{type(y_pred)= }")

    df_pred_results[myPredCol] = y_pred

    # make scatter plots for PREDICTED non-fossil energy columns in our predicted dataset (df_pred_results) --> BEL
    if myPredCol == 'pred_onshore_wind_price':
        df_pred_results.plot.scatter(
            x='year', y=myPredCol, title=f'Scatterplot of PREDICTED years versus {myPredCol}')
        plt.show()

    return df_pred_results


# Predict future prices for non-fossil columns
for col in non_fossil_energy_price_columns:
    df_pred_results = predict_future_prices(df_pred_results, col)

# Predicted Prices are for the period [2019 - 2035].
# Be aware that I made an 'interesting' price comparison with some 'earlier' years just as a test !!
df_pred_results
print(df_pred_results)

# Determine "Best future price" for non-fossil energy (after carefully examining the predicted results dataframe)
df_Best_3_future_price_non_fossil = df_pred_results.loc[df_pred_results['year'] == 2035]

# Melt dataframe reform data in our favor
df_Best_3_future_price_non_fossil = df_Best_3_future_price_non_fossil.melt(
    id_vars=['country', 'year'])

# Rename columns to "pred_non_fossil_energy_tech" & "future_price_in_2035"
df_Best_3_future_price_non_fossil.rename(
    columns={"variable": "pred_non_fossil_energy_tech", 'value': 'pred_future_price_in_2035'}, inplace=True)

# Sort to determine which non-fossil fuel energy technology will have the best price in the future
df_Best_3_future_price_non_fossil = df_Best_3_future_price_non_fossil.sort_values(
    by=['pred_future_price_in_2035'], ascending=True)

df_Best_3_future_price_non_fossil.reset_index(drop=True, inplace=True)

print('*' * 70)
print("[Top 3 Best future price for non-fossil fuel energy]:")
print(df_Best_3_future_price_non_fossil)


"""
Conclusion
----------
The top 3 non-fossil fuel energy technologies which will have the best price in the future are the following:
1) Solar energy as a whole.
2) CSP (Concentrated Solar Power) plants.
3) Onshore wind energy.

"""

# [ FOR FURTHER EXPLANATION PLEASE SEE MY FINAL REPORT FOR THIS ASSIGNMENT ] !!


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#                           THE END OF MY PY CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
