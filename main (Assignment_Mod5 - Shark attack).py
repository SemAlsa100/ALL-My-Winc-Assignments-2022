import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Warning: Use your own dataset path hier:
df = pd.read_csv('./ALL Assignments/Assignment_005_Mod5 (Shark attack)/1_Shark attack Mn/data/Shark attacks.csv',
                 sep=",", encoding="ISO-8859-1")
df.head()
df.tail()

df.columns
# OUTPUT:
# Index(['Case Number', 'Date', 'Year', 'Type', 'Country', 'Area', 'Location',
#        'Activity', 'Name', 'Sex ', 'Age', 'Injury', 'Fatal (Y/N)', 'Time',
#        'Species ', 'Investigator or Source', 'pdf', 'href formula', 'href',
#        'Case Number.1', 'Case Number.2', 'original order', 'Unnamed: 22',
#        'Unnamed: 23'],
#       dtype='object')

# Remove unnamed & repeated columns:
df2 = df.drop(columns=['Country', 'Area', 'Location', 'Name', 'Sex ', 'Time', 'Investigator or Source',
              'pdf', 'href formula', 'Case Number.1', 'Case Number.2', 'original order', 'Unnamed: 22', 'Unnamed: 23', 'Injury', 'Date', 'Year', 'href'])

df2

# Rename columns:
df3 = df2.rename(columns={"Type": "Type Incident",
                 "Age": "Age Victim", "Species ": "Species Shark", "Fatal (Y/N)": "Fatal (Y_N)"})
df3.columns
# OUTPUT:
# Index(['Case Number', 'Type Incident', 'Activity', 'Age Victim', 'Fatal (Y_N)',
#        'Species Shark'],

df3.shape
df3.head(20)
df3.tail(20)


def analyzeDF(df):
    print('*' * 40)
    print(df.info())
    print('*' * 40)
    print(df.describe())
    print('*' * 40)
    print(df.isnull().sum())        # Column wise
    print('*' * 40)
    print(df.isnull().mean())       # Column wise
    print('*' * 40)
    print(df.isnull().sum(axis=1))  # Row wise
    print('*' * 40)


# Test by running analyzeDF:
analyzeDF(df3)

df4 = df3.dropna(how='all')
df4.dropna(thresh=4, inplace=True)

df4 = df4.sort_values('Case Number')
df4.reset_index(inplace=True)
df4 = df4.sort_values('index')
df4

# Analyze further
print('*' * 40)
print(df4.info())
print('*' * 40)
df4.shape
df4.head(20)
df4.tail(20)

# Analyze further
print('*' * 40)
print(df4.loc[df4['Case Number'] == 0])
print('*' * 40)
print(len(df4.loc[df4['Case Number'] == 0]))
print('*' * 40)


# Analyze further

def print_separator(sep, num, msg):
    print("\n")
    print(sep * num)
    print(f"{msg}")
    print(sep * num)


def look_at_unique_values(column):
    unique_values_cutoff = 160
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
        print(list(values)[: num_items_to_slice])
        print(list(values)[-num_items_to_slice:])

    column = df[column_name]
    unique_values = column.unique()
    try:
        sorted = np.sort(unique_values)
        print("Unique values sorted, head and tail:")
        show_head_and_tail(sorted)
    except TypeError as error:
        print(f"Could not sort values: {error}")
        print("..so let's try filtering NULL values and then sorting")
        non_null_uniques = df.loc[~df[column_name].isnull(
        ), column_name].unique()
        sorted = np.sort(non_null_uniques)
        show_head_and_tail(sorted)


def cast_to_type(column, maybe_type):
    try:
        column.astype(maybe_type)
        print(f"Casting to {maybe_type} was successful")
    except ValueError as error:
        print(f"Could not cast to {maybe_type}: {error}")


def find_non_default_missing_values(df, column_name, maybe_type):
    long_separator_amount = 80
    short_separator_amount = 40

    print_separator("*", long_separator_amount,
                    f"Finding non default missing values for column \"{column_name}\"")

    print(f"Column \"{column_name}\" has datatype: {df.dtypes[column_name]}")

    column = df[column_name]

    # A
    print_separator("-", short_separator_amount, "A: Looking at unique values")
    look_at_unique_values(column)

    # B
    print_separator("-", short_separator_amount,
                    "B: Sorting and looking at the edges")
    look_at_edges(df, column_name)

    # C
    print_separator("-", short_separator_amount,
                    f"C: Casting to type: {maybe_type}")
    cast_to_type(column, maybe_type)

    # D
    print_separator("-", short_separator_amount, "D: Looking at frequency")
    print(column.value_counts(dropna=False))

    print("\n")


# Replace bad or missing data

def replace_value(df, column_name, missing_old, missing_new):
    # ⚠️ Mutates df
    df[column_name] = df[column_name].replace({missing_old: missing_new})

# Columns for analysis: ['Case Number', 'Type Incident', 'Activity', 'Age Victim', 'Species Shark', Fatal (Y_N)]


# ✅ 'Case Number'
find_non_default_missing_values(df4, 'Case Number', "string")

# ✅ 'Type Incident'
df4.shape
df4 = df4[df4['Type Incident'] != 'Invalid']
df4.shape

# ✅ 'Activity'
df4.loc[df4['Activity'] == '.']
replace_value(df4, 'Activity', np.nan, 'Unknown')
replace_value(df4, 'Activity', '.', 'Unknown')
df4.loc[df4['Activity'] == 'Unknown']

find_non_default_missing_values(df4, 'Activity', 'string')

# ✅ 'Age Victim'
df4.loc[df4['Age Victim'] == np.nan]  # == ""]
replace_value(df4, 'Age Victim', np.nan, 18)
replace_value(df4, 'Age Victim', "", 18)

df4['Age Victim'] = pd.to_numeric(
    df4['Age Victim'], errors='coerce').fillna(18).astype(np.int64)

find_non_default_missing_values(df4, 'Age Victim', 'int64')


# ✅ 'Species Shark':
replace_value(df4, 'Species Shark', np.nan, 'Unknown')
replace_value(df4, 'Species Shark', " ", 'Unknown')

find_non_default_missing_values(df4, 'Species Shark', 'string')

# ✅ 'Fatal (Y_N)'
replace_value(df4, 'Fatal (Y_N)', np.nan, 'UNKNOWN')
replace_value(df4, 'Fatal (Y_N)', ' N', 'N')
replace_value(df4, 'Fatal (Y_N)', '2017', 'UNKNOWN')
replace_value(df4, 'Fatal (Y_N)', 'y', 'Y')
replace_value(df4, 'Fatal (Y_N)', 'M', 'UNKNOWN')

find_non_default_missing_values(df4, 'Fatal (Y_N)', 'string')


# *******************************************************
# Answer ALL Questions:
# *******************************************************
# Question 1/4:
print('*' * 40)
print("Question 1/4:")
print(df4['Species Shark'].value_counts(dropna=False))
mySeries = df4['Species Shark'].value_counts(dropna=False)
myDF = pd.DataFrame({'Species Shark': mySeries.index,
                    'sharkCount': mySeries.values})

myDF['sharkPerc'] = round(
    (myDF['sharkCount'] / myDF['sharkCount'].sum()) * 100, 2)
print(myDF['sharkCount'].sum())
myDF

# *************
# Question 2/4:
print('*' * 40)
print("Question 2/4:")

df_child = df4.loc[df4['Age Victim'] < 18]
print(len(df4['Age Victim']))
print(len(df_child['Age Victim']))
print(round((len(df_child['Age Victim']) / len(df4['Age Victim'])) * 100, 2))

# *************
# Question 3/4:
print('*' * 40)
print("Question 3/4:")
df_provc = df4.loc[(df4['Fatal (Y_N)'] == 'Y') &
                   (df4['Type Incident'] == 'Provoked')]
df_unprovc = df4.loc[(df4['Fatal (Y_N)'] == 'Y') & (
    df4['Type Incident'] == 'Unprovoked')]
len_all_prv_unprv = len(df_provc) + len(df_unprovc)

print(len(df_provc))
print(len(df_unprovc))
print(len_all_prv_unprv)
print(round((len(df_provc)/len_all_prv_unprv) * 100, 2))

# *************
# Question 4/4:
print('*' * 40)
print("Question 4/4:")
print(df4['Activity'].value_counts(dropna=False))
print(type(df4['Activity'].value_counts(dropna=False)))
# OUTPUT:
# Surfing                                                 930
# Swimming                                                778
# Fishing                                                 409
# Spearfishing                                            307

mySeries = df4['Activity'].value_counts(dropna=False)
myDF = pd.DataFrame({'activity': mySeries.index, 'actvCount': mySeries.values})

print(myDF['actvCount'].sum())
myDF['actvPerc'] = round(
    (myDF['actvCount'] / myDF['actvCount'].sum()) * 100, 2)
myDF
