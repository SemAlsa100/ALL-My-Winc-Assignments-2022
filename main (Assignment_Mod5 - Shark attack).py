
import re
from itertools import count
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


# Rename columns:
df2 = df2.rename(columns={"Type": "Type Incident",
                 "Age": "Age Victim", "Species ": "Species Shark", "Fatal (Y/N)": "Fatal (Y_N)"})
df2.columns
# OUTPUT:
# Index(['Case Number', 'Type Incident', 'Activity', 'Age Victim', 'Fatal (Y_N)', 'Species Shark'],

df2.shape
df2.head(20)
df2.tail(20)


# Start Analyzing data
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


# Analyze by running analyzeDF:
analyzeDF(df2)

# Clean up
df_noNA = df2.dropna(how='all')
df_noNA.dropna(thresh=4, inplace=True)

df_noNA = df_noNA.sort_values('Case Number')
df_noNA.reset_index(inplace=True)
df_noNA = df_noNA.sort_values('index')
df_noNA

# Examine:
df_noNA[df_noNA['Activity'].astype(str).str.contains("<|.")]
df_noNA[df_noNA['Activity'].astype(str).str.contains("surf")]
df_noNA.loc[df_noNA['Activity'].astype(str).str.contains("surf|Surf")]
df_noNA[df_noNA['Activity'].astype(str).str.contains("<")]
df_noNA.loc[df_noNA['Activity'].astype(str).str.contains("swim|Swim")]


# Analyze further
# analyzeDF(df_noNA)
print('*' * 40)
print(df_noNA.info())
print('*' * 40)
df_noNA.shape
df_noNA.head(20)
df_noNA.tail(20)

# Analyze further
# data = data.loc[data["cases"] != 0]
print('*' * 40)
print(df_noNA.loc[df_noNA['Case Number'] == 0])
print('*' * 40)
print(len(df_noNA.loc[df_noNA['Case Number'] == 0]))
print('*' * 40)


# **************************************************************************
# **************************************************************************
# For each column:
# ---------------
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
# ***************************************************************************
# ***************************************************************************
#

# Make a neat separator for beter readability
def print_separator(sep, num, msg):
    print("\n")
    print(sep * num)
    print(f"{msg}")
    print(sep * num)


# Tactic A: looking at all the unique values
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


# Tactic B: sorting and looking at the edges
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


# Tactic C: casting to a type
def cast_to_type(column, maybe_type):
    try:
        column.astype(maybe_type)
        print(f"Casting to {maybe_type} was successful")
    except ValueError as error:
        print(f"Could not cast to {maybe_type}: {error}")


# Implement ALL 4 tactics (A, B, C and D)
def find_non_default_missing_values(df, column_name, maybe_type):
    long_separator_amount = 80
    short_separator_amount = 40

    print_separator("*", long_separator_amount,
                    f"Finding non default missing values for column \"{column_name}\"")

    print(f"Column \"{column_name}\" has datatype: {df.dtypes[column_name]}")

    column = df[column_name]

    # Tactic A: looking at all the unique values
    print_separator("-", short_separator_amount, "A: Looking at unique values")
    look_at_unique_values(column)

    # Tactic B: sorting and looking at the edges
    print_separator("-", short_separator_amount,
                    "B: Sorting and looking at the edges")
    look_at_edges(df, column_name)

    # Tactic C: casting to a type
    print_separator("-", short_separator_amount,
                    f"C: Casting to type: {maybe_type}")
    cast_to_type(column, maybe_type)

    # Tactic D: looking at the frequency
    print_separator("-", short_separator_amount, "D: Looking at frequency")
    print(column.value_counts(dropna=False))

    print("\n")


# Replace bad or missing data
def replace_value(df, column_name, missing_old, missing_new):
    # ⚠️ Mutates df
    df[column_name] = df[column_name].replace({missing_old: missing_new})


# ***************************************************************************************************************
# Columns for analysis: ['Case Number', 'Type Incident', 'Activity', 'Age Victim', 'Species Shark', Fatal (Y_N)]
# ***************************************************************************************************************


# ✅ 'Case Number': no missing values
find_non_default_missing_values(df_noNA, 'Case Number', "string")


# ✅ 'Type Incident'
find_non_default_missing_values(df_noNA, 'Type Incident', "string")

# My Assumption: It is legitimate to delete Shark attacks with 'Type Incident' = 'Invalid', because they are officially invalid !
df_noNA.shape
df_noNA = df_noNA[df_noNA['Type Incident'] != 'Invalid']
df_noNA.shape


# ✅ 'Activity'
find_non_default_missing_values(df_noNA, 'Activity', 'string')

print(df_noNA['Activity'].value_counts(dropna=False))

# My Assumption: I have found many activities, which contain the name of one of the top 4 activities (Surfing, Swimming, Fishing and Spearfishing).
# Thus, I merged those activities with their corresponding activity.
# This led to much higher numbers of activities and there weights too.


# replace a string value of a column with a certain value in one row at a time (in a DataFrame)
def get_actv_1(row):
    a = str(row["Activity"])
    # b = row["Column B"]
    if "surf" in a.lower():
        return 'Surfing'
    return a


df_noNA["Activity"] = df_noNA.apply(get_actv_1, axis=1)


def get_actv_2(row):
    a = str(row["Activity"])
    # b = row["Column B"]
    if "swim" in a.lower():
        return 'Swimming'
    return a


df_noNA["Activity"] = df_noNA.apply(get_actv_2, axis=1)


def get_actv_3(row):
    a = str(row["Activity"])
    # b = row["Column B"]
    if "spearfishing" in a.lower():
        return 'Spearfishing'
    return a


df_noNA["Activity"] = df_noNA.apply(get_actv_3, axis=1)


def get_actv_4(row):
    a = str(row["Activity"])
    # b = row["Column B"]
    if "spearfishing" not in a.lower():
        if "fishing" in a.lower():
            return 'Fishing'
    return a


df_noNA["Activity"] = df_noNA.apply(get_actv_4, axis=1)


# My Assumption: Activities with np.nan or 'strange' values (such as '.') must be changed to a uniform vale of 'Unknown'.
df_noNA.loc[df_noNA['Activity'] == '.']
replace_value(df_noNA, 'Activity', np.nan, 'Unknown')
replace_value(df_noNA, 'Activity', '.', 'Unknown')
df_noNA.loc[df_noNA['Activity'] == 'Unknown']

print(df_noNA['Activity'].value_counts(dropna=False).head(50))
print(len(df_noNA['Activity']))

find_non_default_missing_values(df_noNA, 'Activity', 'string')


# ✅ 'Age Victim'
# My 1st Assumption: I have converted NaN values in the Age column to 18 year old. This means that a missing age is equal to an adult of 18 years old.
# My 2nd Assumption: If an age is a string, this should be converted to numeric. If the numeric value is an empty string or 0-zero then this will get 18 years old.

df_noNA.loc[df_noNA['Age Victim'] == np.nan]
replace_value(df_noNA, 'Age Victim', np.nan, 18)
replace_value(df_noNA, 'Age Victim', "", 18)

df_noNA['Age Victim'] = pd.to_numeric(
    df_noNA['Age Victim'], errors='coerce').fillna(18).astype(np.int64)

find_non_default_missing_values(df_noNA, 'Age Victim', 'int64')


# ✅ 'Species Shark':
# My Assumption: values with np.nan or 'strange' values (such as '.', empty string, spaces...etc.) must be changed to a uniform vale of 'Unknown'.
replace_value(df_noNA, 'Species Shark', np.nan, 'Unknown')
replace_value(df_noNA, 'Species Shark', " ", 'Unknown')
replace_value(df_noNA, 'Species Shark', "", 'Unknown')

find_non_default_missing_values(df_noNA, 'Species Shark', 'string')


# My Assumption: I have found many values in the column 'Species Shark', which contain the name of one of the top 3 species (White shark, Tiger shark and Bull shark).
# Thus, I filtered and merged those values with their corresponding specie.
# Also I have changed the word 'Sharks' to 'Shark' to make better merge.
# This led to much higher numbers of shark species and there weights too.

# replace a string value of a column with a certain value in one row at a time (in a DataFrame)


# Determine the index (if any) of a certain value in a list. A value can :
def get_index(myList, item, n_occur):
    c = count()
    try:
        idx = next(i for i, j in enumerate(myList)
                   if j == item and next(c) == n_occur-1)
        return idx
    except:
        return -1


# Collect information and join them for a beter merge:
def scan_and_change_specie(row):
    joined_word = ''
    idx = -1
    idx2 = -1

    a = str(row["Species Shark"])

    a = re.sub(r'([^a-zA-Z ]+?)', "", a).strip()

    # for x in "1234567890":
    #     a = a.replace(str(x), '')

    a = a.lower()
    a = a.replace('sharks', 'shark')

    myList = a.split()

    idx = get_index(myList, 'shark', 1)
    if idx not in [-1]:
        if idx != 0:
            joined_word = myList[idx-1] + ' ' + 'shark'
        elif idx == 0:
            idx2 = get_index(myList, 'shark', 2)
            if idx2 not in [-1]:
                joined_word = joined_word + ' ' + \
                    myList[idx2-1] + ' ' + 'shark'

        return joined_word.title()
    return a.title()


df_noNA["Species Shark"] = df_noNA.apply(scan_and_change_specie, axis=1)

replace_value(df_noNA, 'Species Shark', np.nan, 'Unknown')
replace_value(df_noNA, 'Species Shark', " ", 'Unknown')
replace_value(df_noNA, 'Species Shark', "", 'Unknown')


find_non_default_missing_values(df_noNA, 'Species Shark', 'string')

# frequency
print(df_noNA["Species Shark"].value_counts(dropna=False).head(60))


# ✅ 'Fatal (Y_N)'
# My Assumption: values with np.nan or 'strange' values (such as numeric, '.', empty string, spaces...etc.) must be changed to a uniform vale of 'Unknown'.
# My Assumption: values with unnecessary spaces should be trimed on both sides.
# My Assumption: values with small letters have to be capitalized.
# My Assumption: values with wrong contents (not in ['Y', 'N', 'UNKNOWN']) have to be adjusted to 'UNKNOWN'.

# np.nan or 'strange' values (such as numeric, '.', empty string, spaces...etc.) must be changed to a uniform vale of 'Unknown'.
replace_value(df_noNA, 'Fatal (Y_N)', np.nan, 'UNKNOWN')
replace_value(df_noNA, 'Fatal (Y_N)', ' N', 'N')
replace_value(df_noNA, 'Fatal (Y_N)', '2017', 'UNKNOWN')
replace_value(df_noNA, 'Fatal (Y_N)', 'y', 'Y')
replace_value(df_noNA, 'Fatal (Y_N)', 'M', 'UNKNOWN')

find_non_default_missing_values(df_noNA, 'Fatal (Y_N)', 'string')


# *******************************************************
# Answer ALL Questions:
# *******************************************************

# Question 1/4:
print('*' * 40)
print("Question 1/4:")

df_killer_shark = df_noNA.loc[(df_noNA['Fatal (Y_N)'] == 'Y')]
print(df_killer_shark['Species Shark'].value_counts(dropna=False))
mySeries = df_killer_shark['Species Shark'].value_counts(dropna=False)
myDF = pd.DataFrame({'Species Shark': mySeries.index,
                    'sharkCountFatal': mySeries.values})

myDF['sharkPerc'] = round(
    (myDF['sharkCountFatal'] / myDF['sharkCountFatal'].sum()) * 100, 2)
print(myDF)

# Count Total Number Deadly Shark Attacks
print(myDF['sharkCountFatal'].sum())
# Count Total Number ALL Shark Attacks
print(len(df_noNA['Species Shark']))
myDF.head(60)

# *************
# Question 2/4:
print('*' * 40)
print("Question 2/4:")

# My Assumption: A child is a human with age less than 18 year. So I have searched for children with fatal shark attack.
df_child_fatal = df_noNA.loc[(df_noNA['Age Victim'] < 18) &
                             (df_noNA['Fatal (Y_N)'] == 'Y')]
df_all_fatal = df_noNA.loc[df_noNA['Fatal (Y_N)'] == 'Y']
# count children with fatal incident and count all cases with fatal incident
print(len(df_child_fatal['Age Victim']))
print(len(df_all_fatal['Age Victim']))
# count all cases with shark attack
# print(len(df_noNA['Age Victim']))

# calculate the percentage of children with fatal incident and all cases with fatal incident
print(round((len(df_child_fatal['Age Victim']) /
      len(df_all_fatal['Age Victim'])) * 100, 2))

# *************
# Question 3/4:
print('*' * 40)
print("Question 3/4:")
df_provc = df_noNA.loc[(df_noNA['Fatal (Y_N)'] == 'Y') &
                       (df_noNA['Type Incident'] == 'Provoked')]
df_all_fatal = df_noNA.loc[df_noNA['Fatal (Y_N)'] == 'Y']

print(len(df_provc))
print(len(df_all_fatal))
print(round((len(df_provc)/len(df_all_fatal)) * 100, 2))

# *************
# Question 4/4:
print('*' * 40)
print("Question 4/4:")
print(df_noNA['Activity'].value_counts(dropna=False))
print(type(df_noNA['Activity'].value_counts(dropna=False)))

mySeries = df_noNA['Activity'].value_counts(dropna=False)
myDF = pd.DataFrame({'activity': mySeries.index, 'actvCount': mySeries.values})

print(myDF['actvCount'].sum())
myDF['actvPerc'] = round(
    (myDF['actvCount'] / myDF['actvCount'].sum()) * 100, 2)
print(myDF)
