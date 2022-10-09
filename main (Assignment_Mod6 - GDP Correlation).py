
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pingouin import corr as pingouin_corr
from copy import deepcopy


# get dataset "Gini Coefficient" for all countries
df_Gini = pd.read_csv("./ALL Assignments/Assignment_006_Mod6 (GDP)/1_GDP_Mn/data/economic-inequality-gini-index.csv",
                      sep=",", encoding="ISO-8859-1")
df_Gini.head(20)
df_Gini.columns
df_Gini.shape
# OUTPUT:   ['Entity', 'Code', 'Year', 'Gini coefficient']
# OUTPUT:   (2126, 4)

# get dataset "Historical GDP" for all countries
df_GDP = pd.read_csv("./ALL Assignments/Assignment_006_Mod6 (GDP)/1_GDP_Mn/data/gdp-per-capita-maddison-2020.csv",
                     sep=",", encoding="ISO-8859-1")
df_GDP.head(20)
df_GDP.columns
df_GDP.shape
# OUTPUT:   ['Entity', 'Code', 'Year', 'GDP per capita', '417485-annotations']
# OUTPUT:   (19876, 5)
df_GDP['417485-annotations'].value_counts()
df_GDP.Entity.value_counts()

# Choose relevant Gini columns and rename them
df_Gini = df_Gini[['Entity', 'Year', 'Gini coefficient']]
df_Gini.rename(columns={"Entity": "Country",
                        'Gini coefficient': 'Gini'}, inplace=True)   # BEL
df_Gini

# Choose relevant GDP columns and rename them
df_GDP = df_GDP[['Entity', 'Year', 'GDP per capita']]
df_GDP.rename(columns={"Entity": "Country",
              'GDP per capita': 'GDP'}, inplace=True)   # BEL
df_GDP

# delete null rows
df_Gini.dropna(inplace=True)
df_GDP.dropna(inplace=True)

# Merge the two datasets into one central dataset!
df_GDP_Gini = pd.merge(df_GDP, df_Gini, on=["Country", "Year"])
df_GDP_Gini
df_GDP_Gini.columns
# OUTPUT:   ['Country', 'Year', 'GDP', 'Gini']
df_GDP_Gini.Country.value_counts()

# Make list of all countries
countryList = df_GDP_Gini.Country.unique().tolist()
type(countryList)
len(countryList)
countryList


# get a dataframe name
def get_df_name(df):
    name = [x for x in globals() if globals()[x] is df][0]
    return name


# analze statistics datafrmaes
def analizeStatistics(myAnalysisDF):
    print('*' * 40)
    print(f"{get_df_name(myAnalysisDF):}")
    print(f"{myAnalysisDF.shape= }")
    print("myAnalysisDF.head(50)=")
    print(f"{myAnalysisDF.head(50)}")
    print("myAnalysisDF.tail(50)=")
    print(f"{myAnalysisDF.tail(50)}")
    print("myAnalysisDF.value_counts()=")
    print(f"{myAnalysisDF.value_counts()}")

    try:
        print("myAnalysisDF.Country.value_counts().sum()=")
        print(f"{myAnalysisDF.Country.value_counts().sum()}")
    except AttributeError as error:
        print(f"The GDP Error: {error}.")
    # finally:
        print('*' * 40)

    try:
        print("myAnalysisDF.Country.unique().shape=")
        print(f"{myAnalysisDF.Country.unique().shape}")
    except AttributeError as error:
        print(f"The GDP Error: {error}.")
    # finally:
        print('*' * 40)

    print("myAnalysisDF.Corr_Strength.value_counts()")
    print(f"{myAnalysisDF.Corr_Strength.value_counts()}")


# *********************************************************************
# *********************************************************************
# Countries Statistics for Individual Countries AND together [METHOD 1]
# *********************************************************************
# *********************************************************************
#
print('*' * 40)
print('*' * 40)
print("Statistics Individual Countries AND together [METHOD 1]")
print('*' * 40)

countryList2 = ["Honduras", "Turkmenistan", "United States"]
countryList3 = ["Honduras"]

# Create an empty list for countries which have too small number of observations (<= 4 obs.)
# for correlation to be calculated!
notAddedCountries = []


def processCountries(myCtryList, myGDP_GiniDF):

    # Create an empty dataframe as a basic dataframe for collecting statistics for all countries
    myDF = pd.DataFrame(
        data=None, columns=myGDP_GiniDF.columns, index=myGDP_GiniDF.index)
    myDF.dropna(inplace=True)

    # Loop over countries
    for myCountry in myCtryList:
        if len(myCtryList) == 1 and myCountry == "ALL Countries":
            df_oneCtry = myGDP_GiniDF
            print('*' * 40)
            print('*' * 40)
        else:
            df_oneCtry = myGDP_GiniDF.loc[myGDP_GiniDF['Country'] == myCountry]

        df_oneCtry_2 = df_oneCtry[['GDP', 'Gini']].dropna()

        print(
            f"Country: <{myCountry}> has {len(df_oneCtry_2)} observation(s).")

        if len(df_oneCtry_2) > 4:
            # get and add Corr1:
            df_crr_1 = df_oneCtry_2.corr(method='pearson')
            crr1 = round(df_crr_1.iloc[1]['GDP'], 6)
            df_oneCtry = df_oneCtry.assign(Corr1=crr1)

            # get and add Corr2_r:
            df_crr_2 = pingouin_corr(df_oneCtry_2['GDP'], df_oneCtry_2['Gini'])
            crr2_r = round(df_crr_2.iloc[0]['r'], 6)
            df_oneCtry = df_oneCtry.assign(Corr2_r=crr2_r)

            # get and add Corr2_p_value:
            crr2_p_val = round(df_crr_2.iloc[0]['p-val'], 6)
            df_oneCtry = df_oneCtry.assign(Corr2_p_val=crr2_p_val)

            # add Correlation Type:
            if crr2_r > 0.1:
                Crr_Sgn = "Positive"
            elif -0.1 <= crr2_r <= 0.1:
                Crr_Sgn = "Neutral"
            elif crr2_r < -0.1:
                Crr_Sgn = "Negative"
            df_oneCtry = df_oneCtry.assign(Corr_Sign=Crr_Sgn)

            # add Correlation Strength:
            if -0.10 < crr2_r < 0.10:
                Crr_Strngth = 'Negligible'
            elif (0.10 <= crr2_r < 0.20) or (-0.20 < crr2_r <= -0.10):
                Crr_Strngth = 'Weak'
            elif (0.20 <= crr2_r < 0.40) or (-0.40 < crr2_r <= -0.20):
                Crr_Strngth = 'Moderate'
            elif (0.40 <= crr2_r < 0.60) or (-0.60 < crr2_r <= -0.40):
                Crr_Strngth = 'Relatively Strong'
            elif (0.60 <= crr2_r < 0.80) or (-0.80 < crr2_r <= -0.60):
                Crr_Strngth = 'Strong'
            elif (0.80 <= crr2_r <= 1.00) or (-1.00 <= crr2_r <= -0.80):
                Crr_Strngth = 'Very Strong'
            df_oneCtry = df_oneCtry.assign(Corr_Strength=Crr_Strngth)

            # add Significance:
            if crr2_p_val < 0.05:
                Sgnfc = 'Significant'
            elif crr2_p_val >= 0.05:
                Sgnfc = 'Not Significant'
            df_oneCtry = df_oneCtry.assign(Significance=Sgnfc)

            # merge the results with the ALL countries dataframe:
            myDF = pd.concat([myDF, df_oneCtry])
            # df_GDP_Gini_ALL = pd.concat([df_GDP_Gini_ALL, df_oneCtry])

            if len(myCtryList) == 1 and myCountry == "ALL Countries":

                # select first row of the dataframe
                myDF = myDF[:1]    # As a dataframe!
                myDF['Country'] = "ALL Countries"
                myDF['Year'] = ""
                myDF['GDP'] = ""
                myDF['Gini'] = ""

                print(f"{type(myDF)= }")

            print(f"Country: < {myCountry} > has been added.")
        else:
            notAddedCountries.append(myCountry)
    return myDF


print('*' * 40)

# Generate statistics [METHOD 1]:
# ******************************

# process Countries one by one and then generate one huge merged set of statistics
df_GDP_Gini_individually = processCountries(countryList, df_GDP_Gini)


# Analize statistics outcome
analizeStatistics(df_GDP_Gini_individually)

# process ALL Countries together (["ALL Countries"]) to generate one set of statistics
df_GDP_Gini_ALL_together = processCountries(["ALL Countries"], df_GDP_Gini)


# Analize statistics outcome
analizeStatistics(df_GDP_Gini_ALL_together)

print('*' * 40)

# Display countries which have too small number of observations (<= 4 obs.) for correlation to be calculated!
print(f"{len(notAddedCountries)= }")
print(f"{notAddedCountries= }")


# [ PLEASE SEE MY EXPLANATION UNDER "FACTS & CONCLUSION" FURTHER IN THIS PY-MODULE ] !!


# *********************************************************************
# *********************************************************************
#      Overall Statistics for ALL Countries Mutually [METHOD 2]
# *********************************************************************
# *********************************************************************

def overallCountriesStatistics(myAllDF):
    print('*' * 40)
    print('*' * 40)
    print("ALL Countries Mutual Statistics [METHOD 2]")
    print('*' * 40)

    df_ALL = myAllDF[['GDP', 'Gini']].dropna()
    print(f"ALL Countries dataframe: 'df_ALL'")
    print(df_ALL.head())
    print(df_ALL.columns)

    # Let us calculate the Correlation Coefficient:
    # ---------------------------------------------

    # 1) Pandas internal Correlation Coefficient:
    df_ALL.corr(method='pearson')
    # OUTPUT:
    """
            GDP      Gini
    GDP   1.000000 -0.429966
    Gini -0.429966  1.000000
    """
    df_corr_1 = df_ALL.corr(method='pearson')
    print(type(df_corr_1))
    print(df_corr_1)

    corr1 = round(df_corr_1.iloc[1]['GDP'], 6)
    print(type(corr1))
    print(corr1)

    # 2) Correlation Coefficient using pingouin library:
    pingouin_corr(df_ALL['GDP'], df_ALL['Gini'])
    # OUTPUT:
    """
                n         r           CI95%         p-val       BF10  power
    pearson  1811 -0.429966  [-0.47, -0.39]  2.203046e-82  4.744e+78    1.0
    """

    df_corr_2 = pingouin_corr(df_ALL['GDP'], df_ALL['Gini'])
    print(type(df_corr_2))
    print(df_corr_2)

    corr2_r = round(df_corr_2.iloc[0]['r'], 6)
    corr2_p_val = round(df_corr_2.iloc[0]['p-val'], 6)
    print(corr2_r, corr2_p_val)

    print('*' * 40)
    print('*' * 40)


# Generate statistics [METHOD 2]:
# ******************************
overallCountriesStatistics(df_GDP_Gini)


# [ PLEASE SEE FURTHER MY EXPLANATION UNDER "FACTS & CONCLUSION" ] !!


# ***********************************************************************************************************#
#                                         ((  Facts & Conclusion ))                                          #
# ***********************************************************************************************************#

"""
# If the p-value is low (generally less than 0.05), then your correlation is statistically significant, and you can use the calculated Pearson coefficient.
# If the p-value is not low (generally higher than 0.05), then your correlation is not statistically significant (it might have happened just by chance) and you should not rely upon your Pearson coefficient.

# Rules of thumb (vuistregels) of "Pearson Correlation":
    0.00 < 0.10 - Negligible
    0.10 < 0.20 - Weak
    0.20 < 0.40 - Moderate
    0.40 < 0.60 - Relatively strong
    0.60 < 0.80 - Strong
    0.80 < 1.00 - Very strong

Conclusion
----------
Question: Is there a relation between a country's Gross Domestic Product(GDP) and its income inequality?
Answer:
[1]: For ALL countries mutually:
    a)   both corr-methods give us an corr-r-value of <-0.429966>.
    b)  Moreover the given p-val is <2.203046e-82>. This value is obviously less than 0.05. 
    c)  This means that there is a Negative Correlation "Relatively Strong" relation (correlation) between a country's Gross Domestic Product(GDP) and its income inequality.
        This correlation is "statistically significant".  Therefore it is reliable.
[2]: For individual countries:
     The correlation is generally 'Strong' to 'Very Strong'.
     This is a table which shows the numbers of the TOTAL 'Correlation Strength' in individual countries:  

        'Correlation Strength' results:
        ...............................
        Very Strong          419
        Strong               416
        Relatively Strong    374
        Moderate             265
        Weak                 115
        Negligible           102
"""

# *********************************************************#
#                         (( End ))                        #
# *********************************************************#
