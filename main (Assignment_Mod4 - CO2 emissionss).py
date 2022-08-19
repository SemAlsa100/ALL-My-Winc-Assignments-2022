
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tables = pd.read_html(
    "https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions")
global_co2_emissions = tables[1]

global_co2_emissions = global_co2_emissions.loc[3:]

global_co2_emissions.columns = ["country", "fco2e_mt_1990", "fco2e_mt_2005", "fco2e_mt_2017", "fco2e_pow_2017", "fco2e_pch_2017vs1990",
                                "fco2e_per_land_km2_2017", "fco2e_per_capita_2017", "fco2e_total_inc_lucf_2018", "fco2e_total_exc_lucf_2018"]


# ******************************************************************************
# Graph 1: CO2 of the bigger countries
# ******************************************************************************
#
gco2e_5_max = global_co2_emissions.sort_values(
    by=['fco2e_mt_2017'], ascending=False)

gco2e_5_max = gco2e_5_max.iloc[:5]

graph1_data = gco2e_5_max[
    [
        "country",
        "fco2e_mt_1990",
        "fco2e_mt_2005",
        "fco2e_mt_2017"
    ]
]

# Rename the columns
graph1_data.columns = ["country", "1990", "2005", "2017"]
print(graph1_data)

plt.rcdefaults()
fig, ax = plt.subplots()

years = graph1_data.columns[1:]

for index, row in graph1_data.iterrows():
    plt.plot(years, row[1:], label=row[0])

plt.title('CO2 of the bigger countries')
plt.xlabel('Year')
plt.ylabel('Fossil CO2 emissions in Mt CO2')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

plt.show()


# ******************************************************************************
# Graph 2: worst and best changers
# ******************************************************************************
#
global_co2_emissions['re_Ch_1990'] = 100
global_co2_emissions['re_Ch_2005'] = (
    global_co2_emissions['fco2e_mt_2005'] / global_co2_emissions['fco2e_mt_1990']) * 100
global_co2_emissions['re_Ch_2017'] = (
    global_co2_emissions['fco2e_mt_2017'] / global_co2_emissions['fco2e_mt_1990']) * 100

gco2e_top_3 = global_co2_emissions.sort_values(
    by=['re_Ch_2017'], ascending=False)
gco2e_top_3 = gco2e_top_3.iloc[:3]
print(gco2e_top_3)

gco2e_bottom_3 = global_co2_emissions.sort_values(
    by=['re_Ch_2017'], ascending=True)
gco2e_bottom_3 = gco2e_bottom_3.iloc[:3]
print(gco2e_top_3)


# ******************************************************************************
# Graph 2A: worst and best changers
# ******************************************************************************
#
graph_2a_data = pd.concat([gco2e_top_3, gco2e_bottom_3])

graph_2a_data = graph_2a_data[
    [
        "country",
        "re_Ch_1990",
        "re_Ch_2005",
        "re_Ch_2017"
    ]
]
# Rename the columns
graph_2a_data.columns = ["country", "1990", "2005", "2017"]

plt.rcdefaults()
fig, ax = plt.subplots()

years = graph_2a_data.columns[1:]

for index, row in graph_2a_data.iterrows():
    plt.plot(years, row[1:], label=row[0])

plt.title('Worst and best changers')
plt.xlabel('Year')
plt.ylabel('Relative amount of CO2 emitted')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

plt.show()

# ******************************************************************************
# Graph 2B: worst and best changers for countries that had at least five Mt of CO2 emissions in 1990
# ******************************************************************************
#
print(global_co2_emissions.info())
global_co2_emissions.drop(
    ['re_Ch_1990', 're_Ch_2005', 're_Ch_2017'], inplace=True, axis=1)
print(global_co2_emissions.info())

global_co2_emissions = global_co2_emissions.loc[global_co2_emissions.fco2e_mt_1990 >= 5]
print(f"Graph 2B: {global_co2_emissions= }")

global_co2_emissions['re_Ch_1990'] = 100
global_co2_emissions['re_Ch_2005'] = (
    global_co2_emissions['fco2e_mt_2005'] / global_co2_emissions['fco2e_mt_1990']) * 100
global_co2_emissions['re_Ch_2017'] = (
    global_co2_emissions['fco2e_mt_2017'] / global_co2_emissions['fco2e_mt_1990']) * 100


gco2e_top_3 = global_co2_emissions.sort_values(
    by=['re_Ch_2017'], ascending=False)
gco2e_top_3 = gco2e_top_3.iloc[:3]

gco2e_bottom_3 = global_co2_emissions.sort_values(
    by=['re_Ch_2017'], ascending=True)
gco2e_bottom_3 = gco2e_bottom_3.iloc[:3]

graph_2b_data = pd.concat([gco2e_top_3, gco2e_bottom_3])

graph_2b_data = graph_2b_data[
    [
        "country",
        "re_Ch_1990",
        "re_Ch_2005",
        "re_Ch_2017"
    ]
]
# Rename the columns
graph_2b_data.columns = ["country", "1990", "2005", "2017"]

plt.rcdefaults()
fig, ax = plt.subplots()

years = graph_2b_data.columns[1:]

for index, row in graph_2b_data.iterrows():
    plt.plot(years, row[1:], label=row[0])

plt.title(
    'Worst and best changers for countries with at least 5 Mt of CO2 emissions in 1990')
plt.xlabel('Year')
plt.ylabel('Relative amount of CO2 emitted')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

plt.show()
