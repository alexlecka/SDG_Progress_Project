"""

This is the main script for the assignment. It follows the instructions given in the provided pdf.

How is the progress towards the Sustainable Development Goald (SDGs) developing over time, and what can we expect in 2030?
Data available here: https://unstats.un.org/sdgs/indicators/database/

Chosen SDG: 6.1.1
Chosen countries: Lithuania and Estonia

"""

from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication, QLabel
from Assignment_B_functions import evaluate_model, validate_model
from Assignment_B_functions import plot_task6a, plot_task6b
from Assignment_B_functions import logistic, calibration
from tabulate import tabulate
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
import time
import sys

#%% 

start = time.time()
print('Starting to measure time.')
print()

# Task 1: Import data with pandas

# After checking the data outside of Python (in Excel), I found that missing values are encoded with NaN.
# I identified the relevant data sheet to import and the columns of the data that will be relevant to
# the analysis.

# define path and import data from the downloaded .xlsx file (define sheet and relevant columns)
file = 'data/SDG_data_6.1.1.xlsx'
columns = ['SeriesDescription', 'GeoAreaCode', 'GeoAreaName', 'TimePeriod', 'Value', 'Location', 'Units']
df = pd.read_excel(file, na_values = 'NaN', sheet_name = 'Goal6', usecols = columns)

# for data exploration
# print(df.head())

# The exploration shows that there is an indicator for 3 different locations - urban, rural, and
# combined (all area).

#%% 

# Task 2: Filter data to one country at a time

# create a dictionary with a key being the country - for this assignment, I chose to look at the data 
# collected for location labeled as ALLAREA - this means that some countries won't have sufficient data 
# to fit a model (note for future)

countries = np.unique(df['GeoAreaName'])
country_codes = np.unique(df['GeoAreaCode'])
analysis_data = {}

loc_for_analysis = 'ALLAREA' # change to 'URBAN' or 'RURAL' if desired
for country in countries:
    analysis_data[country] = df[(df['Location'] == loc_for_analysis) & (df['GeoAreaName'] == country)]
    
#%%

# Task 3: Make sanity checks and print them into the console

# calculate the amount of non-missing values per country
present_values = []
for country in countries:
    values = np.asarray(analysis_data[country]['Value'], dtype = 'float')
    nonmissing = len(values) - sum(np.isnan(values))
    present_values.append(nonmissing)

# give an overview about the non-missing values
print('Overview of the non-missing values in data:')
print('On average, {}% of a population of a country is using safely managed water services.'.format(round(np.mean(df['Value']), 3)))
print('The minimum of non-missing values for the indicator is {}%.'.format(np.min(df['Value'])))
print('The maximum of non_missing values for the indicator is {}%.'.format(np.max(df['Value'])))
print()

#%%

# Task 4: Main analysis - fitting

# calibrate the logistic model, save the model coefficients in list fits
fits = []

# for the countries with no data, the model cannot be calibrated - this will be recorded 
# in a list of booleans is_calibrated
is_calibrated = []

# x_values and y_values will hold the observed values (years and SDG indicator values)
x_values = []
y_values = []

for country in analysis_data.keys(): # loop over all countries
    years = np.asarray(analysis_data[country]['TimePeriod'], dtype = 'int')
    values = np.asarray(analysis_data[country]['Value'], dtype = 'float')
    present_idx = np.invert(np.isnan(values)) # filter the observed values to not contain NAN
    if sum(present_idx) < 2: # less than two observations - model cannot be fitted
        fits.append(np.array(np.full(4, np.nan)))
        is_calibrated.append(0)
        x_values.append(np.nan)
        y_values.append(np.nan)
    else:
        x = years[present_idx]
        y = values[present_idx]
        x_values.append(x)
        y_values.append(y)
        fit = calibration(x, y) # fit model
        fits.append(np.asarray(fit)) # save fit coefficients
        is_calibrated.append(1)

str_countries = ''
for idx in np.where(np.asarray(is_calibrated) == 0)[0]:
    str_countries += countries[idx] + ', '
print('Could not fit model for {}.'.format(str_countries[:-2]))
print()

# start building a results data frame
SDG_indicator = '6.1.1'
SDG_description = 'Proportion of population using safely managed drinking water services, by urban/rural (%)'
analysis = {'SDG_indicator':np.full(len(countries), SDG_indicator), 'SDG_description':np.full(len(countries), SDG_description),
            'country':countries, 'country_code': country_codes, 'nonmissing':present_values, 'x':x_values, 'y':y_values, 'fit':fits,
            'calibrated':is_calibrated}
analysis_res = pd.DataFrame(data = analysis)

#%%

# Task 4: Main analysis - calculate the growth rate and the expected indicator value in 2030

growth_rate = []
exp_value = []

for i, row in analysis_res.iterrows():
    if row['calibrated'] == 0: # if the model is not calibrated, neither can be calculated
        growth_rate.append(np.nan)
        exp_value.append(np.nan)
    else:
        growth_rate.append(row['fit'][-1])
        exp_value.append(logistic(2030, row['fit'][0], row['fit'][1], row['fit'][2], row['fit'][3]))
analysis_res['growth_rate'] = growth_rate
analysis_res['expected_indicator'] = exp_value

#%%

# Task 4: Main analysis - evaluation

# evaluate the logistis fits (code in the script with functions)
analysis_res = evaluate_model(analysis_res)

#%%

# Task 8: Create radio button for interaction

class Window(QMainWindow):
    """
    Pop-up window
    
    This class creates an interaction window for cross-validation in task 8.
    
    Args:
        y_observed (array_like): The observed values.
        y_simulated (array_like): The simulated values.
        
    Returns:
        float: The R squared.
    """
    global n_splits
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cross-validation")
        self.setGeometry(100, 100, 400, 400)
        self.setStyleSheet('background-color : lightGray')
        self.UiComponents()
        self.show()
        self.n_splits = 5

    def UiComponents(self):  
        
        font = QFont('Times', 15)
        font.setBold(True)
        
        head = QLabel("Choose the number of subsets\nto be generated for validation", self)
        head.setGeometry(50, 150, 300, 100)
        head.setFont(font)
        head.setAlignment(Qt.AlignCenter)
        
        font = QFont('Times', 13)
        
		# creating the first button
        self.button1 = QPushButton("5-fold cross validation", self)
        self.button1.setGeometry(100, 50, 200, 100)
        self.button1.setStyleSheet('background-color : darkGray')
        self.button1.setFont(font)
        self.button1.clicked.connect(self.click_action1)
        self.button1.clicked.connect(self.closeit)
        
        # creating the second button
        self.button2 = QPushButton("10-fold cross validation", self)
        self.button2.setGeometry(100, 250, 200, 100)
        self.button2.setStyleSheet('background-color : darkGray')
        self.button2.setFont(font)
        self.button2.clicked.connect(self.click_action2)
        self.button2.clicked.connect(self.closeit)        

    def click_action1(self):
        # selection of 5-fold cross validation
        self.n_splits = 5
  
        # disabling the button
        self.button2.setDisabled(True)
          
    def click_action2(self):
        # selection of 10-fold cross validation
        self.n_splits = 10        
  
        # disabling the button
        self.button1.setDisabled(True)
        
    def closeit(self):
        self.close()

# Task 4: Main analysis - validation

interaction = False # change to False for no interaction
k = 5

# validate models with same 3 criteria and k-fold cross-validation (k = 5 or k = 10) 
if interaction:
    # launch the pyqt5 app
    app = QApplication(sys.argv)
      
    # create instance
    window = Window() 
      
    # start the app
    app.exec()
    
    print('You chose {}-fold validation.'.format(window.n_splits))
    k = window.n_splits
else:
    print('No interaction, 5-fold cross-validation as a default.')
    
print()
analysis_res = validate_model(analysis_res, k)

#%% Intermezzo: evaluation and validation table

r2_eval_mean = round(analysis_res['r2_evaluation'].mean(), 3)
nrmse_eval_mean = round(analysis_res['nrmse_evaluation'].mean(), 3)
pbias_eval_mean = round(analysis_res['pbias_evaluation'].mean(), 3)
r2_valid_mean = round(analysis_res['r2_validation({}-fold)'.format(k)].mean(), 3)
nrmse_valid_mean = round(analysis_res['nrmse_validation({}-fold)'.format(k)].mean(), 3)
pbias_valid_mean = round(analysis_res['pbias_validation({}-fold)'.format(k)].mean(), 3)

table = [['', 'R2', 'NRMSE', 'percentage bias'], 
         ['evaluation', r2_eval_mean, pbias_eval_mean, pbias_eval_mean],
         ['validation ({}-fold)'.format(k), r2_valid_mean, nrmse_valid_mean, pbias_valid_mean]]

print('The mean of measures of evaluation and validation as calculated above:')
print(tabulate(table, headers = 'firstrow'))
print()

#%%

# Task 5: Making plots - contrasting countries - will save in working directory

# choose countries to be plotted
countries_to_plot = ['Lithuania', 'Estonia']

plot_task6a(countries_to_plot, analysis_res)

#%%

# Task 5: Making plots - top30 countries - will save in working directory

# read in the population data from UN
path = 'data/Population_data.xlsx'
df = pd.read_excel(path, header = [1], skiprows = 15, usecols = ['Country code', 'Region, subregion, country or area *', 
                                                                 '2020'])
df = df.rename(columns = {'Region, subregion, country or area *' : 'Country'})
codes_calibrated = analysis_res[analysis_res['calibrated'] == 1]['country_code']
        
# filter to countries that have calibrated models, sort the dataframe by descending population
df_calibrated = df.loc[df['Country code'].isin(codes_calibrated)]
df_calibrated = df_calibrated.sort_values('2020', ascending = False)
df_calibrated = df_calibrated.reset_index(drop = True)
df_calibrated['2020'] = df_calibrated['2020']*1000 

# define the 30 most populous countries out of the previously obtained dataframe
top_30_countries = df_calibrated['Country'][:30]
top_30_subset = analysis_res.loc[analysis_res['country'].isin(top_30_countries)]

plot_task6b(top_30_countries, top_30_subset, plot_title = 'plot_top30.png')

#%%

# Task 6: Export the main results to a text file

# export selected SDG indicator and its description, countries, growth rates and indicator values,
# expected indicator values in 2030, performance values for the evaluation, performance values
# for the validation
save_path = 'main_results.csv'
print('Results are being saved to {}.'.format(save_path))
analysis_res.to_csv(save_path, columns = ['SDG_indicator', 'SDG_description', 'country',
                   'growth_rate', 'x', 'y', 'expected_indicator', 'r2_evaluation',
                   'nrmse_evaluation', 'pbias_evaluation', 'r2_validation({}-fold)'.format(k), 
                   'nrmse_validation({}-fold)'.format(k), 'pbias_validation({}-fold)'.format(k)])

print()
end = time.time()
print('The elapsed time is {} seconds.'.format(round(end - start, 2)))