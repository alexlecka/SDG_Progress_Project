from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_r2(y_observed, y_simulated):
    """
    R squared
    
    This function calculates the R squared.
    
    Args:
        y_observed (array_like): The observed values.
        y_simulated (array_like): The simulated values.
        
    Returns:
        float: The R squared.
    """
    # below, if a condition is true, the array is 'nearly constant', which also includes being constant - 
    # this deals with an edge case for the perasonr function as well as the instruction from the assignment
    cond1 = np.linalg.norm(y_observed - np.mean(y_observed)) < 1e-13 * abs(np.mean(y_observed))
    cond2 = np.linalg.norm(y_simulated - np.mean(y_simulated)) < 1e-13 * abs(np.mean(y_simulated))
    if cond1 and cond2:
        return 1.
    elif cond1 or cond2:
        return 0.
    else:
        return (pearsonr(y_observed, y_simulated)[0])**2

def calculate_nrmse(y_observed, y_simulated):
    """
    Normalized mean squared error (NRMSE)
    
    This function calculates the NRMSE.
    
    Args:
        y_observed (array_like): The observed values.
        y_simulated (array_like): The simulated values.
        
    Returns:
        float: The NRMSE.
    """
    if np.mean(y_observed) == 0. and np.std(y_observed) == 0.:
        return np.nan
    else:
        rmse = np.sqrt(np.sum(np.power((y_simulated - y_observed), 2))/len(y_simulated))
        return rmse/np.mean(y_observed)

def calculate_pbias(y_observed, y_simulated):
    """
    Percentage bias (pbias)
    
    This function calculates the pbias.
    
    Args:
        y_observed (array_like): The observed values.
        y_simulated (array_like): The simulated values.
        
    Returns:
        float: The pbias.
    """
    if np.mean(y_observed) == 0. and np.std(y_observed) == 0.:
        return np.nan
    else:
        return 100*(np.sum(y_simulated - y_observed)/np.sum(y_observed))
    
def evaluate_model(analysis_res):
    """
    Model evaluation
    
    This function evaluates models in a provided data frame using R squared,\
        NRMSE and pbias.
    
    Args:
        analysis_res (dataframe): Data frame containing the fitted models\
            ('calibrated' = 1) in column 'fit', and the independent and\
                dependent variable in 'x' and 'y', respectively.
        
    Returns:
        dataframe: Data frame with 3 new columns for evaluation.
    """
    r2_evaluation = []
    nrmse_evaluation = []
    pbias_evaluation = []
    
    for i, row in analysis_res.iterrows():
        if row['calibrated'] == 0: # if the model was not calibrated, evaluation cannot be done
            r2_evaluation.append(np.nan)
            nrmse_evaluation.append(np.nan)
            pbias_evaluation.append(np.nan)
        else:
            y_true = row['y']
            y_pred = logistic(row['x'], *row['fit'])
    
    #         evaluate model with R2
            r2 = calculate_r2(y_true, y_pred)
            r2_evaluation.append(r2)
    
    #         evaluate model with NRMSE
            nrmse = calculate_nrmse(y_true, y_pred)
            nrmse_evaluation.append(nrmse)
    
    #         evaluate model with percent bias (pbias)
            pbias = calculate_pbias(y_true, y_pred)
            pbias_evaluation.append(pbias)
    
    analysis_res['r2_evaluation'] = r2_evaluation
    analysis_res['nrmse_evaluation'] = nrmse_evaluation
    analysis_res['pbias_evaluation'] = pbias_evaluation

    return analysis_res
    
def validate_model(analysis_res, k):
    """
    k-fold model cross-validation
    
    This function validates fitted models in a provided dataframe using k-fold\
        cross-validation with R squared, NRMSE and pbias.
    
    Args:
        analysis_res (dataframe): Data frame containing the fitted models\
            ('calibrated' = 1) in column 'fit', and the independent and\
                dependent variable in 'x' and 'y', respectively.
        k (int): Number of subsets to split the set into.
        
    Returns:
        dataframe: Data frame with 3 new columns for validation.
    """
    kf = KFold(n_splits = k)
    
    r2_validation = []
    nrmse_validation = []
    pbias_validation = []
    error = 0
    
    for i, row in analysis_res.iterrows():
        if row['calibrated'] == 0: # if the model was not calibrated, validation cannot be done
            for i in range(k):
                r2_validation.append(np.nan)
                nrmse_validation.append(np.nan)
                pbias_validation.append(np.nan)
        else:
            x_true = row['x']
            y_true = row['y']
            if len(x_true) < k:
                error = 1
                print('Cannot perform {}-fold validation for {} due to not enough data.'.format(k, row['country']))
                for i in range(k):
                    r2_validation.append(np.nan)
                    nrmse_validation.append(np.nan)
                    pbias_validation.append(np.nan)
            else:
                for train_index, test_index in kf.split(x_true):
                    x_train, x_test = x_true[train_index], x_true[test_index]
                    y_train, y_test = y_true[train_index], y_true[test_index]
                    fit = calibration(x_train, y_train)
                    y_pred = logistic(x_test, *fit)
        
        #             5-fold cross validation R2
                    r2 = calculate_r2(y_test, y_pred)
                    r2_validation.append(r2)
        
        #             5-fold cross validation NRMSE
                    nrmse = calculate_nrmse(y_test, y_pred)
                    nrmse_validation.append(nrmse)
        
        #             5-fold cross validation percentage bias
                    pbias = calculate_pbias(y_test, y_pred)
                    pbias_validation.append(pbias)
        
    # create chunks of length k and take mean of these to arrive at the validation values
    chunks_score = [r2_validation[x:x + k] for x in range(0, len(r2_validation), k)]
    analysis_res['r2_validation({}-fold)'.format(k)] = [np.mean(chunk) for chunk in chunks_score]
    
    chunks_score = [nrmse_validation[x:x + k] for x in range(0, len(nrmse_validation), k)]
    analysis_res['nrmse_validation({}-fold)'.format(k)] = [np.mean(chunk) for chunk in chunks_score]
    
    chunks_score = [pbias_validation[x:x + k] for x in range(0, len(pbias_validation), k)]
    analysis_res['pbias_validation({}-fold)'.format(k)] = [np.mean(chunk) for chunk in chunks_score]
    
    if error:
        print()
    
    return analysis_res

def plot_task6a(country_list, df):
    """
    Country plotting as requested in task 6a
    
    This function creates an SDG progress and simulation plot for selected countries.
    
    Args:
        country_list (list): List containing the names of countries as strings to be\
            displayed in the plot. 
        df (dataframe): Dataframe containing the coefficients of fitted models per country \
            in column 'fit' (with respect to column 'country') and the independent and \
            dependent variable in 'x' and 'y', respectively. 
    """
    my_subset = df.loc[df['country'].isin(country_list)]
    my_subset = my_subset[my_subset['calibrated'] == 1]

    cmap = plt.cm.get_cmap('jet')
    idxs = np.linspace(0, 1, len(my_subset)*3)
    fig, ax = plt.subplots(1)
    idx = 0
    for i, row in my_subset.iterrows():
        x = row['x']
        y = row['y']
        x_plot = np.arange(2000, 2030.25, step = 0.25)
        fit = row['fit']
        ax.plot(x_plot, logistic(x_plot, *fit), label = 'logistic fit {}'.format(row['country']), color = cmap(idxs[idx]))
        ax.scatter(x, y, label = 'historical data {}'.format(row['country']), color = cmap(idxs[idx + 1]))
        ax.scatter(2030, logistic(2030, *fit), label = '2030 expected value ({})'.format(np.round(logistic(2030, *fit), 3)), marker = 'x', color = cmap(idxs[idx + 2]))
        idx += 3

    ax.set_xlabel('year')
    ax.set_ylabel('SDG indicator (%)') # Proportion of population using safely managed drinking water services (%)
    ax.set_xlim(1999, 2031)
    ax.legend(bbox_to_anchor = (1, 1, 0, 0))
    
    # build the title of the plot as it should be saved - if too many countries, just call plot_countries.png
    t = 'plot_'
    if len(country_list) > 10:
        t += 'countries'
    else:
        for i in range(len(country_list)):
            t += '{}_'
        t = t[:-1]
    t += '.png'
    
    plt.savefig(t.format(*country_list), bbox_inches = 'tight')
    # plt.show()
    plt.close()

def plot_task6b(country_list, df, plot_title = ''):
    """
    Ordered dot plot as requested in task 6b
    
    This function creates an ordered dot plot to show growth rates and expected\
        indicator values for the top 30 populous countries in the world\
        (that are present in the data and have calibrated models).
    
    Args:
        country_list (list): List containing the names of countries to be displayed\
            in the plot. This is the list of the 30 most populous countries.
        df (dataframe): Dataframe containing the growth rate per country \
            in column 'growth_rate' and the expected indicator value in\
            column 'expected_indicator' (with respect to column 'country').
        plot_title (string) (optional): The title of the plot under which it\
            should be saved.
    """
    # first filter out countries with not recorded growth rates and sort the dataframe by 
    # growth rate in ascending order
    my_subset = df[np.isnan(df['growth_rate']) == False].sort_values('growth_rate', ascending = True)
    
    fig, ax = plt.subplots(1, figsize = (6, 10))
    my_plot = ax.scatter(my_subset['growth_rate'].values, np.arange(0, len(my_subset)), c = my_subset['expected_indicator'].values)
    ax.set_xlabel('growth rate')
    ax.set_ylabel('country')
    plt.yticks(np.arange(0, len(my_subset)), labels = my_subset['country'].values)
    plt.rcParams.update({'font.size': 18})
    ax.tick_params(axis = "y", labelsize = 14)
    
    # add a color bar
    cbar = plt.colorbar(my_plot)
    cbar.set_label('Expected indicator value (2030)')
    plt.savefig(plot_title, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    
def logistic(x, start, K, x_peak, r):
    """
    Logistic model
    
    This function runs a logistic model.
    
    Args:
        x (array_like): The control variable as a sequence of numeric values \
        in a list or a numpy array.
        start (float): The initial value of the return variable.
        K (float): The carrying capacity.
        x_peak (float): The x-value with the steepest growth.
        r (float): The growth rate.
        
    Returns:
        array_like: A numpy array or a single floating-point number with \
        the return variable.
    """
    
    if isinstance(x, list):
        x = np.array(x)
    return start + K / (1 + np.exp(r * (x_peak-x)))

def calibration(x, y):
    """
    Calibration
    
    This function calibrates a logistic model.
    The logistic model can have a positive or negative growth.
    
    Args:
        x (array_like): The explanatory variable as a sequence of numeric values \
        in a list or a numpy array.
        y (array_like): The response variable as a sequence of numeric values \
        in a list or a numpy array.
        
    Returns:
        tuple: A tuple including four values: 1) the initial value (start), \
        2) the carrying capacity (K), 3) the x-value with the steepest growth \
        (x_peak), and 4) the growth rate (r).
    """
    if isinstance(x, pd.Series): x = x.to_numpy(dtype='int')
    if isinstance(y, pd.Series): y = y.to_numpy(dtype='float')
    
    if len(np.unique(y)) == 1:
        return y[0], 0, 2000.0, 0
    
    # initial parameter guesses
    slope = [None] * (len(x) - 1)
    for i in range(len(slope)):
        slope[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        slope[i] = abs(slope[i])
    x_peak = x[slope.index(max(slope))] + 0.5
    
    if y[0] < y[-1]: # positive growth
        start = min(y)
        K = 2 * (sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2 - start)
    else: # negative growth
        K = 2 * (max(y) - sum([y[slope.index(max(slope))], \
                        y[slope.index(max(slope))+1]])/2)
        start = max(y) - K
    
    # curve fitting
    popt, _ = curve_fit(logistic, x, y, p0 = [start, K, x_peak, 0], maxfev = 10000,
                        bounds = ([0.5*start, 0.5*K, 1995, -10],
                                  [2*(start+0.001), 2*K, 2030, 10]))
    # +0.001 so that upper bound always larger than lower bound even if start = 0
    return popt
