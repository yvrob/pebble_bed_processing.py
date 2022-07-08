import numpy as np
import pandas as pd
import serpentTools
import difflib
import matplotlib.pyplot as plt

def find_matching(string, list_strings, infer_if_incorrect=True, lower_ok=True, cutoff=0.4):
    if lower_ok:
        keys = [key.lower() for key in list_strings]
        search_parameter = string.lower()
    else:
        keys = [key for key in list_strings]
        search_parameter = str(string)

    if search_parameter not in keys:
        if infer_if_incorrect:
            match_string = difflib.get_close_matches(search_parameter, keys, cutoff=cutoff, n=100000)
            if len(match_string) == 0:
                raise Exception(f'String {string} not found, even when trying to infer.')
            match_indices = [keys.index(key) for key in match_string] 
            real_matches = [list_strings[i] for i in match_indices]
            print(f'String {string} not found, close matches are: {real_matches}. Picking the first one.')
            string = real_matches[0]
        else:
            raise Exception(f'String {string} not found.')
    else:
        index = keys.index(string.lower())
        string = list_strings[index]
    return string

def extract_res(file, parameter, infer_if_incorrect=True, which_rows='all', which_columns='all'):
    res_reader = serpentTools.read(file)
    parameter = find_matching(parameter, list(res_reader.resdata.keys()), infer_if_incorrect=infer_if_incorrect, lower_ok=True, cutoff=0.4)
    array = res_reader.resdata[parameter]
    if isinstance(which_rows, str) and which_rows == 'all':
        which_rows = [i for i in range(array.shape[0])]
    array = array[which_rows, :]
    if isinstance(which_columns, str) and which_columns == 'all':
        which_columns = [i for i in range(array.shape[1])]
    array = array[:, which_columns]

    return array

def plot_res(file, parameter, infer_if_incorrect=True, which_rows='all', which_columns='all', which_columns_errors=None, use_time=True, ylabel=None, plot_labels='', new_fig=True):
    array = extract_res(file, parameter, infer_if_incorrect, which_rows, which_columns)
    if use_time:
        x = extract_res(file, 'burnDays', which_rows=which_rows, which_columns=0)
        xlabel = 'Time [EPFD]'
    else:
        x = extract_res(file, 'burnStep', which_rows=which_rows, which_columns=0)
        xlabel = 'Step #'

    if new_fig:
        plt.figure()

    if isinstance(which_columns_errors, type(None)):
        plt.plot(x, array, label=plot_labels)
    else:
        array_err = extract_res(file, parameter, infer_if_incorrect, which_rows=which_rows, which_columns=which_columns_errors)
        plt.errorbar(x, array, yerr=array_err, label=plot_labels)
    plt.xlabel(xlabel)
    if isinstance(ylabel, type(None)):
        plt.ylabel(parameter)
    else:
        plt.ylabel(ylabel)

def read_discarded(file, Nlines=None):
    if isinstance(Nlines, type(None)):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file, nrows=Nlines)
    return df    

def statistical_distribution(file, parameter, bins=10, no_init=True, Nlines=None, alpha=0.6, infer_if_incorrect=True, which_rows='all', xlabel=None, plot_labels='', new_fig=True):
    df = read_discarded(file)
    if isinstance(which_rows, str) and which_rows == 'all':
        which_rows = [i for i in range(df.shape[0])]
    df = df.iloc[which_rows]
    parameter = find_matching(parameter, df.columns, infer_if_incorrect=infer_if_incorrect, lower_ok=True, cutoff=0.4)
    if no_init:
        df = df.loc[~np.isnan(df.init)]
    array = df[parameter]

    if new_fig:
        plt.figure()
    plt.gca().set_axisbelow(True)
    array.hist(bins=bins, alpha=alpha)
    if isinstance(xlabel, type(None)):
        plt.xlabel(parameter)
    else:
        plt.xlabel(xlabel)
    plt.ylabel('Number of occurences')
    return array

#file = '/home/yryves/serpent_cases/domain_decomposition_dvlpt/test_larger2/wrk_Serpent/input_res.m'
#plot_res(file, 'anakeff', which_columns=0, which_columns_errors=1)
#plt.show()
file = '/home/yryves/serpent_cases/domain_decomposition_dvlpt/test_larger2/waste_53'
bu = statistical_distribution(file, 'burnup', bins=30)
plt.savefig('bu_waste.png', dpi=300)
passes = statistical_distribution(file, 'pass', no_init=True, bins=30)
plt.savefig('pass_waste.png', dpi=300)
