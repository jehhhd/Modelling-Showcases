def fit(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import t

    percentage_increases = calculate_percentage_increases(df, 'Close')[1]

    params = t.fit(percentage_increases)

    # Extract the fitted parameters
    fitted_df, fitted_loc, fitted_scale = params[0], params[1], params[2]

    # Ensure the parameters are valid
    if fitted_df <= 0:
        raise ValueError("Invalid fitted parameters: df must be positive")

    # Generate theoretical quantiles using the ppf function of the fitted distribution
    theoretical_quantiles_t = t.ppf(np.linspace(0.01, 0.99, 30), df=fitted_df, loc=fitted_loc, scale=fitted_scale)

    # Generate sample quantiles from the data
    sample_quantiles_t = np.percentile(percentage_increases, np.linspace(1, 99, 30))

    # Get theoretical quantiles for normal distribution using qq_plot function
    theoretical_quantiles_normal, _ = qq_plot(percentage_increases)

    # Generate sample quantiles for normal distribution
    _, sample_quantiles_normal = qq_plot(percentage_increases)

    # Create QQ plot
    plt.figure(figsize=(8, 6))

    # Plot blue dots for t distribution
    plt.plot(theoretical_quantiles_t, sample_quantiles_t, 'bo', label='Sample Quantiles (t)')

    # Plot red dots for normal distribution
    plt.plot(theoretical_quantiles_normal, sample_quantiles_normal, 'ro', label='Sample Quantiles (Normal)')

    plt.plot(theoretical_quantiles_t, theoretical_quantiles_t, 'r--', label='Theoretical Quantiles')

    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    plt.title('QQ Plot for t Distribution')

    plt.legend()

    plt.grid(True)

    plt.show()

def qq_plot(data):
    import numpy as np
    """
    Function to create a QQ plot for a normal distribution.
    
    Parameters:
    data (array-like): Input data for QQ plot.
    
    Returns:
    theoretical_quantiles, sample_quantiles: Theoretical and sample quantiles for the normal distribution.
    """
    # Generate theoretical quantiles using a normal distribution
    theoretical_quantiles = np.percentile(np.random.normal(loc=np.mean(data), scale=np.std(data), size=len(data)), np.linspace(1, 99, 30))

    # Generate sample quantiles from the original data
    sample_quantiles = np.percentile(data, np.linspace(1, 99, 30))

    return theoretical_quantiles, sample_quantiles

def calculate_percentage_increases(df, column):
    results = {}
    n = 1
    while True:
        percentage_increases = (df[column].shift(-n) - df[column]) / df[column]
        percentage_increases = percentage_increases.dropna().to_numpy()
        if len(percentage_increases) < 100:
            break
        results[n] = percentage_increases
        n += 1
    return results

def trading_sim_options(df, theo_price_tested, value_parameter):
    import matplotlib.pyplot as plt
    import pandas as pd

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    subset_traded = df[((df[theo_price_tested]-df['option_price'])/df[theo_price_tested]) > value_parameter]
    subset_traded['cumulative_profit'] = subset_traded['profit'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(subset_traded['date'], subset_traded['cumulative_profit'], label='Cumulative Profit')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.title('Cumulative Profit Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()



def pair_sim(plot, dff, ratio_retreat, offset):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    #Initiate tracking tools
    amount_eth = 0
    amount_btc = 0
    bets = 0

    #These trackers test that the corridor algorithm is working as expected
    tracker = []
    bets_tracker = []
    time_list = []
    idx_list = []
    theo_ratio_tracker = []

    #'overall' stores how much is invested
    overall_tracker = []
    overall = 0

    #Starting 'ratio' value
    theo_ratio = dff.loc[dff.index.min(), 'ratio']

    #Initialise corridor
    upper = theo_ratio + offset
    lower = theo_ratio - offset

    #Iterate through hours
    for i in dff.index:

        #'btc' and 'eth' are used as place holders for pairs' stock tickers
        btc = dff.loc[i,'btc']
        eth = dff.loc[i,'eth']
        ratio = dff.loc[i,'ratio']

        #Trade condition
        if ratio>upper:
            # print(dff.loc[i, 'Time'])
            #Trade $1 size
            amount_eth += (1/2)/eth
            amount_btc -= (1/2)/btc

            #New corridor
            theo_ratio += ratio_retreat
            upper = theo_ratio + offset
            lower = theo_ratio - offset

            bets += 1
            bets_tracker.append(1)

            overall += 1

        elif ratio<lower:
            # print(dff.loc[i, 'Time'])
            #Trade $1 size
            amount_eth -= (1/2)/eth
            amount_btc += (1/2)/btc

            #New corrifor
            theo_ratio -= ratio_retreat
            upper = theo_ratio + offset
            lower = theo_ratio - offset

            bets += 1
            bets_tracker.append(1)

            overall -= 1

        else:
            bets_tracker.append(0)

        #Tracking
        tracker.append(amount_eth*eth + amount_btc*btc - 0.005*bets)
        theo_ratio_tracker.append(theo_ratio)
        time_list.append(dff.loc[i, 'Time'])
        overall_tracker.append(overall)
        idx_list.append(i)

    
    #Store results
    res = pd.DataFrame({'time': time_list, 'idx': idx_list, 'cumulative': tracker, 'btc':list(dff['btc']), 'eth':list(dff['eth']), 'ratio':list(dff['ratio']), 'bets': bets_tracker, 'theo_ratio': theo_ratio_tracker, 'overall': np.abs(overall_tracker)/2})
    
    #Plotting
    if plot == True:
        print('% Trades: '+str(bets/dff.shape[0]))
        cumulative_values = res['cumulative']
        indices = res['idx']
        bets = res['bets']

        # Find the minimum y value
        min_y_value = min(cumulative_values)
        min_x_value = min(indices)

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(20, 5))

        # Plot cumulative values on primary y-axis
        ax1.plot(indices, cumulative_values, 'b-')
        ax1.set_ylim(bottom=min_y_value)
        ax1.set_xlim(left=min_x_value)
        ax1.grid(True, axis='y')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Cumulative Plot Starting from Minimum Y Value')
        ax1.set_yticks(list(ax1.get_yticks()) + [min_y_value])

        # Create secondary y-axis and plot bets
        ax2 = ax1.twinx()
        ax2.plot(indices, bets, 'r-', alpha=0.3)
        ax2.set_ylabel('Bets', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        every_390th_value = indices[389::390]
        for x in every_390th_value:
            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

        dff['Time'] = pd.to_datetime(dff['Time'])
        time_labels = dff.loc[indices, 'Time'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Format as needed
        ax1.set_xticks(indices[::390])  # Set x-ticks at the same interval as vertical lines
        ax1.set_xticklabels(time_labels[::390], rotation=45, ha='right')  # Rotate for better readability

        plt.show()

    return res, bets




def combine_pair_sims(plot, filtered_dfs1, filtered_dfs_by_year, ratio_retreat, years):
    import warnings
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    # Suppress specific FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")


    filtered_dfs_combined = {}
    for year, year_dict in filtered_dfs_by_year.items():
        if year in years:
            for key, value in year_dict.items():
                filtered_dfs_combined[key] = value



    bets_overall = 0
    hours_overall = 0
    returns = 0
    invested = filtered_dfs1[list(filtered_dfs1.keys())[0]].set_index('Time')['ratio']
    returned = invested.copy()
    invested[:] = 0
    returned[:] = 0
    cumulative_problem = {}
    pair_totals = {}
    year_totals = {}

    #Iterate through pairs
    for x in list(filtered_dfs_combined.keys()):

        df = filtered_dfs_combined[x]

        if df.shape[0]>20:
            
            #Run simulation code with optimal parameters
            [res, bets] = pair_sim(False, df, ratio_retreat, ratio_retreat)
            returns += res.loc[res.index[-1], 'cumulative']
            bets_overall += bets
            hours_overall += df.shape[0]

            key = re.split(r'\d', x, 1)[0]
            if key not in pair_totals:
                pair_totals[key] = 0
            pair_totals[key] += res.loc[res.index[-1], 'cumulative']

            last_year = res['time'].iloc[-1].year
            if last_year not in year_totals:
                year_totals[last_year] = 0
            year_totals[last_year] += res.loc[res.index[-1], 'cumulative']

            #Combine metrics of individual pairs 
            if res.loc[res.index[-1], 'time'] in cumulative_problem:
                cumulative_problem[res.loc[res.index[-1], 'time']] += res.loc[res.index[-1], 'cumulative']
            else:
                cumulative_problem[res.loc[res.index[-1], 'time']] = res.loc[res.index[-1], 'cumulative']
            invested = invested.add(res.set_index('time')['overall'],  fill_value=0)
            returned = returned.add(res.set_index('time')['cumulative'],  fill_value=0)




    #Combine metrics of individual pairs (taking into account final P&L for pairs' trading periods that should be carried forward statically)
    summ = 0
    for key, value in cumulative_problem.items():
        # Find all indices in 'returned' that are later than the current key
        later_indices = returned.index[returned.index > key]
        
        # Add the value to all these indices
        returned.loc[later_indices] += value
        summ += value

    num_years = (returned.index[-1] - returned.index[0]).days / 365.0
    if plot == True:
        print('% Return Per Trade: '+str(2*100*returned[-1]/bets_overall))
        print('Trades: '+str(bets_overall/2))
        # res_df = pd.DataFrame(returned)
        # res_df['invested'] = invested
        # split_dfs = np.array_split(res_df[res_df['invested'] != 0], 6)
        # monthly_returns = []
        # for i, part in enumerate(split_dfs):
        #     monthly_returns.append((part[0].iloc[-1] - part[0].iloc[0])/np.mean(part['invested']))
        # print('Sharpe: '+str(np.mean(monthly_returns)/np.std(monthly_returns)))
        plt.plot(returned)
        plt.ylabel('P&L')
        plt.show()
        return pair_totals
    else:
        return 2*100*returned[-1]/bets_overall







def prepare_pairs_data(australian_pairs, earnings_dates_dict):
    import pandas as pd

    # Initialize dictionaries to store filtered dataframes for each year
    filtered_dfs_by_year = {year: {} for year in range(2020, 2026)}
    filtered_dfs1 = {}

    for file in australian_pairs:
        df = pd.read_csv('Data/ibkr_stocks1/'+file + '.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        # Rename columns
        df.rename(columns={df.columns[1]: 'btc', df.columns[2]: 'eth'}, inplace=True)
        df.rename(columns={'date': 'Time'}, inplace=True)
        
        filtered_dfs1[file] = df

    # Split each dataframe into 5 mini dataframes based on between earnings dates/dividend periods and remove non-market hours
    for file in australian_pairs:
        df = filtered_dfs1[file]
        df['Time'] = df['Time'].str.split('+').str[0]
        #df['Time'] = df['Time'].str.rsplit('-', n=1).str[0]
        earnings_dates = earnings_dates_dict[file]
        
        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'])
        # Convert earnings dates to datetime
        earnings_dates = pd.to_datetime(earnings_dates)
        earnings_dates = sorted(earnings_dates)
        
        # Initialize list to hold split dataframes
        split_dfs = []
        
        # Split the dataframe into parts
        previous_date = df['Time'].min()
        for date in earnings_dates:
            # Exclude the day before, the earnings date, and the day after
            exclusion_dates = [
                (date - pd.Timedelta(days=1)).date(),
                date.date(),
                (date + pd.Timedelta(days=1)).date()
            ]

            df = df[~df['Time'].apply(lambda x: x.date()).isin(exclusion_dates)]

            # Split the dataframe
            split_df = df[(df['Time'] >= pd.Timestamp(previous_date)) & (df['Time'] < pd.Timestamp(date))]
            
            # Ensure the split_df is from 2023 or later
            if not split_df.empty and split_df['Time'].iloc[-1].year >= 2019:
                split_dfs.append(split_df)

            # Update previous_date to the day after the excluded window
            previous_date = date

        # Add the last segment
        split_df = df[df['Time'] >= previous_date]
        split_df.reset_index(inplace=True, drop=True)
        split_dfs.append(split_df)
        
        # Add split dataframes to the appropriate dictionary
        for i, split_df in enumerate(split_dfs):
            split_df.reset_index(inplace=True, drop=True)
            
            # Scale ratio values to near 1
            if not split_df.empty:
                split_df['ratio'] = split_df['ratio'] / split_df.loc[0, 'ratio']
            
            # Determine which dictionary to add the split dataframe to based on year
            year = split_df['Time'].iloc[1].year if not split_df.empty else None
            if year in filtered_dfs_by_year:
                filtered_dfs_by_year[year][f"{file}{i+1}"] = split_df

    # Example access to one of the split dataframes, note that 'btc' and 'eth' are used as placeholders for the names of each company pair to help code syntax
    return filtered_dfs_by_year, filtered_dfs1

    
