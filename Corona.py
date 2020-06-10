import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import git
import datetime
import subprocess
import pikepdf
import httplib2
import os
import oauth2client
import base64
import mimetypes
import email.encoders
import tempfile
import pathlib

from matplotlib.backends.backend_pdf import PdfPages
from oauth2client import client, tools, file
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from apiclient import errors, discovery
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase


def Corona_App():

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # LOCATIONS ...
    ##########################################################################################################
    # We will use the SYSTEM assigned location for TEMP files
    # We will use the directory from which this script was started as the working directory
    # here we create a "Corona Data" directory, where we download a copy of the John Hopkins Repository (the data)
    # 'https://github.com/CSSEGISandData/COVID-19.git'
    # and inside the "Corona Data" directory we will STORE the resulting report in the directory "Reports"
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    tempdir = pathlib.Path(tempfile.gettempdir())
    workingdir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

    data_root = pathlib.Path(workingdir / "Corona Data")
    data_root.mkdir(parents=True, exist_ok=True)

    temp_data_root = tempdir / "CoronaAPP"
    if temp_data_root.exists(): os.system('rmdir /s/Q "' + str(temp_data_root.absolute()) + '"')
    temp_data_root.mkdir(parents=True)

    temp_figures_pdf = temp_data_root / (str(datetime.date.today()) + " Corona Death Graphs FIGURES.pdf")
    temp_cover_pdf = temp_data_root / (str(datetime.date.today()) + " Corona Death Graphs COVER.pdf")

    report_root = pathlib.Path(data_root / "Reports")
    report_root.mkdir(parents=True, exist_ok=True)

    report_filename = str(datetime.date.today()) + " Corona Death Graphs.pdf"
    report_loc = report_root / report_filename

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # CORE assumptions for all calculations ...
    ##########################################################################################################
    # We use KAHN's formula to calculate estimated number of infected people at any given time
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    estimated_mortality_rate = 0.01  # 1%
    estimated_time_to_die = 17.33  # Days from infection to death (real infection NOT contagious
    estimated_days_to_double = 6.18

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # DATE / TIME string representations
    ##########################################################################################################
    # Adjust to your match your locality
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    date_stamp_plot = datetime.datetime.now().strftime("%d %b %Y\n%H:%M")

    date_stamp_email = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # EMAIL settings
    ##########################################################################################################
    # Adjust to match your preferences / accounts
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    email_to = ""
    email_cc = ""
    email_bcc = """
    
    youremail@mailbox.com
    myemail@mailbox.com
    
	"""


    # add a "," for each NEWLINE => Split all entries => remove "," => join all addresses, separated with ","
    email_to = ",".join([x.replace(",", "") for x in email_to.replace("\n,", "").split()])
    email_cc = ",".join([x.replace(",", "") for x in email_cc.replace("\n,", "").split()])
    email_bcc = ",".join([x.replace(",", "") for x in email_bcc.replace("\n,", "").split()])

    email_sender = "youremail@mailbox.com"

    # storing the subject
    email_subject = "COVID-19 Update for Santa Clara " + date_stamp_email

    email_body_plain = """
	
	This is a DAILY update of the COVID-19 pandemic.
	
	The numbers are based on:
	
			2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE
	
	FEEDBACK / QUESTIONS:
	
			Please send an email to youremail@mailbox.com
			   
	Want to be ADDED to the list:
	
			Please send an email to youremaillist@mailbox.com
				
	"""

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # AUXILIARY settings
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    # We created a HTML layout template using MS WORD
    Cover_Letter = workingdir / "Corona_Cover.html"
    with open(Cover_Letter) as file:
        HTML_Cover_Text = file.read()
        file.close()


    fig_text = {'x': 0.65, 'y': 0.76, 's': date_stamp_plot, 'fontsize': 65, 'color': 'black', 'ha': 'left',
                'va': 'bottom',
                'alpha': 0.5}

    multiplot_padding = 35

    # create a PdfPages object
    pdf = PdfPages(temp_figures_pdf)

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # Getting the DATA from the Johns Hopkins GIT repository
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    Corona_Dir = data_root / "Corona by Johns Hopkins"

    if (Corona_Dir.exists()):

        git.Repo(Corona_Dir).remotes.origin.pull()

        if git.Repo(Corona_Dir).is_dirty(untracked_files=True):

            os.system('rmdir /s/Q "' + str(Corona_Dir.absolute()) + '"')

            Corona_Dir.mkdir(parents=True)

            git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)

    else:  # NO Dir

        Corona_Dir.mkdir(parents=True)

        git.Repo.clone_from('https://github.com/CSSEGISandData/COVID-19.git', Corona_Dir, branch='master', depth=1)

    confirmed_loc = Corona_Dir / "csse_covid_19_data/csse_covid_19_time_series" / "time_series_covid19_confirmed_global.csv"
    death_loc     = Corona_Dir / "csse_covid_19_data/csse_covid_19_time_series" / "time_series_covid19_deaths_global.csv"
    recovered_loc = Corona_Dir / "csse_covid_19_data/csse_covid_19_time_series" / "time_series_covid19_recovered_global.csv"
    US_death_loc  = Corona_Dir / "csse_covid_19_data/csse_covid_19_time_series" / "time_series_covid19_deaths_US.csv"

    confirmed_data = pd.read_csv(confirmed_loc)
    death_data = pd.read_csv(death_loc)
    recovered_data = pd.read_csv(recovered_loc)
    US_death_data = pd.read_csv(US_death_loc)

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # DATA cleanup for further processing
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    def column_swap(A):
        return A.iloc[:, [1, 0] + list(range(2, A.shape[1]))]

    confirmed_data = column_swap(confirmed_data)
    death_data = column_swap(death_data)
    recovered_data = column_swap(recovered_data)
    US_death_data = column_swap(US_death_data.iloc[:, 6:])

    Cal_death_data = US_death_data[US_death_data.Province_State == 'California'].rename(
        columns={'Combined_Key': 'County'})

    Cal_death_data.County = Cal_death_data.County.str.split(',').str[0].str.strip()

    US_death_data = US_death_data.drop('Combined_Key', axis=1)

    US_population = US_death_data['Population'].astype('int64')
    Cal_population_county = Cal_death_data.groupby(['County'])[Cal_death_data.columns[5]].sum().astype('int64')

    US_population_state = US_death_data.groupby(['Province_State'])[US_death_data.columns[4]].sum().astype('int64')
    US_death_data = US_death_data.drop('Population', axis=1)

    Cal_population = sum(Cal_population_county)
    Cal_death_data = Cal_death_data.drop('Population', axis=1)
    Cal_death_data = Cal_death_data.drop('Country_Region', axis=1)

    a = list(Cal_death_data.columns)
    a.insert(1, a.pop(a.index('County')))
    Cal_death_data = Cal_death_data[a]

    # FIX error in RAW data
    row = Cal_death_data.loc[Cal_death_data.County == 'Sacramento'].index.values.max()
    Cal_death_data.at[row, '4/9/20'] = 23

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # DATA pre-processing
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    # We insert 20 (estimated_time_to_die) days to the beginning of the dataset,
    # to capture the estimated infected at that time.

    # We know the original data starts with 1-22-2020
    start_date = pd.to_datetime(confirmed_data.columns[4], format='%m/%d/%y')
    for date in range(int(round(estimated_time_to_die))):
        insert_date = start_date - pd.Timedelta(days=date + 1)

        confirmed_data.insert(4, insert_date.strftime("%m/%d/%y"), 0)
        death_data.insert(4, insert_date.strftime("%m/%d/%y"), 0)
        recovered_data.insert(4, insert_date.strftime("%m/%d/%y"), 0)
        US_death_data.insert(4, insert_date.strftime("%m/%d/%y"), 0)
        Cal_death_data.insert(4, insert_date.strftime("%m/%d/%y"), 0)

    # We want to also analyze the COUNTRY data
    # ==> need to combine the regions for each country
    confirmed_data_country = confirmed_data.groupby(['Country/Region'])[confirmed_data.columns[4:]].sum()
    death_data_country = death_data.groupby(['Country/Region'])[death_data.columns[4:]].sum()
    recovered_data_country = recovered_data.groupby(['Country/Region'])[recovered_data.columns[4:]].sum()
    US_death_data_state = US_death_data.groupby(['Province_State'])[US_death_data.columns[4:]].sum()
    Cal_death_data_county = Cal_death_data.groupby(['County'])[Cal_death_data.columns[4:]].sum()

    confirmed_data_continental = confirmed_data.loc[confirmed_data['Lat'] > 30][confirmed_data.columns[4:]].sum()
    death_data_continental = death_data.loc[death_data['Lat'] > 30][death_data.columns[4:]].sum()
    recovered_data_continental = recovered_data.loc[recovered_data['Lat'] > 30][recovered_data.columns[4:]].sum()

    confirmed_data_subtropical = confirmed_data.loc[confirmed_data['Lat'] <= 30][confirmed_data.columns[4:]].sum()
    death_data_subtropical = death_data.loc[death_data['Lat'] <= 30][death_data.columns[4:]].sum()
    recovered_data_subtropical = recovered_data.loc[recovered_data['Lat'] <= 30][recovered_data.columns[4:]].sum()

    US_death_data_east = US_death_data.loc[US_death_data['Long_'] > -100][US_death_data.columns[4:]].sum()
    US_death_data_west = US_death_data.loc[US_death_data['Long_'] <= -100][US_death_data.columns[4:]].sum()

    Cal_death_data_north = Cal_death_data.loc[Cal_death_data['Lat'] > 35.791111][Cal_death_data.columns[4:]].sum()
    Cal_death_data_south = Cal_death_data.loc[Cal_death_data['Lat'] <= 35.791111][Cal_death_data.columns[4:]].sum()

    def clean_missing_data_nan(df):
        df.apply(pd.to_numeric)
        if isinstance(df, pd.DataFrame):
            return df.mask(df.eq(0)).ffill(axis=1).fillna(0)
        else:
            return df.mask(df.eq(0)).ffill(axis=0).fillna(0)

    numeric_confirmed_data = clean_missing_data_nan(confirmed_data.iloc[:, 4:])
    numeric_death_data = clean_missing_data_nan(death_data.iloc[:, 4:])
    numeric_recovered_data = clean_missing_data_nan(recovered_data.iloc[:, 4:])
    numeric_US_death_data = clean_missing_data_nan(US_death_data.iloc[:, 4:])
    numeric_Cal_death_data = clean_missing_data_nan(Cal_death_data.iloc[:, 4:])

    numeric_confirmed_data_country = clean_missing_data_nan(confirmed_data_country.iloc[:, 4:])
    numeric_death_data_country = clean_missing_data_nan(death_data_country.iloc[:, 4:])
    numeric_recovered_data_country = clean_missing_data_nan(recovered_data_country.iloc[:, 4:])
    numeric_US_death_data_state = clean_missing_data_nan(US_death_data_state.iloc[:, 4:])
    numeric_Cal_death_data_county = clean_missing_data_nan(Cal_death_data_county.iloc[:, 4:])

    numeric_confirmed_data_continental = clean_missing_data_nan(confirmed_data_continental)
    numeric_death_data_continental = clean_missing_data_nan(death_data_continental)
    numeric_recovered_data_continental = clean_missing_data_nan(recovered_data_continental)

    numeric_confirmed_data_subtropical = clean_missing_data_nan(confirmed_data_subtropical)
    numeric_death_data_subtropical = clean_missing_data_nan(death_data_subtropical)
    numeric_recovered_data_subtropical = clean_missing_data_nan(recovered_data_subtropical)

    numeric_US_death_data_east = clean_missing_data_nan(US_death_data_east)
    numeric_US_death_data_west = clean_missing_data_nan(US_death_data_west)

    numeric_Cal_death_data_north = clean_missing_data_nan(Cal_death_data_north)
    numeric_Cal_death_data_south = clean_missing_data_nan(Cal_death_data_south)

    def change_column_to_datetime(df):
        if isinstance(df, pd.DataFrame):
            df.columns = pd.to_datetime(df.columns, format='%m/%d/%y')

        else:
            df.index = pd.to_datetime(df.index, format='%m/%d/%y')

    change_column_to_datetime(numeric_confirmed_data)
    change_column_to_datetime(numeric_death_data)
    change_column_to_datetime(numeric_recovered_data)
    change_column_to_datetime(numeric_US_death_data)
    change_column_to_datetime(numeric_Cal_death_data)
    change_column_to_datetime(numeric_Cal_death_data)

    change_column_to_datetime(numeric_confirmed_data_country)
    change_column_to_datetime(numeric_death_data_country)
    change_column_to_datetime(numeric_recovered_data_country)
    change_column_to_datetime(numeric_US_death_data_state)
    change_column_to_datetime(numeric_Cal_death_data_county)

    change_column_to_datetime(numeric_confirmed_data_continental)
    change_column_to_datetime(numeric_death_data_continental)
    change_column_to_datetime(numeric_recovered_data_continental)

    change_column_to_datetime(numeric_confirmed_data_subtropical)
    change_column_to_datetime(numeric_death_data_subtropical)
    change_column_to_datetime(numeric_recovered_data_subtropical)

    change_column_to_datetime(numeric_US_death_data_east)
    change_column_to_datetime(numeric_US_death_data_west)

    change_column_to_datetime(numeric_Cal_death_data_north)
    change_column_to_datetime(numeric_Cal_death_data_south)

    #########################################################################
    ## EXTRACT daily CHANGES
    #########################################################################

    # Extract the DAILY changes from the data.
    def make_daily(df):
        if isinstance(df, pd.DataFrame):
            nparray = np.array(df)
            nparray[:, 1:] -= nparray[:, :-1].copy()
            return nparray

        else:
            nparray = np.array(df)
            nparray[1:] -= nparray[:-1].copy()
            return nparray

    # estimated_infected_total.iloc[:,1:] -= estimated_infected_total.iloc[:,:-1].copy()
    numeric_confirmed_data_daily = make_daily(numeric_confirmed_data)
    numeric_death_data_daily = make_daily(numeric_death_data)
    numeric_recovered_data_daily = make_daily(numeric_recovered_data)
    numeric_US_death_data_daily = make_daily(numeric_US_death_data)
    numeric_Cal_death_data_daily = make_daily(numeric_Cal_death_data)

    numeric_confirmed_data_daily_country = make_daily(numeric_confirmed_data_country)
    numeric_death_data_daily_country = make_daily(numeric_death_data_country)
    numeric_recovered_data_daily_country = make_daily(numeric_recovered_data_country)
    numeric_US_death_data_daily_state = make_daily(numeric_US_death_data_state)
    numeric_Cal_death_data_daily_county = make_daily(numeric_Cal_death_data_county)

    numeric_confirmed_data_daily_continental = make_daily(numeric_confirmed_data_continental)
    numeric_death_data_daily_continental = make_daily(numeric_death_data_continental)
    numeric_recovered_data_daily_continental = make_daily(numeric_recovered_data_continental)

    numeric_confirmed_data_daily_subtropical = make_daily(numeric_confirmed_data_subtropical)
    numeric_death_data_daily_subtropical = make_daily(numeric_death_data_subtropical)
    numeric_recovered_data_daily_subtropical = make_daily(numeric_recovered_data_subtropical)

    numeric_US_death_data_daily_east = make_daily(numeric_US_death_data_east)
    numeric_US_death_data_daily_west = make_daily(numeric_US_death_data_west)

    numeric_Cal_death_data_daily_north = make_daily(numeric_Cal_death_data_north)
    numeric_Cal_death_data_daily_south = make_daily(numeric_Cal_death_data_south)

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # DATA processing
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    def estimated_days_to_double_calc(nparray):
        a = 1

    def calculate_estimated_infected(nparray):
        # calculate the # of infected people based on # of death
        # Death = 1 with mortality rate 1%  ==> 20 days ago there were 100 infected
        # the 100 infected double every 5 days ==> total # of infected at the time we record 1 death

        if nparray.ndim > 1:

            tdd = estimated_days_to_double_calc(nparray)
            new = (nparray / estimated_mortality_rate) * (2 ** (estimated_time_to_die / estimated_days_to_double))
            new = np.roll(new, - int(round(estimated_time_to_die)), axis=1)
            total = np.cumsum(new, axis=1)
            # The number of death TODAY only tell us what the estimated infected people where
            # 20 (estimated_time_to_die) days ago.
            total = np.roll(total, - int(round(estimated_time_to_die)), axis=1)

            return (new, total)

        else:

            new = (nparray / estimated_mortality_rate) * (2 ** (estimated_time_to_die / estimated_days_to_double))
            new = np.roll(new, - int(round(estimated_time_to_die)))
            total = np.cumsum(new)
            total = np.roll(total, - int(round(estimated_time_to_die)))

            return (new, total)

    estimated_infected_new, estimated_infected_total = calculate_estimated_infected(numeric_death_data_daily)

    US_estimated_infected_new, US_estimated_infected_total = calculate_estimated_infected(numeric_US_death_data_daily)

    Cal_estimated_infected_new, Cal_estimated_infected_total = calculate_estimated_infected(
        numeric_Cal_death_data_daily)

    estimated_infected_new_country, estimated_infected_total_country = calculate_estimated_infected(
        numeric_death_data_daily_country)

    US_estimated_infected_new_state, US_estimated_infected_total_state = calculate_estimated_infected(
        numeric_US_death_data_daily_state)

    Cal_estimated_infected_new_county, Cal_estimated_infected_total_county = calculate_estimated_infected(
        numeric_Cal_death_data_daily_county)

    estimated_infected_new_continental, estimated_infected_total_continental = calculate_estimated_infected(
        numeric_death_data_daily_continental)

    estimated_infected_new_subtropical, estimated_infected_total_subtropical = calculate_estimated_infected(
        numeric_death_data_daily_subtropical)

    US_estimated_infected_new_east, US_estimated_infected_total_east = calculate_estimated_infected(
        numeric_US_death_data_daily_east)

    US_estimated_infected_new_west, US_estimated_infected_total_west = calculate_estimated_infected(
        numeric_US_death_data_daily_west)

    Cal_estimated_infected_new_north, Cal_estimated_infected_total_north = calculate_estimated_infected(
        numeric_Cal_death_data_daily_north)

    Cal_estimated_infected_new_south, Cal_estimated_infected_total_south = calculate_estimated_infected(
        numeric_Cal_death_data_daily_south)

    def growth_rate(df):
        if isinstance(df, pd.DataFrame):
            n = np.array(df.iloc[:, 1:])
            d = np.array(df.iloc[:, :-1].copy())
            result = np.where(d == 0, d, n / d)
            result = np.where(result == 0, 0, result - 1) * 100
            return result
        else:
            n = np.array(df.iloc[1:])
            d = np.array(df.iloc[:-1].copy())
            result = np.where(d == 0, d, n / d)
            result = np.where(result == 0, 0, result - 1) * 100
            return result

    World_grows_rate = growth_rate(numeric_death_data_country)

    US_grows_rate = growth_rate(numeric_US_death_data_state)

    Cal_grows_rate = growth_rate(numeric_Cal_death_data_county)

    def probability_infected(population, total_infected):
        d = population.loc[population.values == 0] = population.max()
        n = total_infected.T
        result = np.amax(np.nan_to_num((np.where(d == 0, d, n / d)).T, copy=True, nan=0.0, posinf=0.0, neginf=0.0),
                         axis=1)
        return result

    US_probability_infected_state = probability_infected(US_population_state, US_estimated_infected_total_state)

    Cal_probability_infected_county = probability_infected(Cal_population_county, Cal_estimated_infected_total_county)

    death_data_analysis = death_data.iloc[:, :4]

    death_data_analysis["Total_Cumulative_Confirmed_Death"] = numeric_death_data[numeric_death_data.columns[-1]]
    death_data_analysis["Total_Cumulative_Estimated_Infected"] = estimated_infected_total[:, - int(round(estimated_time_to_die)) - 1]
    death_data_analysis["Total_Cumulative_Confirmed_Infected"] = numeric_confirmed_data[
        numeric_confirmed_data.columns[-1]]
    death_data_analysis["Total_Cumulative_Confirmed_Recovered"] = numeric_recovered_data[
        numeric_recovered_data.columns[-1]]

    death_data_analysis["TODAY_New_Death"] = numeric_death_data[numeric_death_data.columns[-1]] - numeric_death_data[
        numeric_death_data.columns[-2]]

    death_data_analysis["TODAY_New_Estimated_Infected"] = estimated_infected_total[:,
                                                          - int(round(estimated_time_to_die)) - 1] - estimated_infected_total[:,
                                                                                        - int(round(estimated_time_to_die)) - 2]

    death_data_analysis["TODAY_New_Confirmed_Infected"] = numeric_confirmed_data[numeric_confirmed_data.columns[-1]] - \
                                                          numeric_confirmed_data[numeric_confirmed_data.columns[-2]]
    death_data_analysis["TODAY_New_Confirmed_Recovered"] = numeric_recovered_data[numeric_recovered_data.columns[-1]] - \
                                                           numeric_recovered_data[numeric_recovered_data.columns[-2]]

    US_death_data_analysis = US_death_data.iloc[:, :4]

    US_death_data_analysis["Total_Cumulative_Confirmed_Death"] = numeric_US_death_data[
        numeric_US_death_data.columns[-1]]
    US_death_data_analysis["Total_Cumulative_Estimated_Infected"] = US_estimated_infected_total[:,
                                                                    - int(round(estimated_time_to_die)) - 1]

    US_death_data_analysis["TODAY_New_Death"] = numeric_US_death_data[numeric_US_death_data.columns[-1]] - \
                                                numeric_US_death_data[numeric_US_death_data.columns[-2]]

    US_death_data_analysis["TODAY_New_Estimated_Infected"] = US_estimated_infected_total[:,
                                                             - int(round(estimated_time_to_die)) - 1] - US_estimated_infected_total[
                                                                                           :,
                                                                                           - int(round(estimated_time_to_die)) - 2]

    Cal_death_data_analysis = Cal_death_data.iloc[:, :4]

    Cal_death_data_analysis["Total_Cumulative_Confirmed_Death"] = numeric_Cal_death_data[
        numeric_Cal_death_data.columns[-1]]
    Cal_death_data_analysis["Total_Cumulative_Estimated_Infected"] = Cal_estimated_infected_total[:,
                                                                     - int(round(estimated_time_to_die)) - 1]

    Cal_death_data_analysis["TODAY_New_Death"] = numeric_Cal_death_data[numeric_Cal_death_data.columns[-1]] - \
                                                 numeric_Cal_death_data[numeric_Cal_death_data.columns[-2]]

    Cal_death_data_analysis["TODAY_New_Estimated_Infected"] = Cal_estimated_infected_total[:,
                                                              - int(round(estimated_time_to_die)) - 1] - Cal_estimated_infected_total[
                                                                                            :,
                                                                                            - int(round(estimated_time_to_die)) - 2]

    US_index_location = death_data_analysis.loc[death_data['Country/Region'] == "US"].index
    US_data = death_data_analysis.iloc[US_index_location]

    WW_ranking = death_data_country.iloc[:, -1].sort_values(ascending=False)
    WW_top_nine = WW_ranking[:9].index

    # Get 9 biggest states
    US_ranking = US_death_data_state.iloc[:, -1].sort_values(ascending=False)
    US_top_nine = US_ranking[:9].index

    # Get 9 biggest countys
    Cal_ranking = Cal_death_data_county.iloc[:, -1].sort_values(ascending=False)
    Cal_top_nine = Cal_ranking[:9].index

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # DATA Output / Plot
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    plt.clf()
    plt.close()
    plt.close('all')

    ##########################################################################################################
    # HARDCODED the countries
    ##########################################################################################################

    # Country: (populations, color)
    world = {'US': [330537265, 111],
             'Germany': [83719907, 222],
             'Indonesia': [272826336, 333],
             'Netherlands': [17134872, 444],
             'United Kingdom': [67886011, 555],
             'Israel': [8655535, 666],
             'Italy': [60461826, 777],
             'Spain': [46754778, 888],
             'China': [1437994234, 999],
             'Singapore': [5850342, 000],
             'Korea, South': [51269185, 111222],
             }

    ##########################################################################################################
    ##########################################################################################################
    # UTILITY functions for data output
    ##########################################################################################################
    ##########################################################################################################

    def first_death(arr):
        a = pd.DataFrame(arr)
        b = (a > 0).idxmax(axis=0)[0]
        c = a.iloc[b:, :]
        d = c.T
        # d.index = range(len(d.index))
        d.columns = range(len(d.columns))
        return d

    def first_value(nparray):
        return nparray[np.argmax(nparray > 0):]

    def last_value(df):
        col = df.columns
        ende = df.iloc[-1, :]
        for each_column in range(len(df.columns)):
            a = col[each_column]
            b = ende[each_column]
            c = df[col[each_column]] == ende[each_column]
            d = df.loc[df[col[each_column]] == ende[each_column]]
            e = df.loc[df[col[each_column]] == ende[each_column]].iloc[1:, each_column]
            df.iloc[d.index[1:], each_column] = np.NaN

    def last_value_index(df):
        last_value_idx = []
        for row in range(len(df)):

            last_element = df.iloc[row, -1]

            if last_element != df.iloc[row, -2]:
                last_value_idx.append(df.iloc[row, :].index[-1])

            else:

                for idx in reversed(df.iloc[row, :].index):
                    element = df.iloc[row, idx]
                    if element != last_element:
                        last_value_idx.append(idx)
                        break  # loop over element in row

        return last_value_idx

    def gridplot(df, hline=False):

        # Initialize the figure
        plt.xticks([], [])
        plt.yticks([], [])

        # create a color palette
        palette = plt.get_cmap('tab20')

        # multiple line plot
        num = -1
        for column in df:  # .iloc[:,:-2]:
            num += 1

            plt.style.use('seaborn-darkgrid')

            # plt.subplot(4, 3, num)
            ax = fig.add_subplot(inner_grid[num])

            plot_data = df[column]

            # plot_data = df[column].ewm(span=5,adjust=False).mean()
            farbe = palette(num)

            if num == 6 and hline: farbe = "#900C3F"
            plt.plot(plot_data, marker='o', color=farbe, linewidth=1, alpha=0.9, label=column)
            plt.title(column, color=palette(num))  # fontsize=60, fontweight=0,

            # Same limits for everybody!
            plt.xlim(0, 60)  # len(df))

            # Not ticks everywhere
            if num in range(4, 7, 11):
                plt.tick_params(labelbottom='off')

            if hline:
                ypoints = 0
                plt.axhline(ypoints, 0, 60, color=palette(6))
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    # COLOR palette
    palette = 'tab20'

    ##################################################################################################################
    ##################################################################################################################
    ## WORLDWIDE
    ##################################################################################################################
    ##################################################################################################################

    #########################################################################
    ## Worldwide TOTAL Death (Continental / Subtropics)
    #########################################################################

    death_grows = pd.DataFrame()

    labels = []

    for country in world:
        location = death_data_country.index.get_loc(country)
        population = world[country][0]
        labels.append(country)
        data = numeric_death_data_daily_country[location]
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        death_grows = pd.concat([death_grows, death_row], ignore_index=True)

    data = numeric_death_data_daily_continental
    daily_death = first_death(data)
    death_row = daily_death
    death_row.columns = range(len(death_row.columns))
    death_grows = pd.concat([death_grows, death_row], ignore_index=True)
    labels.append('Continental')

    data = numeric_death_data_daily_subtropical
    daily_death = first_death(data)
    death_row = daily_death
    death_row = death_row.iloc[:, 40:]
    death_row.columns = range(len(death_row.columns))
    death_grows = pd.concat([death_grows, death_row], ignore_index=True)
    labels.append('Subtropical')

    death_grows.fillna(0, inplace=True)
    death_grows.iloc[:, 1:] = death_grows.iloc[:, 1:].cumsum(axis=1)

    last = death_grows.iloc[:, 1:].values.argmax(axis=1)
    for end in range(len(last)):
        death_grows.iloc[end, last[end] + 1:] = np.NaN

    # of DEATH grows
    # Normalized ... starting all with their FIRST case, rather than a certain DATE

    a = death_grows.T
    a.columns = labels

    # DIN A4
    fig, ax = plt.subplots(figsize=(11.69 * 3, 8.27 * 2))  # DIN A4

    ax.axis('off')

    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4})
    sns.set_style("white")
    plt.xticks([], [])
    plt.yticks([], [])
    ax.tick_params(labelsize=14)

    fig.text(**fig_text)

    outer_grid = fig.add_gridspec(2, 3, left=0.03, top=0.95, bottom=0.05, hspace=0.25)  # , wspace=0.0, hspace=0.0)

    ##########################################################################################################
    b = a.iloc[:, -2:]

    ax = fig.add_subplot(outer_grid[2])
    plt.style.use('seaborn-darkgrid')
    sns.lineplot(data=b, dashes=False, palette=['blue', 'red'])
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL Death", fontsize=20)
    plt.title("Looks like the CONTINENT is a few weeks ahead of the SUBTROPICS", fontsize=20)  # , y=1.02)

    #########################################################################
    ## Worldwide TOTAL Death per 1M of population (NORMALIZED)
    #########################################################################

    death_grows = pd.DataFrame()

    labels = []

    for country in world:
        country_name = world[country][1]
        location = death_data_country.index.get_loc(country)
        population = world[country][0]
        pop_per_mil = population // 1000000
        labels.append(country)
        data = np.array(numeric_death_data_country)[location]

        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)

        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        death_grows = pd.concat([death_grows, death_row], ignore_index=True)

    death_grows.fillna(0, inplace=True)

    last = last_value_index(death_grows)
    for end in range(len(last)):
        death_grows.iloc[end, last[end] + 1:] = np.NaN

    # of DEATH grows
    # Normalized ... starting all with their FIRST case, rather than a certain DATE

    a = death_grows.T
    a.columns = labels


    b = a.iloc[:60, :]  # .iloc[:,:-2]

    # Combined PLOT of all the chosen countries
    ax = fig.add_subplot(outer_grid[0])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL death per 1M of population", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of death per 1M pop", fontsize=20)

    plt.style.use('seaborn-white')

    ## Start GRID Plot ... each country separate
    ax = fig.add_subplot(outer_grid[1])

    plt.axis("off")

    plt.style.use('seaborn-darkgrid')

    plt.title("Apples 2 Apples comparison of # of death per 1M pop", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',

    inner_grid = outer_grid[1].subgridspec(4, 3, wspace=0.2, hspace=0.5)

    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    #########################################################################
    ## Worldwide DAILY Death per 1M of population (NORMALIZED)
    #########################################################################

    death_grows = pd.DataFrame()
    plot_data_df = pd.DataFrame()

    labels = []

    for country in world:
        location = death_data_country.index.get_loc(country)
        population = world[country][0]
        pop_per_mil = population // 1000000
        labels.append(country)
        data = np.array(numeric_death_data_daily_country)[location]
        # data = data / pop_per_mil
        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)
        daily_death = first_death(data)

        plot_data_row = first_value(World_grows_rate[location])
        outlier = first_value(World_grows_rate[location])[3:]
        # outlier[np.isinf(outlier)]=np.nan
        outlier = np.nanmean(outlier) + (np.nanstd(outlier) * 2)
        plot_data_row[plot_data_row > outlier] = np.NaN
        # plot_data_row[0] = world[country][1]
        plot_data_row = pd.DataFrame(plot_data_row).T
        plot_data_row.columns = range(len(plot_data_row.columns))
        plot_data_df = pd.concat([plot_data_df, plot_data_row], ignore_index=True)

        death_row = daily_death
        # death_row[0] = world[country][1]
        death_row.columns = range(len(death_row.columns))
        death_grows = pd.concat([death_grows, death_row], ignore_index=True)

    death_grows.fillna(0, inplace=True)

    last = last_value_index(death_grows)
    for end in range(len(last)):
        death_grows.iloc[end, last[end] + 1:] = np.NaN

    plot_data_df = plot_data_df.fillna(method='ffill', axis=1)

    # of DEATH grows
    # Normalized ... starting all with their FIRST case, rather than a certain DATE

    a = death_grows.T
    a.columns = labels

    b = a.ewm(span=5, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    b = b.iloc[:60, :]

    ax = fig.add_subplot(outer_grid[3])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("DAILY Death #ers", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of DAILY death per 1M pop", fontsize=20)

    plt.style.use('seaborn-white')

    ax = fig.add_subplot(outer_grid[4])
    plt.axis("off")
    plt.style.use('seaborn-darkgrid')
    plt.title("Apples 2 Apples comparison of # of DAILY death per 1M pop", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',
    inner_grid = outer_grid[4].subgridspec(4, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    a = plot_data_df.T
    a.columns = labels

    b = a.ewm(span=3, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    plt.style.use('seaborn-white')

    ax = fig.add_subplot(outer_grid[5])
    plt.axis("off")

    plt.title("Apples 2 Apples comparison DAILY Growth Rate", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',

    inner_grid = outer_grid[5].subgridspec(4, 3, wspace=0.2, hspace=0.5)

    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b, True)


    pdf.savefig(dpi=fig.dpi)

    #########################################################################
    ## Worldwide Output the DATA / close plot
    #########################################################################

    plt.clf()
    plt.close()
    plt.close('all')

    ##################################################################################################################
    ##################################################################################################################
    ## USA
    ##################################################################################################################
    ##################################################################################################################


    #########################################################################
    ## USA TOTAL Death (EAST / WEST)
    #########################################################################

    US_death_grows = pd.DataFrame()

    labels = []

    for state in US_top_nine:
        location = US_death_data_state.index.get_loc(state)
        population = US_population_state[state]
        labels.append(state)
        data = numeric_US_death_data_daily_state[location]
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)

    data = numeric_US_death_data_daily_east
    daily_death = first_death(data)
    death_row = daily_death
    death_row.columns = range(len(death_row.columns))
    US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)
    labels.append('US East')

    data = numeric_US_death_data_daily_west
    daily_death = first_death(data)
    death_row = daily_death
    # death_row = death_row.iloc[:, 60:]
    death_row.columns = range(len(death_row.columns))
    US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)
    labels.append('US West')

    US_death_grows.fillna(0, inplace=True)
    US_death_grows.iloc[:, 1:] = US_death_grows.iloc[:, 1:].cumsum(axis=1)

    last = last_value_index(US_death_grows)
    for end in range(len(last)):
        US_death_grows.iloc[end, last[end] + 1:] = np.NaN

    # of DEATH grows
    # Normalized ... starting all with their FIRST case, rather than a certain DATE

    a = US_death_grows.T
    a.columns = labels

    # DIN A4
    fig, ax = plt.subplots(figsize=(11.69 * 3, 8.27 * 2))  # DIN A4
    ax.axis('off')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4})
    sns.set_style("white")
    plt.xticks([], [])
    plt.yticks([], [])
    ax.tick_params(labelsize=14)

    fig.text(**fig_text)

    outer_grid = fig.add_gridspec(2, 3, left=0.03, top=0.95, bottom=0.05, hspace=0.25)  # , wspace=0.0, hspace=0.0)

    b = a.iloc[:, -2:]

    ########################################################################################
    ax = fig.add_subplot(outer_grid[2])
    plt.style.use('seaborn-darkgrid')
    sns.lineplot(data=b, dashes=False, palette=['orange', 'green'])
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL Death", fontsize=20)
    plt.title("Looks like the EAST is a little ahead", fontsize=20)  # , y=1.02)

    #########################################################################
    ## USA TOTAL Death per 1M of population (NORMALIZED)
    #########################################################################

    US_death_grows = pd.DataFrame()

    labels = []

    for state in US_top_nine:
        location = US_death_data_state.index.get_loc(state)
        population = US_population_state[state]
        pop_per_mil = population // 1000000
        labels.append(state)
        data = np.array(numeric_US_death_data_state)[location]
        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)

    US_death_grows.fillna(0, inplace=True)

    last = last_value_index(US_death_grows)
    for end in range(len(last)):
        US_death_grows.iloc[end, last[end] + 1:] = np.NaN


    a = US_death_grows.T
    a.columns = labels

    b = a  # .iloc[:,:-2]

    ########################################################################################
    ax = fig.add_subplot(outer_grid[0])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL death per 1M of population", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of death per 1M pop", fontsize=20)
    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[1])
    plt.axis("off")
    plt.style.use('seaborn-darkgrid')

    plt.title("Apples 2 Apples comparison of # of death per 1M pop", fontsize=20, pad=multiplot_padding)
    inner_grid = outer_grid[1].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    #########################################################################
    ## USA DAILY Death per 1M of population (NORMALIZED)
    #########################################################################

    US_death_grows = pd.DataFrame()

    labels = []

    for state in US_top_nine:
        location = US_death_data_state.index.get_loc(state)
        population = US_population_state[state]
        labels.append(state)
        data = numeric_US_death_data_daily_state[location]
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)

    data = numeric_US_death_data_daily_east
    daily_death = first_death(data)
    death_row = daily_death
    death_row.columns = range(len(death_row.columns))
    US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)
    labels.append('US East')

    ###  Maybe ADD US Mid

    data = numeric_US_death_data_daily_west
    daily_death = first_death(data)
    death_row = daily_death
    # death_row = death_row.iloc[:, 60:]
    death_row.columns = range(len(death_row.columns))
    US_death_grows = pd.concat([US_death_grows, death_row], ignore_index=True)
    labels.append('US West')

    US_death_grows.fillna(0, inplace=True)

    last = last_value_index(US_death_grows)

    for end in range(len(last)):
        US_death_grows.iloc[end, last[end] + 1:] = np.NaN

    a = US_death_grows.T
    a.columns = labels

    b = a.iloc[:, :-2]

    #########################################################################
    ## USA Daily Death per 1M of population (NORMALIZED)
    #########################################################################

    US_death_grows = pd.DataFrame()
    US_plot_data_df = pd.DataFrame()

    labels = []

    for state in US_top_nine:
        location = US_death_data_state.index.get_loc(state)
        population = US_population_state[state]
        pop_per_mil = population // 1000000
        labels.append(state)
        data = np.array(numeric_US_death_data_daily_state)[location]
        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)
        daily_death = first_death(data)

        US_plot_data_row = first_value(US_grows_rate[location])
        outlier = first_value(US_grows_rate[location])[3:]
        outlier = np.nanmean(outlier) + (np.nanstd(outlier) * 2)
        US_plot_data_row[US_plot_data_row > outlier] = np.NaN
        US_plot_data_row = pd.DataFrame(US_plot_data_row).T
        US_plot_data_row.columns = range(len(US_plot_data_row.columns))
        US_plot_data_df = pd.concat([US_plot_data_df, US_plot_data_row], ignore_index=True)

        US_death_row = daily_death
        US_death_row.columns = range(len(US_death_row.columns))
        US_death_grows = pd.concat([US_death_grows, US_death_row], ignore_index=True)

    US_death_grows.fillna(0, inplace=True)

    last = last_value_index(US_death_grows)
    for end in range(len(last)):
        US_death_grows.iloc[end, last[end] + 1:] = np.NaN

    US_plot_data_df.fillna(method='ffill', axis=1, inplace=True)

    # of DEATH grows
    # Normalized ... starting all with their FIRST case, rather than a certain DATE

    a = US_death_grows.T
    a.columns = labels

    b = a.ewm(span=5, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    ########################################################################################
    ax = fig.add_subplot(outer_grid[3])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("DAILY Death #ers", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of DAILY death per 1M pop", fontsize=20)

    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[4])
    plt.axis("off")
    plt.style.use('seaborn-darkgrid')

    plt.title("Apples 2 Apples comparison of # of DAILY death per 1M pop", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',
    inner_grid = outer_grid[4].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    a = US_plot_data_df.T
    a.columns = labels

    # last_value(b)
    b = a.ewm(span=3, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[5])
    plt.axis("off")
    plt.title("Apples 2 Apples comparison DAILY Growth Rate", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',
    # style='italic')  # ,1
    inner_grid = outer_grid[5].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b, True)

    pdf.savefig(dpi=fig.dpi)

    #########################################################################
    ## Output the DATA
    #########################################################################

    plt.clf()
    plt.close()
    plt.close('all')

    ##################################################################################################################
    ##################################################################################################################
    ## CALIFORNIA
    ##################################################################################################################
    ##################################################################################################################

    #########################################################################
    ## Worldwide TOTAL Death (NORTH / SOUTH)
    #########################################################################

    Cal_death_grows = pd.DataFrame()

    labels = []

    for county in Cal_top_nine:
        location = Cal_death_data_county.index.get_loc(county)
        population = Cal_population_county[county]
        labels.append(county)
        data = numeric_Cal_death_data_daily_county[location]
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        Cal_death_grows = pd.concat([Cal_death_grows, death_row], ignore_index=True)

    data = numeric_Cal_death_data_daily_north
    daily_death = first_death(data)
    death_row = daily_death
    death_row.columns = range(len(death_row.columns))
    Cal_death_grows = pd.concat([Cal_death_grows, death_row], ignore_index=True)
    labels.append('Northern California')

    data = numeric_Cal_death_data_daily_south
    daily_death = first_death(data)
    death_row = daily_death
    # death_row = death_row.iloc[:,40:]
    death_row.columns = range(len(death_row.columns))
    Cal_death_grows = pd.concat([Cal_death_grows, death_row], ignore_index=True)
    labels.append('Southern California')

    Cal_death_grows.fillna(0, inplace=True)
    Cal_death_grows.iloc[:, 1:] = Cal_death_grows.iloc[:, 1:].cumsum(axis=1)

    last = last_value_index(Cal_death_grows)
    for end in range(len(last)):
        Cal_death_grows.iloc[end, last[end] + 1:] = np.NaN

    a = Cal_death_grows.T
    a.columns = labels

    # DIN A4
    fig, ax = plt.subplots(figsize=(11.69 * 3, 8.27 * 2))  # DIN A4
    ax.axis('off')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4})
    sns.set_style("white")
    plt.xticks([], [])
    plt.yticks([], [])
    ax.tick_params(labelsize=14)

    fig.text(**fig_text)

    outer_grid = fig.add_gridspec(2, 3, left=0.03, top=0.95, bottom=0.05, hspace=0.25)  # , wspace=0.0, hspace=0.0)

    b = a.iloc[:, -2:]

    ########################################################################################
    ax = fig.add_subplot(outer_grid[2])
    plt.style.use('seaborn-darkgrid')
    sns.lineplot(data=b, dashes=False, palette=['blue', 'red'])
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL Death", fontsize=20)
    plt.title("Looks like the Northern California is doing a better job !", fontsize=20)  # , y=1.02)

    #########################################################################
    ## California TOTAL Death per 100,000 of population (NORMALIZED)
    #########################################################################

    Cal_death_grows = pd.DataFrame()

    labels = []

    for county in Cal_top_nine:
        location = Cal_death_data_county.index.get_loc(county)
        population = Cal_population_county[county]
        pop_per_mil = population // 100000
        labels.append(county)
        data = np.array(numeric_Cal_death_data_county)[location]
        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)
        daily_death = first_death(data)
        death_row = daily_death
        death_row.columns = range(len(death_row.columns))
        Cal_death_grows = pd.concat([Cal_death_grows, death_row], ignore_index=True)

    a = Cal_death_grows.T
    a.columns = labels

    b = a  # .iloc[:,:-2]

    ########################################################################################
    ax = fig.add_subplot(outer_grid[0])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("TOTAL death per 1M of population", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of death per 100,000 pop", fontsize=20)
    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[1])
    plt.axis("off")
    plt.style.use('seaborn-darkgrid')

    plt.title("Apples 2 Apples comparison of # of death per 100,000 pop", fontsize=20, pad=multiplot_padding)
    inner_grid = outer_grid[1].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    #########################################################################
    ## California DAILY Death per 100,000 of population (NORMALIZED)
    #########################################################################

    Cal_death_grows = pd.DataFrame()
    Cal_plot_data_df = pd.DataFrame()

    labels = []

    for county in Cal_top_nine:
        location = Cal_death_data_county.index.get_loc(county)
        population = Cal_population_county[county]
        pop_per_mil = population // 100000
        labels.append(county)
        data = np.array(numeric_Cal_death_data_daily_county)[location]
        # data = data / pop_per_mil
        data = np.where(pop_per_mil == 0, pop_per_mil, data / pop_per_mil)
        daily_death = first_death(data)

        Cal_plot_data_row = first_value(Cal_grows_rate[location])
        outlier = first_value(Cal_grows_rate[location])[3:]
        # outlier[np.isinf(outlier)]=np.nan
        outlier = np.nanmean(outlier) + (np.nanstd(outlier) * 2)
        Cal_plot_data_row[Cal_plot_data_row > outlier] = np.NaN
        # Cal_plot_data_row[0] = world[country][1]
        Cal_plot_data_row = pd.DataFrame(Cal_plot_data_row).T
        Cal_plot_data_row.columns = range(len(Cal_plot_data_row.columns))
        Cal_plot_data_df = pd.concat([Cal_plot_data_df, Cal_plot_data_row], ignore_index=True)

        Cal_death_row = daily_death
        Cal_death_row.columns = range(len(Cal_death_row.columns))
        Cal_death_grows = pd.concat([Cal_death_grows, Cal_death_row], ignore_index=True)

    Cal_death_grows.fillna(0, inplace=True)
    # death_grows.iloc[:,1:]  = death_grows.iloc[:,1:].cumsum(axis=1)

    last = last_value_index(Cal_death_grows)
    for end in range(len(last)):
        Cal_death_grows.iloc[end, last[end] + 1:] = np.NaN

    Cal_plot_data_df.fillna(method='ffill', axis=1, inplace=True)

    a = Cal_death_grows.T
    a.columns = labels

    b = a.ewm(span=7, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    ########################################################################################
    ax = fig.add_subplot(outer_grid[3])
    plt.style.use('seaborn-white')
    sns.lineplot(data=b, dashes=False, palette=palette)
    sns.despine()

    plt.legend(b.columns, loc='upper left', frameon=False)
    plt.xlabel("Days since FIRST death", fontsize=20)
    plt.ylabel("DAILY Death #ers", fontsize=20)
    plt.title("Apples 2 Apples comparison of # of DAILY death per 100,000 pop", fontsize=20)

    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[4])
    plt.axis("off")
    plt.style.use('seaborn-darkgrid')

    plt.title("Apples 2 Apples comparison of # of DAILY death per 100,000 pop", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',

    inner_grid = outer_grid[4].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])

    gridplot(b)

    a = Cal_plot_data_df.T
    a.columns = labels

    b = a.ewm(span=5, adjust=False, ignore_na=True).mean()

    for end in range(len(last)):
        b.iloc[last[end] + 1:, end] = np.NaN

    plt.style.use('seaborn-white')

    ########################################################################################
    ax = fig.add_subplot(outer_grid[5])
    plt.axis("off")
    plt.title("Apples 2 Apples comparison of # of DAILY growth rate", fontsize=20,
              pad=multiplot_padding)  # , fontweight=0, color='black',

    inner_grid = outer_grid[5].subgridspec(3, 3, wspace=0.2, hspace=0.5)
    plt.xticks([], [])
    plt.yticks([], [])
    redline = True
    gridplot(b, redline)

    pdf.savefig(dpi=fig.dpi)

    # remember to close the object to ensure writing multiple plots
    pdf.close()

    #########################################################################
    ## California Output the DATA / close plot
    #########################################################################

    plt.clf()
    plt.close()
    plt.close('all')

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # Prepare the REPORT
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    ##################################################
    # put in all the variables into the HTML cover text file
    ##################################################
    def growth_rate_string(growths_rate):
        percentage = growths_rate
        if percentage >= 10:
            # include RED color
            string = '<span style="color: #ff0000;">' + f'{percentage:.2f}' + "%</span>"
        else:
            # normal (black)
            # string = f'{growths_rate:.2f}'
            string = f'{percentage:.2f}' + "%"
        return string

    HTML_Cover_Text = HTML_Cover_Text.replace("99 mmm, yyyy", datetime.datetime.now().strftime("%d %b %Y"))
    HTML_Cover_Text = HTML_Cover_Text.replace("---emr", f'{(estimated_mortality_rate) * 100:.0f}' + " %")
    HTML_Cover_Text = HTML_Cover_Text.replace("---ettd", estimated_time_to_die.__str__())
    HTML_Cover_Text = HTML_Cover_Text.replace("---edtd", estimated_days_to_double.__str__())

    HTML_Cover_Text = HTML_Cover_Text.replace("---rankus", (WW_ranking.index.get_loc('US') + 1).__str__())
    HTML_Cover_Text = HTML_Cover_Text.replace("---rankcal", (US_ranking.index.get_loc('California') + 1).__str__())
    HTML_Cover_Text = HTML_Cover_Text.replace("---ranksc", (Cal_ranking.index.get_loc('Santa Clara') + 1).__str__())

    HTML_Cover_Text = HTML_Cover_Text.replace("88 mmm, yyyy", numeric_death_data_country.columns[-1].strftime("%d %b %Y"))
    for country in range(len(world)):
        temp = death_data_analysis.loc[death_data_analysis['Country/Region'] == list(world.keys())[country]].iloc[:,
               4:].sum().round().astype('int64')

        ranking_number = list(WW_ranking.index).index(list(world.keys())[country]) + 1

        text2switch = """  <p class=MsoNormal align=center style='text-align:center'>---ww""" + country.__str__() + """<b><span\n  style='mso-fareast-font-family:"Times New Roman"'><o:p></o:p></span></b></p>"""

        rankingtext = """
<p class=MsoNormal align=center style='text-align:center'>""" + str(ranking_number) + """<b><span
style='mso-fareast-font-family:"Times New Roman"'><o:p></o:p></span></b></p>
        """
        top_ranking = """
<h3 align=center style='text-align:center'><strong><span style='mso-fareast-font-family:
"Times New Roman";color:red'>""" + str(ranking_number) + """</span></strong><span style='mso-fareast-font-family:
"Times New Roman"'><o:p></o:p></span></h3>"""

        if ranking_number <= 3: ranking = top_ranking
        else: ranking = rankingtext

        HTML_Cover_Text = HTML_Cover_Text.replace(text2switch, ranking)
        HTML_Cover_Text = HTML_Cover_Text.replace("---wr" + country.__str__(), list(world.keys())[country])
        HTML_Cover_Text = HTML_Cover_Text.replace("---wtd" + country.__str__(), f'{temp[0]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---wnd" + country.__str__(), f'{temp[4]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---wei" + country.__str__(), f'{temp[1]:,}')

        growths_rate = growth_rate_string(
            World_grows_rate[death_data_country.index.get_loc(list(world.keys())[country])][-1])

        # HTML_Cover_Text = HTML_Cover_Text.replace("---wgr" + country.__str__(), f'{growths_rate:.2f}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---wgr" + country.__str__(), growths_rate)

    HTML_Cover_Text = HTML_Cover_Text.replace("77 mmm, yyyy", numeric_US_death_data_state.columns[-1].strftime("%d %b %Y"))
    for state in range(len(US_top_nine)):
        temp = US_death_data_analysis.loc[US_death_data_analysis['Province_State'] == US_top_nine[state]].iloc[:,
               4:].sum().round().astype('int64')

        HTML_Cover_Text = HTML_Cover_Text.replace("---ar" + state.__str__(), US_top_nine[state])
        HTML_Cover_Text = HTML_Cover_Text.replace("---atd" + state.__str__(), f'{temp[0]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---and" + state.__str__(), f'{temp[2]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---aei" + state.__str__(), f'{temp[1]:,}')

        growths_rate = growth_rate_string(US_grows_rate[US_death_data_state.index.get_loc(US_top_nine[state])][-1])

        # HTML_Cover_Text = HTML_Cover_Text.replace("---agr" + state.__str__(), f'{growths_rate:.2f}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---agr" + state.__str__(), growths_rate)

    HTML_Cover_Text = HTML_Cover_Text.replace("66 mmm, yyyy", numeric_Cal_death_data_county.columns[-1].strftime("%d %b %Y"))
    for county in range(len(Cal_top_nine)):
        temp = Cal_death_data_analysis.loc[Cal_death_data_analysis['County'] == Cal_top_nine[county]].iloc[:, 4:].sum().round().astype('int64')

        HTML_Cover_Text = HTML_Cover_Text.replace("---cr" + county.__str__(), Cal_top_nine[county])
        HTML_Cover_Text = HTML_Cover_Text.replace("---ctd" + county.__str__(), f'{temp[0]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---cnd" + county.__str__(), f'{temp[2]:,}')
        HTML_Cover_Text = HTML_Cover_Text.replace("---cei" + county.__str__(), f'{temp[1]:,}')

        growths_rate = growth_rate_string(Cal_grows_rate[Cal_death_data_county.index.get_loc(Cal_top_nine[county])][-1])

        HTML_Cover_Text = HTML_Cover_Text.replace("---cgr" + county.__str__(), growths_rate)

    # The HTML Cover file has been filled in with all the data !  ==> time to save
    html_cover_file = temp_data_root / (str(datetime.date.today()) + ".html")
    with open(html_cover_file, "w") as text_file:
        text_file.write(HTML_Cover_Text)

    ##########################################################################################################
    # USING WK <html> TO pdf: https://wkhtmltopdf.org/
    # convert HTML cover text to PDF
    ##########################################################################################################

    input_file = html_cover_file
    output_file = temp_cover_pdf

    input_file_enc = chr(34) + str(input_file.absolute()) + chr(34)
    output_file_enc = chr(34) + str(output_file.absolute()) + chr(34)

    converter_loc = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe "
    options = " --dpi 900 -s Letter "

    converter_commando = chr(34) + converter_loc + chr(34) + ' '  + options + ' ' + input_file_enc + ' ' + output_file_enc
    subprocess.run(converter_commando, shell=True)

    ##########################################################################################################
    # Merge the TWO pdf files (COVER & FIGURES) ==> into ONE REPORT file
    ##########################################################################################################

    figure_pdf = pikepdf.Pdf.open(temp_figures_pdf)

    cover_pdf = pikepdf.Pdf.open(temp_cover_pdf)
    cover_pdf.pages.extend(figure_pdf.pages)
    Cover_opt = "linearize (True)"
    cover_pdf.save(report_loc)

    cover_pdf.close()
    figure_pdf.close()

    # remove all TEMPORARY data and directory
    os.system('rmdir /s/Q "' + str(temp_data_root.absolute()) + '"')

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    # SEND the report via GMAIL ...
    ##########################################################################################################
    # USING Gmail API & OAuth:
    # https://stackoverflow.com/questions/37201250/sending-email-via-gmail-python
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################

    SCOPES = 'https://www.googleapis.com/auth/gmail.send'
    CLIENT_SECRET_FILE = '../Data/client_secret_abcdefghijklmnop123456789.apps.googleusercontent.com.json'
    APPLICATION_NAME = 'Gmail API Python Send Email'

    def get_credentials():
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir, 'gmail-python-email-send.json')
        store = oauth2client.file.Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
            flow.user_agent = APPLICATION_NAME
            credentials = tools.run_flow(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials

    def SendMessage(sender, to, cc, bcc, subject, msgHtml, msgPlain, attachmentFile=None):
        credentials = get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('gmail', 'v1', http=http)
        if attachmentFile:
            message1 = createMessageWithAttachment(sender, to, cc, bcc, subject, msgHtml, msgPlain, attachmentFile)
        else:
            message1 = CreateMessageHtml(sender, to, cc, bcc, subject, msgHtml, msgPlain)
        result = SendMessageInternal(service, "me", message1)
        return result

    def SendMessageInternal(service, user_id, message):
        try:
            message = (service.users().messages().send(userId=user_id, body=message).execute())
            print('Message Id: %s' % message['id'])
            return message
        except errors.HttpError as error:
            print('An error occurred: %s' % error)
            return "Error"

    def CreateMessageHtml(sender, to, cc, bcc, subject, msgHtml, msgPlain):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to
        msg['cc'] = cc
        msg['bcc'] = bcc
        msg.attach(MIMEText(msgPlain, 'plain'))
        msg.attach(MIMEText(msgHtml, 'html'))
        # return {'raw': base64.urlsafe_b64encode(msg.as_string())}
        return {'raw': base64.urlsafe_b64encode(msg.as_string().encode()).decode()}

    def createMessageWithAttachment(
            sender, to, cc, bcc, subject, msgHtml, msgPlain, attachmentFile):
        """Create a message for an email.

		Args:
		  sender: Email address of the sender.
		  to: Email address of the receiver.
		  subject: The subject of the email message.
		  msgHtml: Html message to be sent
		  msgPlain: Alternative plain text message for older email clients
		  attachmentFile: The path to the file to be attached.

		Returns:
		  An object containing a base64url encoded email object.
		"""
        message = MIMEMultipart('mixed')
        message['to'] = to
        message['cc'] = cc
        message['bcc'] = bcc
        message['from'] = sender
        message['subject'] = subject

        messageA = MIMEMultipart('alternative')
        messageR = MIMEMultipart('related')

        messageR.attach(MIMEText(msgHtml, 'html'))
        messageA.attach(MIMEText(msgPlain, 'plain'))
        messageA.attach(messageR)

        message.attach(messageA)

        print("create_message_with_attachment: file: %s" % attachmentFile)
        content_type, encoding = mimetypes.guess_type(attachmentFile)

        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        main_type, sub_type = content_type.split('/', 1)
        if main_type == 'text':
            fp = open(attachmentFile, 'rb')
            msg = MIMEText(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'image':
            fp = open(attachmentFile, 'rb')
            msg = MIMEImage(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'audio':
            fp = open(attachmentFile, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
            fp.close()
        else:
            fp = open(attachmentFile, 'rb')
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
            fp.close()
        filename = os.path.basename(attachmentFile)
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        email.encoders.encode_base64(msg)
        message.attach(msg)

        return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}

    email_body_HTML = HTML_Cover_Text
    # SendMessage(sender, to, subject, msgHtml, msgPlain)
    # Send message with attachment:
    SendMessage(email_sender, email_to, email_cc, email_bcc, email_subject, email_body_HTML, email_body_plain,
                report_loc)

# MAIN procedure
Corona_App()
