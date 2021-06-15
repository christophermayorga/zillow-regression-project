import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

os.path.isfile('telco_df.csv')


# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    

# Use the above helper function and a sql query in a single function.
def new_zillow_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    telco_sql = "SELECT parcelid, propertylandusetypeid, propertylandusedesc, \
                 transactiondate, calculatedfinishedsquarefeet, bedroomcnt,\
                 bathroomcnt, buildingqualitytypeid, fips, regionidzip, yearbuilt, taxvaluedollarcnt,\
                 assessmentyear, taxamount \
                 FROM predictions_2017 \
                 JOIN properties_2017 using (parcelid) \
                 JOIN propertylandusetype using (propertylandusetypeid) \
                 WHERE month(transactiondate) >= 05 and month(transactiondate) <= 08 \
                 ORDER BY buildingqualitytypeid;" \
    
    
    return pd.read_sql(telco_sql, get_connection('zillow'))



def get_zillow_data(cached=False):
    '''
    This function reads in telco churn data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    return df