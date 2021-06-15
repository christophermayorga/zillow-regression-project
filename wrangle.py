import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from datetime import date 
from scipy import stats

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

# modeling methods
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE 
import sklearn.preprocessing


os.path.isfile('zillow_df.csv')


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
    zillow_sql = "SELECT parcelid, propertylandusetypeid, propertylandusedesc, unitcnt, \
                 transactiondate, calculatedfinishedsquarefeet, bedroomcnt, \
                 bathroomcnt, fips, regionidzip, yearbuilt, taxvaluedollarcnt, latitude, longitude,  \
                 assessmentyear, taxamount \
                 FROM predictions_2017 \
                 JOIN properties_2017 using (parcelid) \
                 JOIN propertylandusetype using (propertylandusetypeid) \
                 WHERE month(transactiondate) >= 05 and month(transactiondate) <= 08 \
                 AND (propertylandusetypeid > 250 \
                 AND propertylandusetypeid < 280 \
                 AND propertylandusetypeid != 270  \
                 AND propertylandusetypeid != 271 \
                 OR unitcnt = 1) \
                 ORDER BY transactiondate DESC" 
    
    
    return pd.read_sql(zillow_sql, get_connection('zillow'))


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

def clean_zillow(df):
    
    
    '''
    For this practice zillow data frame, we will:
    
    Locate NaNs in different columns and removing those from the dataset. 
    
    Reformatting the latitude and longitude into proper format to allow for plotting within plotly.
    
    Create new features
    
    Rename Columns for readability
    
    Drop un-needed columns
    
    Convert some columns to objects
    
    Rid the dataframe of rows where the tax assessed value is over a certain number
    
    The parcel id will be used as our index
    
    We will return: df, a cleaned pandas dataframe
    '''
    df = get_zillow_data(cached=False)
    
    # Identify properties that do not qualify as single family homes
    indexpropids = df.loc[df['propertylandusetypeid'].isin([31, 46, 47, 246, 247, 248, 260, 267, 270, 271, 290, 291])].index
    
    # Delete properties that do not qualify as single family homes
    df.drop(indexpropids , inplace=True)
    
    # Create new column (age_of_home)
    today = pd.to_datetime('today')
    df['age_of_home'] = today.year - df['yearbuilt']
    
               
    # Remove NaNs from finished square feet
    df.loc[df['calculatedfinishedsquarefeet'].isin(['NaN'])].head()
    indexsize = df.loc[df['calculatedfinishedsquarefeet'].isin(['NaN'])].index
    df.drop(indexsize , inplace=True)
    
    # Replace '0' bathrooms with the median which is 2
    
    median_baths = df['bathroomcnt'].median(skipna=True)
    
    df['bathroomcnt']=df.bathroomcnt.mask(df.bathroomcnt == 0,median_baths)
    
     # Replace '0' bedrooms with the median which is 3
    
    median_beds = df['bedroomcnt'].median(skipna=True)
    
    df['bedroomcnt']=df.bedroomcnt.mask(df.bedroomcnt == 0,median_baths)
    
    # Remove NaNs from zip code
    
    indexzip = df.loc[df['regionidzip'].isin(['NaN'])].index
    df.drop(indexzip , inplace=True)
    
    # Remove NaNs from tax amount
    
    indextax = df.loc[df['taxamount'].isin(['NaN'])].index
    df.drop(indextax , inplace=True)
    
     # Remove NaNs from year_built
    
    indextax = df.loc[df['yearbuilt'].isin(['NaN'])].index
    df.drop(indextax , inplace=True)
    
    # Remove NaNs from tax value dollar count
    indextaxvalue = df.loc[df['taxvaluedollarcnt'].isin(['NaN'])].index
    df.drop(indextaxvalue, inplace=True)
    
    
    # Remove decimal from latitude and longitude
    df['latitude'] = df['latitude'].astype(int)
    df['longitude'] = df['longitude'].astype(int)
    
    # Convert latitude and longitude to positonal data points using lambda funtion (i.e. putting a decimal in the correct place)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    
    # Remove properties with a unit count greater than 1
    indexunits = df.loc[df['unitcnt'].isin([2, 3])].index
    
    # Delete properties that do not qualify as single family homes
    df.drop(indexunits , inplace=True)
    
    # Drop unitcnt column
    
    df.drop('unitcnt', axis=1, inplace=True)
    
    # Remove properties with a tax value greater than or equal to 1.68 million dollars and sqfootage greater than 3500 (these are outliers causing issues)
    
    indexpricey = df.loc[df['taxvaluedollarcnt'] >= 1680000].index 
    df.drop(indexpricey , inplace=True)
    
    
    #index3500 = df.loc[df['calculatedfinishedsquarefeet'] >= 3500].index
    #df.drop(index3500 , inplace=True)
    
    
    # Convert age_of_home column from float to an integer (this removes the decimal point)
    df['age_of_home'] = df['age_of_home'].astype(int)
    
      
    # Create bathrooms per sqft
    df['bath_pers_qft'] = df['bathroomcnt'] / df['calculatedfinishedsquarefeet']
    
    # Create bedrooms per sqft
    df['beds_pers_qft'] = df['bedroomcnt'] / df['calculatedfinishedsquarefeet']
    
    # Rename columns
    df.rename(columns = {'parcelid':'parcelid', 'propertylandusetypeid':'landuse_id','propertylandusedesc':'landuse_desc', 'transactiondate':'last_sold_date', 'calculatedfinishedsquarefeet':'total_sqft', 'bedroomcnt':'bedroom_quanity', 'bathroomcnt':'bathroom_quanity',
                     'fips':'fips', 'regionidzip':'zip_code', 'yearbuilt':'year_built', 'taxvaluedollarcnt':'tax_assessed_value', 'latitude':'latitude', 'longitude':'longitude',
                     'assessmentyear':'tax_assess_yr', 'taxamount':'property_tax', 'age_of_home':'age_of_home'}, inplace = True) 
    
    # Set parcelid as the index
    df = df.set_index('parcelid')
    
    # drop un-needed columns
    df.drop('landuse_id', axis=1, inplace=True)
    df.drop('landuse_desc', axis=1, inplace=True)
    df.drop('last_sold_date', axis=1, inplace=True)
    df.drop('tax_assess_yr', axis=1, inplace=True)
    
    # convert columns to object
    
    df['zip_code'] = df['zip_code'].astype(object)
    df['fips'] = df['fips'].astype(object)
    df['year_built'] = df['year_built'].astype(object)
    
    
    return df

def split_data():
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test
       
   

 # wrangle!
def wrangle_zillow():
    '''
    wrangle_zillow will read in our zillow data as a pandas dataframe,
    clean the data
    split the data
    return: train, validate, test sets of pandas dataframes from zillow
    '''
    df = clean_zillow(new_zillow_data())
    
    
    
    return df



def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    
    df = wrangle_zillow()
    
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, X_train, y_train, X_validate, y_validate, X_test, y_test



def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols



def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
     
    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled