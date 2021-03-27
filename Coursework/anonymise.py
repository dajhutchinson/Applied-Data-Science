import forex_python.converter as fx
from datetime import datetime
import numpy as np
import pandas as pd

def save_data(df:pd.DataFrame,file_path):
    df.to_csv(file_path)

# Convert amount to gbp
def convert_currency(amount:float,unix_time:int,cur_currency:str,tar_currency) -> float:
    """
    Determine the value of an amount of one currency in another currency at a specified point in time

    PARAMS
    amount (float) - amount of current currency
    unix_time (int) - unix timestamp of exchange rate to use
    cur_currency (str) - three character code for current currency
    tar_currency (str) - three character code for target currency

    RETURNS
    float - amount of target currency
    """
    time=datetime.utcfromtimestamp(unix_time)
    exchange_rate=fx.get_rate(cur_currency,tar_currency,time)

    return round(amount*exchange_rate,2)

# Convert amount to gbp
def prepare_amount(df,cur_label,cur_currency="USD",tar_currency="GBP") -> pd.Series:
    df_local=df.copy()
    df_local["date"]=pd.to_datetime(df["trans_date_trans_time"].dt.date,format="%Y-%m-%d")

    # determine the exchange rate for each day
    exchange_rates=pd.DataFrame()
    exchange_rates["date"]=pd.to_datetime(df_local["date"].unique(),format="%Y-%m-%d")
    exchange_rates["unix_time"]=(exchange_rates["date"].astype("int64")/1000000000).astype(int)
    exchange_rates["rate"]=exchange_rates.apply(lambda x:convert_currency(1,x["unix_time"],cur_currency,tar_currency),axis=1)

    # merge dataframes
    df_local["date"]=df_local["date"].dt.date
    exchange_rates["date"]=exchange_rates["date"].dt.date
    df_merged=df_local[["date","amt"]].merge(exchange_rates[["date","rate"]],on="date",how="left")

    # calculated exchanged amounts
    tar_label="amount_{}".format(tar_currency)
    df_merged[tar_label]=df_merged.apply(lambda x:x["amt"]*x["rate"],axis=1)

    return df_merged[tar_label]

def standardise_time(series) -> pd.Series:
    min_time=series.min().to_pydatetime()
    min_day=min_time.replace(second=0,minute=0,hour=0)
    return ((series-min_day).dt.total_seconds()).astype(int)

def anonymise(series:pd.Series) -> pd.Series:
    return series.astype("category").cat.codes

def dob_to_age(df) -> pd.Series:
    return (df["trans_date_trans_time"]-df["dob"])//np.timedelta64(1,"Y")

def anonymise_data(df, printing=True) -> pd.DataFrame:
    # prepare data
    if (printing): print("Data ",end="",flush=True)
    full_data["trans_date_trans_time"]=pd.to_datetime(full_data["trans_date_trans_time"],format="%Y-%m-%d %H:%M:%S")
    full_data["dob"]=pd.to_datetime(full_data["dob"],format="%Y-%m-%d")
    if (printing): print("PREPARED")

    # clean data
    clean_df=pd.DataFrame()
    clean_df["is_fraud"]=full_data["is_fraud"]

    if (printing): print("Money ",end="",flush=True)
    clean_df["amount_USD"]=full_data["amt"].copy()
    clean_df["amount_GBP"]=prepare_amount(full_data[["trans_date_trans_time","amt"]],"amt","USD","GBP")
    if (printing): print("DONE")

    if (printing): print("Time ",end="",flush=True)
    clean_df["unix_time"]=full_data["unix_time"].copy()
    clean_df["seconds_from_start"]=standardise_time(full_data["trans_date_trans_time"])
    if (printing): print("DONE")

    if (printing): print("Credit Card ",end="",flush=True)
    clean_df["cc_id"]=anonymise(full_data["cc_num"])
    if (printing): print("DONE")

    if (printing): print("Person  ",end="",flush=True)
    clean_df["person_id"]=anonymise(full_data["first"]+"_"+full_data["last"]+"_"+full_data["job"]+"_"+full_data["dob"].apply(lambda x: x.strftime('%Y-%m-%d')))
    if (printing): print("DONE")

    if (printing): print("Gender ",end="",flush=True)
    clean_df["gender_id"]=anonymise(full_data["gender"])
    if (printing): print("DONE")

    if (printing): print("Job ",end="",flush=True)
    clean_df["job_category"]=full_data["job"]
    if (printing): print("DONE")

    if (printing): print("Age ",end="",flush=True)
    clean_df["age"]=dob_to_age(full_data[["dob","trans_date_trans_time"]])
    if (printing): print("DONE")

    if (printing): print("City Pop ",end="",flush=True)
    clean_df["city_pop_round"]=np.ceil(full_data["city_pop"]/1000).copy().astype(int)
    if (printing): print("DONE")

    if (printing): print("Merchant ",end="",flush=True)
    clean_df["merchant_id"]=anonymise(full_data["merchant"])
    clean_df["merchant_category"]=full_data["category"]
    if (printing): print("DONE")

    return clean_df

print("Data ",end="",flush=True)
training_data=pd.read_csv("data/synthetic_train.csv",index_col=0)
test_data=pd.read_csv("data/synthetic_train.csv",index_col=0)
full_data=pd.concat([training_data, test_data])
print("LOADED")

clean_df=anonymise_data(full_data)
save_data(clean_df,"data/prepared_synthetic_data.csv")
