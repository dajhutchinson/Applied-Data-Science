import forex_python.converter as fx
from datetime import datetime
import numpy as np
import pandas as pd

def save_data(df:pd.DataFrame,file_path):
    df.to_csv(file_path)

def anonymise(series:pd.Series) -> pd.Series:
    return series.astype("category").cat.codes

def dob_to_age(df) -> pd.Series:
    return (df["trans_date_trans_time"]-df["dob"])//np.timedelta64(1,"Y")

def anonymise_data(df, printing=True) -> pd.DataFrame:
    # prepare data
    if (printing): print("Data ",end="",flush=True)
    df["trans_date_trans_time"]=pd.to_datetime(df["trans_date_trans_time"],format="%Y-%m-%d %H:%M:%S")
    df["dob"]=pd.to_datetime(df["dob"],format="%Y-%m-%d")
    if (printing): print("PREPARED")

    # clean data
    clean_df=pd.DataFrame()
    clean_df["is_fraud"]=df["is_fraud"]

    if (printing): print("Time ",end="",flush=True)
    clean_df["unix_time"]=df["unix_time"].copy()
    if (printing): print("DONE")

    if (printing): print("Amount ",end="",flush=True)
    clean_df["amt"]=df["amt"].copy()
    if (printing): print("DONE")

    if (printing): print("Credit Card ",end="",flush=True)
    clean_df["cc_id"]=anonymise(df["cc_num"])
    if (printing): print("DONE")

    if (printing): print("Person  ",end="",flush=True)
    clean_df["person_id"]=anonymise(df["first"]+"_"+df["last"]+"_"+df["job"]+"_"+df["dob"].apply(lambda x: x.strftime('%Y-%m-%d')))
    if (printing): print("DONE")

    if (printing): print("Gender ",end="",flush=True)
    clean_df["gender_id"]=anonymise(df["gender"])
    if (printing): print("DONE")

    if (printing): print("Job ",end="",flush=True)
    clean_df["job_category"]=df["job"].copy()
    if (printing): print("DONE")

    if (printing): print("Age ",end="",flush=True)
    clean_df["age"]=dob_to_age(df[["dob","trans_date_trans_time"]])
    if (printing): print("DONE")

    if (printing): print("City Pop ",end="",flush=True)
    clean_df["city_pop_round"]=50*np.floor(df["city_pop"]/50).copy().astype(int)
    if (printing): print("DONE")

    if (printing): print("Merchant ",end="",flush=True)
    clean_df["merchant_id"]=anonymise(df["merchant"])
    clean_df["merchant_category"]=df["category"]
    if (printing): print("DONE")

    return clean_df

print("Data ",end="",flush=True)
training_data=pd.read_csv("data/synthetic_train.csv",index_col=0)
test_data=pd.read_csv("data/synthetic_train.csv",index_col=0)
full_data=pd.concat([training_data, test_data])
print("LOADED")

clean_df=anonymise_data(full_data)
save_data(clean_df,"data/cleaned_synthetic_data.csv")
