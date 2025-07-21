import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import ipaddress
import os

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Load data
fraud_df = pd.read_csv('data/Fraud_Data.csv')
ip_df = pd.read_csv('data/IpAddress_to_Country.csv')
cc_df = pd.read_csv('data/creditcard.csv')

# Handle missing values
fraud_df = fraud_df.dropna()
ip_df = ip_df.dropna()
cc_df = cc_df.dropna()

# Remove duplicates
fraud_df = fraud_df.drop_duplicates()
ip_df = ip_df.drop_duplicates()
cc_df = cc_df.drop_duplicates()

# Correct data types
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
fraud_df['age'] = fraud_df['age'].astype(int)

# EDA (plots will be saved as PNGs)
plt.figure(figsize=(6,4))
fraud_df['purchase_value'].hist(bins=50)
plt.title('Purchase Value Distribution')
plt.xlabel('Purchase Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("data/processed/purchase_value_hist.png")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x='class', y='purchase_value', data=fraud_df)
plt.title('Purchase Value by Class')
plt.tight_layout()
plt.savefig("data/processed/purchase_value_by_class.png")
plt.close()

# Merge Datasets for Geolocation Analysis
def ip_to_int(ip):
    try:
        return int(ipaddress.IPv4Address(ip))
    except:
        return np.nan

fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

def find_country(ip):
    row = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
    if not row.empty:
        return row.iloc[0]['country']
    return 'Unknown'

fraud_df['country'] = fraud_df['ip_int'].apply(find_country)

# Feature Engineering
user_freq = fraud_df.groupby('user_id').size().rename('transaction_count')
fraud_df = fraud_df.merge(user_freq, on='user_id')
fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600

# Data Transformation
print('Class distribution before SMOTE:')
print(fraud_df['class'].value_counts())

# Drop identifier columns that are not useful for modeling
drop_cols = ['class', 'ip_address', 'signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_int']
X = fraud_df.drop(drop_cols, axis=1)
y = fraud_df['class']

categorical = ['source', 'browser', 'sex', 'country']
X = pd.get_dummies(X, columns=categorical, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print('Class distribution after SMOTE:')
print(np.bincount(y_train_res))

# Save processed data for modeling
np.savez("../data/processed/fraud_train.npz", X=X_train_res, y=y_train_res)
np.savez("../data/processed/fraud_test.npz", X=X_test, y=y_test)

print("Preprocessing complete. Processed data and EDA plots saved in data/processed/")