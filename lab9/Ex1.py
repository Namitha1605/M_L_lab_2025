
#Write a program to partition a dataset (simulated data for regression)  into two parts,
# based on a feature (BP) and for a threshold,t = 80.
# Generate additional two partitioned datasets based on different threshold values of t = [78, 82].

import pandas as pd

data = pd.read_csv("data_ML.csv")

print(data.columns)
print(data.shape)
print(data.head())
print(data.info())

threshold_values = [80,78,82]

def partition_the_data(data,feature,threshold):
    data_high= data[data[feature] >= threshold]
    data_low = data[data[feature] <= threshold]
    return data_high,data_low

for i in threshold_values:
    data_high,data_low = partition_the_data(data,"BP",i)

    print(f"data_high (BP > {i}):  {data_high.head()}")
    print(f"data_low (BP <= {i}): {data_low.head()}")
