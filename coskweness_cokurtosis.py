import pandas as pd

"""
def coskewness(df: pd.DataFrame):
    n = len(df)
    means = df.mean()
    std_devs = df.std(ddof=1)

    index = pd.MultiIndex.from_tuples(
        [(i, j, k) for i in df.columns for j in df.columns for k in df.columns], 
        names=["Var1", "Var2", "Var3"]
    )
    
    values = [
        ((df[i] - means[i]) * (df[j] - means[j]) * (df[k] - means[k])).sum() / (n - 1) / (std_devs[i] * std_devs[j] * std_devs[k])
        for i in df.columns for j in df.columns for k in df.columns
    ]
    
    return pd.DataFrame(values, index=index, columns=["coskewness"])


def cokurtosis(df: pd.DataFrame):
    n = len(df)
    means = df.mean()
    std_devs = df.std(ddof=1)

    index = pd.MultiIndex.from_tuples(
        [(i, j, k, l) for i in df.columns for j in df.columns for k in df.columns for l in df.columns], 
        names=["Var1", "Var2", "Var3", "Var4"]
    )
    
    values = [
        ((df[i] - means[i]) * (df[j] - means[j]) * (df[k] - means[k]) * (df[l] - means[l])).sum() / (n - 1) / (std_devs[i] * std_devs[j] * std_devs[k] * std_devs[l])
        for i in df.columns for j in df.columns for k in df.columns for l in df.columns
    ]
    
    return pd.DataFrame(values, index=index, columns=["cokurtosis"])

def coskewness(df: pd.DataFrame):
    n = len(df)
    means = df.mean()
    std_devs = df.std(ddof=1)
    
    coskew_dict = {}
    for i in df.columns:
        for j in df.columns:
            for k in df.columns:
                coskew_dict[(i, j, k)] = ((df[i] - means[i]) * (df[j] - means[j]) * (df[k] - means[k])).sum() / (n-1)
                coskew_dict[(i, j, k)] /= (std_devs[i] * std_devs[j] * std_devs[k])
    
    return coskew_dict


def cokurtosis(df: pd.DataFrame):
    n = len(df)
    means = df.mean()
    std_devs = df.std(ddof=1)
    
    cokurt_dict = {}
    for i in df.columns:
        for j in df.columns:
            for k in df.columns:
                for l in df.columns:
                    cokurt_dict[(i, j, k, l)] = ((df[i] - means[i]) * (df[j] - means[j]) * (df[k] - means[k]) * (df[l] - means[l])).sum() / (n-1)
                    cokurt_dict[(i, j, k, l)] /= (std_devs[i] * std_devs[j] * std_devs[k] * std_devs[l])
    
    return pd.DataFrame(cokurt_dict, index=df.columns, columns=df.columns)

"""

import numpy as np
import torch


def coskewness(X):
    n = X.shape[0]
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0, ddof=0)
    coskew = np.zeros((X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                coskew[i, j, k] = np.sum(
                    (X[:, i] - means[i]) * (X[:, j] - means[j]) * (X[:, k] - means[k])
                ) / (n - 1)
                
                # Normalize by the product of the standard deviations
                coskew[i, j, k] /= (std_devs[i] * std_devs[j] * std_devs[k])

    return coskew
    
    
def cokurtosis(X):
    n = X.shape[0]
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0, ddof=0)
    cokurt = np.zeros((X.shape[1], X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                for l in range(X.shape[1]):
                    cokurt[i, j, k, l] = np.sum(
                        (X[:, i] - means[i]) *
                        (X[:, j] - means[j]) *
                        (X[:, k] - means[k]) *
                        (X[:, l] - means[l])
                    ) / (n - 1)

                    # Normalize by the product of the standard deviations
                    cokurt[i, j, k, l] /= (std_devs[i] * std_devs[j] * std_devs[k] * std_devs[l])

    return cokurt - 3

# Write torch functions
def coskewness_torch(X):
    n = X.shape[0]
    if torch.isnan(X[0]).any():
        print("Removing NaN values")
        X = X[1:]
        n -= 1
    means = torch.mean(X, axis=0)
    std_devs = torch.std(X, axis=0)
    coskew = torch.zeros((X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                coskew[i, j, k] = torch.sum(
                    (X[:, i] - means[i]) * (X[:, j] - means[j]) * (X[:, k] - means[k])
                ) / n
                
                # Normalize by the product of the standard deviations
                coskew[i, j, k] /= (std_devs[i] * std_devs[j] * std_devs[k])

    return coskew

def cokurtosis_torch(X):
    n = X.shape[0]
    if torch.isnan(X[0]).any():
        X = X[1:]
        n -= 1
    means = torch.mean(X, axis=0)
    std_devs = torch.std(X, axis=0, unbiased=True)
    cokurt = torch.zeros((X.shape[1], X.shape[1], X.shape[1], X.shape[1]))

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            for k in range(X.shape[1]):
                for l in range(X.shape[1]):
                    cokurt[i, j, k, l] = torch.sum(
                        (X[:, i] - means[i]) *
                        (X[:, j] - means[j]) *
                        (X[:, k] - means[k]) *
                        (X[:, l] - means[l])
                    ) / n

                    # Normalize by the product of the standard deviations
                    cokurt[i, j, k, l] /= (std_devs[i] * std_devs[j] * std_devs[k] * std_devs[l])

    return cokurt