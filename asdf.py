import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def softmax(Z):
    # Z shape is (K, N)
    max_Z = np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z - max_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def compute_gradients(X, Y, W, b):
    N = X.shape[0]
    K = W.shape[0]

    # Compute the logits (K, N)
    logits = np.dot(W, X.T) + b[:, np.newaxis]
    # Compute the softmax probabilities (K, N)
    probs = softmax(logits)

    # Compute gradients
    dW = np.dot(probs - Y.T, X) / N  # (K, F)
    dB = np.sum(probs - Y.T, axis=1) / N  # (K,)

    return dW, dB

def logreg_fit(X, Y, W, b, lr):
    dW, dB = compute_gradients(X, Y, W, b)
    W -= lr * dW
    b -= lr * dB

def logreg_train(path: str):
    # dataframe from csv
    df = pd.read_csv(path)
    df.fillna(0, inplace=True) # avoid NaN

    # number of inputs:
    n = df.shape[0]
    # get the input features
    X = StandardScaler().fit_transform(df.iloc[:, 6:].values)
    # get the actual result
    Y = pd.get_dummies(df.iloc[:, 1]).values
    
    # Initialize parameters
    K = Y.shape[1]
    F = X.shape[1]
    W = np.zeros((K, F))
    b = np.zeros(K)

    epochs = 100
    lr = 0.1
    for epoch in range(epochs):
        logreg_fit(X, Y, W, b, lr)
        print(f"Epoch {epoch+1}/{epochs} completed")
    
    np.savetxt("weighs.csv", W, delimiter=",")
    np.savetxt("bias.csv", b, delimiter=",")

if __name__ == "__main__":
    logreg_train("datasets/dataset_train.csv")