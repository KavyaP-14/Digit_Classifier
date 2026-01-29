import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0  # normalize
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


input_size = 64
hidden_size = 64
output_size = 10

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


lr = 0.1

def backward(X, y, Z1, A1, A2):
    global W1, b1, W2, b2
    
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / len(X)
    db2 = np.mean(dZ2, axis=0)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    
    dW1 = X.T @ dZ1 / len(X)
    db1 = np.mean(dZ1, axis=0)
    
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


losses = []
accuracies = []

for epoch in range(300):
    Z1, A1, Z2, A2 = forward(X_train)
    loss = cross_entropy(y_train_oh, A2)
    
    backward(X_train, y_train_oh, Z1, A1, A2)
    
    preds = np.argmax(A2, axis=1)
    acc = np.mean(preds == y_train)
    
    losses.append(loss)
    accuracies.append(acc)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")


_, _, _, test_pred = forward(X_test)
test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
print("Test Accuracy:", test_acc)


plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


plt.plot(accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
