import numpy as np


class MLP:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Inicializar pesos e vieses
        self.w_ih = np.random.randn(n_input, n_hidden)
        self.b_h = np.zeros(n_hidden)
        self.w_ho = np.random.randn(n_hidden, n_output)
        self.b_o = np.zeros(n_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def feedforward(self, x):
        # Camada escondida
        h = self.sigmoid(np.dot(x, self.w_ih) + self.b_h)

        # Camada de saída
        o = self.softmax(np.dot(h, self.w_ho) + self.b_o)

        return o

    def train(self, X, y, lr, epochs):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # Obter o input da rede (x) para a amostra atual
                x = X[i]

                # Camada escondida
                h = self.sigmoid(np.dot(x, self.w_ih) + self.b_h)

                # Forward pass
                o = self.feedforward(X[i])

                # Calcular o erro
                error = y[i] - o

                # Backpropagation
                delta_o = error * o * (1 - o)
                delta_h = np.dot(delta_o, self.w_ho.T) * h * (1 - h)

                # Atualizar pesos e bias
                self.w_ho -= lr * np.dot(h.T, delta_o)
                self.b_o -= lr * delta_o
                self.w_ih -= lr * np.dot(x.T, delta_h)
                self.b_h -= lr * delta_h

    def predict(self, X):
        o = self.feedforward(X)
        return o.argmax(axis=1)
