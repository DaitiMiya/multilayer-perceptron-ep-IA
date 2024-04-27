from App.src import functions
from App.src.MLP import MLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    # Carregar conjuntos de dados (OR, AND e XOR)
    X_OR, y_OR = functions.carregar_dados_csv('Data/portas-logicas-alternativo/problemOR.csv')
    X_AND, y_AND = functions.carregar_dados_csv('Data/portas-logicas-alternativo/problemAND.csv')
    X_XOR, y_XOR = functions.carregar_dados_csv('Data/portas-logicas-alternativo/problemXOR.csv')

    # Definir a rede MLP
    n_input = X_OR.shape[1]  # Número de neurônios na camada de entrada
    n_hidden = 10  # Número de neurônios na camada escondida
    n_output = 1  # Número de neurônios na camada de saída

    mlp = MLP(n_input, n_hidden, n_output)

    # Treinar a rede MLP
    lr = 0.01  # Taxa de aprendizado
    epochs = 1000  # Número de iterações de treinamento

    mlp.train(X_OR, y_OR, lr, epochs)
    mlp.train(X_AND, y_AND, lr, epochs)
    mlp.train(X_XOR, y_XOR, lr, epochs)

    # Avaliar o Desempenho

    # Conjunto de dados OR
    X_OR_train, X_OR_test, y_OR_train, y_OR_test = train_test_split(X_OR, y_OR, test_size=0.2)
    y_OR_pred = mlp.predict(X_OR_test)
    accuracy_OR = accuracy_score(y_OR_test, y_OR_pred)
    precision_OR = precision_score(y_OR_test, y_OR_pred)
    recall_OR = recall_score(y_OR_test, y_OR_pred)
    f1_score_OR = f1_score(y_OR_test, y_OR_pred)

    print("Conjunto de dados OR:")
    print("Acurácia:", accuracy_OR)
    print("Precisão:", precision_OR)
    print("Recall:", recall_OR)
    print("F1-score:", f1_score_OR)

    # Conjunto de dados AND
    X_AND_train, X_AND_test, y_AND_train, y_AND_test = train_test_split(X_AND, y_AND, test_size=0.2)
    y_AND_pred = mlp.predict(X_AND_test)
    accuracy_AND = accuracy_score(y_AND_test, y_AND_pred)
    precision_AND = precision_score(y_AND_test, y_AND_pred)
    recall_AND = recall_score(y_AND_test, y_AND_pred)
    f1_score_AND = f1_score(y_AND_test, y_AND_pred)

    print("Conjunto de dados AND:")
    print("Acurácia:", accuracy_AND)
    print("Precisão:", precision_AND)
    print("Recall:", recall_AND)
    print("F1-score:", f1_score_AND)

    # Conjunto de dados XOR
    X_XOR_train, X_XOR_test, y_XOR_train, y_XOR_test = train_test_split(X_XOR, y_XOR, test_size=0.2)
    y_XOR_pred = mlp.predict(X_XOR_test)
    accuracy_XOR = accuracy_score(y_XOR_test, y_XOR_pred)
    precision_XOR = precision_score(y_XOR_test, y_XOR_pred)
    recall_XOR = recall_score(y_XOR_test, y_XOR_pred)
    f1_score_XOR = f1_score(y_XOR_test, y_XOR_pred)

    print("Conjunto de dados XOR:")
    print("Acurácia:", accuracy_XOR)
    print("Precisão:", precision_XOR)
    print("Recall:", recall_XOR)
    print("F1-score:", f1_score_XOR)

    print("Rede MLP treinada!")