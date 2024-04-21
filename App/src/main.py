from Functions import funcions_data as func

path_X = 'Data/X.txt'
path_y = 'Data/Y_letra.txt'

def get_train_test():

    matriz_pixels, mapeamento = func.read_txt(path_X, path_y)
    

    #36 alfabetos(70% da base usada para treinamento)
    train, treino_resultado_esperado = func.get_train_data(matriz_pixels,mapeamento)
    

    #15 alfabetos(30% da base usada para treinamento)
    test, teste_resultado_esperado = func.get_test_data(matriz_pixels, mapeamento)
    print(teste_resultado_esperado)
    
    return (train,test)
    
def train_parada_antecipada(entradas, saidas):
    return True

if __name__ == "__main__":
    get_train_test()
    #train_parada_antecipada(train, test)