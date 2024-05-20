import numpy as np
import random 

path_X = 'Data/X.txt'
path_y = 'Data/Y_letra.txt'
path_resultado_esperado = 'Data/Y_classe.npy'

def get_train_test():
    try:

        matriz_pixels = read_txt(path_X)

        #36 alfabetos(70% da base usada para treinamento)
        train, treino_resultado_esperado = get_train_data(matriz_pixels)

        #15 alfabetos(30% da base usada para treinamento)
        test, teste_resultado_esperado = get_test_data(matriz_pixels)
        # print(teste_resultado_esperado)
        return (train,treino_resultado_esperado, test, teste_resultado_esperado)
    except Exception as e:    
        print('Erro ao gerar treino e teste')
        raise e

def mapear_valores(indices):
    
    valores_mapeados = []
    resultado_esperado = get_resultado_esperado()

    for index in indices:
        resultado_esperado_indice = resultado_esperado[index]

        valores_mapeados.append(list(resultado_esperado_indice))
    return valores_mapeados

def get_validation_data():
    matriz_pixels = read_txt(path_X)
    matriz_train_test = matriz_pixels[0:50]

    indices = list(range(len(matriz_train_test)))

    # Embaralhando os índices
    random.shuffle(indices)
    # Criando uma nova matriz com a ordem dos arrays embaralhada
    random_matriz_validacao = [matriz_pixels[i] for i in indices]

    matriz_validacao_esperada = mapear_valores(indices)
    return random_matriz_validacao, matriz_validacao_esperada

def get_train_data(matriz_pixels):
    try:
        
        matriz_train_test = matriz_pixels[0:1196]

        indices = list(range(len(matriz_train_test)))
    
        # Embaralhando os índices
        random.shuffle(indices)
        # Criando uma nova matriz com a ordem dos arrays embaralhada
        random_matriz_train_test = [matriz_pixels[i] for i in indices]
    
        matriz_esperada = mapear_valores(indices)
        return random_matriz_train_test, matriz_esperada
    except Exception as e:
        print('Error ao pegar array de treinamento')
        raise e

def get_resultado_esperado():
    resultado_esperado = np.load(path_resultado_esperado)
    return resultado_esperado

def get_test_data(matriz_pixels):
    try:
        matriz_train_test = matriz_pixels[1196:1327]
        indices = list(range(len(matriz_train_test)))
        
        # Embaralhando os índices
        random.shuffle(indices)

        # Criando uma nova matriz com a ordem dos arrays embaralhada
        random_matriz_train_test = [matriz_pixels[i] for i in indices]
        matriz_esperada = mapear_valores(indices)
        return random_matriz_train_test,matriz_esperada
    except Exception as e:
        print('Error ao pegar array de teste')
        raise e


def read_txt(path_pixel):
    try:
        # Lendo os arrays de pixels do arquivo
        arrays_pixels = []
        with open(path_pixel, "r") as arquivo_pixels:
            linhas_pixels = arquivo_pixels.readlines()
            for linha in linhas_pixels:
                # Convertendo a linha em uma lista de inteiros
                array_pixels = list(map(int, linha.replace(",", "").strip().split()))
                arrays_pixels.append(array_pixels)


        return arrays_pixels
    except Exception as e:
        print('Erro em ler arquivo txt')
        raise e