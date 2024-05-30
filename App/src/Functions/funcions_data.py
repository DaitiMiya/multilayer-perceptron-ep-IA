#######################################################################
#                Inteligencia Artificial - ACH2016                    #
#                                                                     #
#  Gandhi Daiti Miyahara 11207773                                     #
#  Lucas Tatsuo Nishida 11208270                                      #
#  Juan Kineipe 11894610                                              #
#  Leonardo Ken Matsuda Cancela 11207665                              #
#  João de Araújo Barbosa da Silva 11369704                           #
#                                                                     #
#######################################################################
import numpy as np
from random import random, sample, shuffle

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
        
        validacao, validacao_esperada = get_validation_data(matriz_pixels)
        # print(teste_resultado_esperado)
        return (train,treino_resultado_esperado, test, teste_resultado_esperado, validacao, validacao_esperada)
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

#Justificar escolha de dados
#Alfabetos inteiros(30 por cento da base +- 13 alfabetos)
def get_validation_data(matriz_pixels):
    indices_aleatorios = sample(range(0,338), 338)
    # # Embaralhando os índices
    random_matriz_train_test = [matriz_pixels[i] for i in indices_aleatorios]
    matriz_validacao_esperada = mapear_valores(indices_aleatorios)
    return random_matriz_train_test, matriz_validacao_esperada

def get_train_data(matriz_pixels):
    try:
        
        # matriz_train_test = matriz_pixels[338:1196]
        indices_aleatorios = sample(range(338,1196), 858)

        
        # indices = list(range(len(matriz_train_test)))
    
        # # Embaralhando os índices
        # random.shuffle(indices)
        # print(indices)
        # # Criando uma nova matriz com a ordem dos arrays embaralhada
        # random_matriz_train_test = [matriz_pixels[indices]]
        random_matriz_train_test = [matriz_pixels[i] for i in indices_aleatorios]
       
        matriz_esperada = mapear_valores(indices_aleatorios)

        return random_matriz_train_test, matriz_esperada
    except Exception as e:
        print('Error ao pegar array de treinamento')
        raise e

def get_resultado_esperado():
    resultado_esperado = np.load(path_resultado_esperado)
    return resultado_esperado

def get_test_data(matriz_pixels):
    try:

        indices_aleatorios = sample(range(1196,1326), 130)
        # Criando uma nova matriz com a ordem dos arrays embaralhada
        random_matriz_train_test = [matriz_pixels[i] for i in indices_aleatorios]
        matriz_esperada = mapear_valores(indices_aleatorios)
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