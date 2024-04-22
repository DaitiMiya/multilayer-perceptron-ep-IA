import numpy as np
import random 

path_X = 'Data/X.txt'
path_y = 'Data/Y_letra.txt'

def get_train_test():
    try:

        matriz_pixels, mapeamento = read_txt(path_X, path_y)
        

        #36 alfabetos(70% da base usada para treinamento)
        train, treino_resultado_esperado = get_train_data(matriz_pixels,mapeamento)

        #15 alfabetos(30% da base usada para treinamento)
        test, teste_resultado_esperado = get_test_data(matriz_pixels, mapeamento)
        # print(teste_resultado_esperado)
        return (train,test)
    except Exception as e:    
        print('Erro ao gerar treino e teste')
        raise e

def mapear_valores(matriz_valores, mapeamento):
    
    valores_mapeados = []
    print(len(matriz_valores))
    for linha_valores in matriz_valores:
        valor_mapeado = None
        for pixel_letra_mapeada, letra  in mapeamento.items():
            if linha_valores == list(pixel_letra_mapeada):
                valor_mapeado = letra
                break
        valores_mapeados.append(valor_mapeado)
    return valores_mapeados

def get_train_data(matriz_pixels, mapeamento):
    try:
        
        matriz_train_test = matriz_pixels[0:936]
        # matriz_train_test = matriz_pixels[0:52]
        # random_matriz_train_test = random.sample(matriz_train_test,936)
        # random_matriz_train_test = random.sample(matriz_train_test,52)
        indices = list(range(len(matriz_train_test)))
        
        # Embaralhando os índices
        random.shuffle(indices)

        # Criando uma nova matriz com a ordem dos arrays embaralhada
        random_matriz_train_test = [matriz_pixels[i] for i in indices]
        matriz_esperada = mapear_valores(random_matriz_train_test,mapeamento)
        return random_matriz_train_test, matriz_esperada
    except Exception as e:
        print('Error ao pegar array de treinamento')
        raise e


def get_test_data(matriz_pixels,mapeamento):
    try:
        matriz_train_test = matriz_pixels[936:1327]
        indices = list(range(len(matriz_train_test)))
        
        # Embaralhando os índices
        random.shuffle(indices)

        # Criando uma nova matriz com a ordem dos arrays embaralhada
        random_matriz_train_test = [matriz_pixels[i] for i in indices]
        matriz_esperada = mapear_valores(random_matriz_train_test,mapeamento)
        return random_matriz_train_test,matriz_esperada
    except Exception as e:
        print('Error ao pegar array de teste')
        raise e


def read_txt(path_pixel, path_letras):
    try:
        # Lendo os arrays de pixels do arquivo
        arrays_pixels = []
        with open(path_pixel, "r") as arquivo_pixels:
            linhas_pixels = arquivo_pixels.readlines()
            for linha in linhas_pixels:
                # Convertendo a linha em uma lista de inteiros
                array_pixels = list(map(int, linha.replace(",", "").strip().split()))
                arrays_pixels.append(array_pixels)

        # Lendo as letras do arquivo
        letras = []
        with open(path_letras, "r") as arquivo_letras:
            letras = arquivo_letras.read().splitlines()

        # Mapeando as letras aos arrays de pixels
        mapeamento = {}
        
        for i in range(len(arrays_pixels)):
            chave = tuple(arrays_pixels[i])  # Convertendo a lista de valores em tupla
            mapeamento[chave] = letras[i]

        return arrays_pixels,mapeamento
    except Exception as e:
        print('Erro em ler arquivo txt')
        raise e