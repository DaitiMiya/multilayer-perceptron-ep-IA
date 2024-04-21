import numpy as np
import random 


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
        
        # matriz_train_test = matriz_pixels[0:936]
        matriz_train_test = matriz_pixels[0:52]
        # random_matriz_train_test = random.sample(matriz_train_test,936)
        random_matriz_train_test = random.sample(matriz_train_test,52)
        print(random_matriz_train_test)
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
        random_matriz_train_test = random.sample(matriz_train_test,390)
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