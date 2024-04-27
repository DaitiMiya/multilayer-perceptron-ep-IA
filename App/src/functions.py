import csv
import numpy as np


def carregar_dados_csv(nome_arquivo):
    dados = []
    with open(nome_arquivo, 'r') as arquivo:
        leitor_csv = csv.reader(arquivo)
        for linha in leitor_csv:
            dados.append(linha)

    dados = np.array(dados)
    X = dados[:, :].astype(np.int32)  # Converter características para float32
    y = dados[:, -1].astype(np.int32)  # Converter rótulos para int32

    return X, y
