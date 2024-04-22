import numpy as np

class MultiLayerPerceptron(): 
    def __init__(self, params=None):     
        if (params == None):
            self.Tamanho_camadas_entrada = 120                      # Input Layer
            self.Tamanho_camadas_escondida = 60                      # Hidden Layer
            self.Tamanho_camadas_saida = 26                       # Outpuy Layer
            self.Taxa_aprendizado = 0.3                  # Learning rate
            self.Maximo_epocas = 600                      # Epochs
            self.Bias_entrada_camada_escondida = 1                   # Bias HiddenLayer
            self.Bias_entrada_camada_saida = 1                  

        else:
            self.tamanho_camadas_entrada = params['TamanhoCamadaEntrada']
            self.tamanho_camadas_escondida = params['TamanhoCamadaEscondida']
            self.tamanho_camadas_saida = params['TamanhoCamadaSaida']
            self.taxa_aprendizado = params['TaxaDeAprendizado']
            self.maximo_epocas = params['Epocas']
            self.BiasHiddenValue = params['BiasCamadaEntrada']
            self.BiasOutputValue = params['BiasCamadaSaida']

def funcao_ativacao():
    ativacao = (lambda x: 1/(1 + np.exp(-x)))
    derivada = (lambda x: x*(1-x))
    
    return ativacao, derivada

def inicializando_pesos(tamanho_camadas_entrada,tamanho_camadas_escondida,tamanho_camadas_saida):
    # Inicialização dos pesos e bias
    np.random.seed(0)  # Para reprodutibilidade
    pesos_entrada_camada_escondida = np.random.uniform(-1, 1, (tamanho_camadas_escondida, tamanho_camadas_entrada))
    pesos_saida_camada_escondida = np.random.uniform(-1, 1, (tamanho_camadas_saida, tamanho_camadas_escondida))

    return pesos_entrada_camada_escondida, pesos_saida_camada_escondida

def treina_com_parada_antecipada(pesos_entrada_camada_escondida, pesos_saida_camada_escondida):
    return True