import numpy as np
import matplotlib.pyplot as plt

class MultiLayerPerceptron(): 
    def __init__(self, params=None):     
        if (params == None):
            self.Tamanho_camadas_entrada = 120                      # Input Layer
            self.Tamanho_camadas_escondida = 60                      # Hidden Layer
            self.Tamanho_camadas_saida = 26                       # Outpuy Layer
            self.Taxa_aprendizado = 0.5                  # Learning rate
            self.Maximo_epocas = 600                      # Epochs
            self.Bias_entrada_camada_escondida = 1                   # Bias HiddenLayer
            self.Bias_entrada_camada_saida = 1                  

        else:
            self.tamanho_camadas_entrada = params['TamanhoCamadaEntrada']
            self.tamanho_camadas_escondida = params['TamanhoCamadaEscondida']
            self.tamanho_camadas_saida = params['TamanhoCamadaSaida']
            self.taxa_aprendizado = params['TaxaDeAprendizado']
            self.maximo_epocas = params['Epocas']
            self.bias_entrada_camada_escondida = params['BiasCamadaEntrada']
            self.bias_saida_camada_escondida = params['BiasCamadaSaida']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def funcao_ativacao():
    ativacao = (lambda x: 1/(1 + np.exp(-x)))
    derivada = (lambda x: x*(1-x))
    
    return ativacao, derivada

def inicializando_pesos(tamanho_camada, neuronios):
    # Inicialização dos pesos e bias
    np.random.seed(0)  # Para reprodutibilidade
    # pesos_entrada_camada_escondida = np.random.uniform(-1, 1, (tamanho_camadas_escondida, tamanho_camadas_entrada))
    # pesos_saida_camada_escondida = np.random.uniform(-1, 1, (tamanho_camadas_saida, tamanho_camadas_escondida))
    pesos = np.random.uniform(-1, 1, (tamanho_camada, neuronios))
    return np.round(pesos,2)


def treina_com_parada_antecipada(train,test, resultado_esperado,tamanho_camadas_entrada,tamanho_camadas_escondida, tamanho_camada_saida, Bias_entrada_camada_escondida, Bias_entrada_camada_saida, max_epocas,taxa_aprendizado):
    ativacao, derivada = funcao_ativacao()
    epoca_atual = 0
    somatorio_erro = 0
    condicao_de_parada = False
    pesos_camada_escondida = inicializando_pesos(tamanho_camadas_entrada, tamanho_camadas_escondida)
    pesos_camada_saida = inicializando_pesos(tamanho_camadas_escondida,tamanho_camada_saida)
    
    resultado_camada_escondida = []
    resultado_camada_saida = []


    #Feedforward
    while epoca_atual < 20:
        for treinamento_atual, resultado_esperado_atual in zip(train,resultado_esperado):
            print(resultado_esperado_atual)
            print(f'----EPOCA ATUAL {epoca_atual}----')
            Ni_camada_escondida = np.dot(treinamento_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
            #60 saidas na ativacao camada escondida
            func_ativacao_camada_escondida = ativacao(Ni_camada_escondida)
            resultado_camada_escondida.append(func_ativacao_camada_escondida)

            Ni_camada_saida = np.dot(func_ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
            #26 saidas na ativacao camada saida 
            func_ativacao_camada_saida = ativacao(Ni_camada_saida)
            resultado_camada_saida.append(func_ativacao_camada_saida)
        
            # Erro da saída
            erro = np.subtract(func_ativacao_camada_saida,resultado_esperado_atual)
           
            #media erro ao quadrado
            somatorio_erro += np.mean(erro**2)/len(train)

            # Backpropagation
            #26 no delta
            delta_saida = erro * derivada(func_ativacao_camada_saida)
            # Correção de pesos da camada de saída
            #No outer criamos uma matriz de 60 linhas e 26 colunas para podermos ajustar
            pesos_camada_saida_correcao = taxa_aprendizado * np.outer(func_ativacao_camada_escondida, delta_saida)
   
            #pesos_camada_saida contem array de 26 pesos
            pesos_camada_saida += pesos_camada_saida_correcao

            # Cálculo do delta da camada escondida
            #pesos_camada_saida.T tem 26x60
            delta_escondida = np.dot(delta_saida, pesos_camada_saida.T) * derivada(func_ativacao_camada_escondida)
            
            # Correção de pesos da camada escondida
            pesos_camada_escondida_correcao = taxa_aprendizado * np.outer(treinamento_atual, delta_escondida)
            pesos_camada_escondida += pesos_camada_escondida_correcao
            print(f'ERRO {somatorio_erro}')
        # Verificação da condição de parada (aqui deve ser ajustado conforme seu critério)
        #Neste exemplo, a condição de parada é baseada na soma do erro
        epoca_atual += 1
    return resultado_camada_escondida, resultado_camada_saida

def treina_com_parada_antecipada_teste(train, treino_resultado_esperado, taxa_aprendizado, numero_epocas, tamanho_camada_saida):
    tamanho_camada_entrada = 120  # Cada entrada tem 120 características
    tamanho_camada_escondida = 60  # A camada escondida tem 60 neurônios
    # Inicializa os pesos e bias
    # Camada de entrada para a camada escondida
    pesos_entrada_escondida = np.random.randn(tamanho_camada_entrada, tamanho_camada_escondida) * 0.01
    bias_escondida = np.zeros((1, tamanho_camada_escondida))

    # Camada escondida para a camada de saída
    pesos_escondida_saida = np.random.randn(tamanho_camada_escondida, tamanho_camada_saida) * 0.01
    bias_saida = np.zeros((1, tamanho_camada_saida))

    for epoca_atual in range(50):
        somatorio_erro = 0
        for treinamento_atual, resultado_esperado in zip(train, treino_resultado_esperado):
            # Feedforward
            Ni_camada_escondida = np.dot(treinamento_atual, pesos_entrada_escondida) + bias_escondida
            resultado_camada_escondida = sigmoid(Ni_camada_escondida)

            Ni_camada_saida = np.dot(resultado_camada_escondida, pesos_escondida_saida) + bias_saida
            resultado_camada_saida = sigmoid(Ni_camada_saida)
            # Erro da saída
            erro = resultado_esperado - resultado_camada_saida
            print(erro)
            somatorio_erro += np.mean(erro**2)

            # # Backpropagation
            # delta_saida = erro * derivada_sigmoid(resultado_camada_saida)
            # pesos_escondida_saida += taxa_aprendizado * np.outer(resultado_camada_escondida, delta_saida)
            # bias_saida += taxa_aprendizado * delta_saida

            # delta_escondida = np.dot(delta_saida, pesos_escondida_saida) * derivada_sigmoid(Ni_camada_escondida)
            # pesos_entrada_escondida += taxa_aprendizado * np.outer(treinamento_atual, delta_escondida)
            # bias_escondida += taxa_aprendizado * delta_escondida

        # Aqui você pode adicionar uma condição de parada baseada em erro mínimo ou outra métrica
        # Por exemplo:
        # if somatorio_erro < um_erro_desejado:
        #     break

    return pesos_entrada_escondida, pesos_escondida_saida, bias_escondida, bias_saida

# def retropropagar_erro(treino_resultado_esperado, resultado_camada_escondida, pesos, Bias_entrada_camada_escondida, Bias_entrada_camada_saida, ativacao, derivada):
#     # Lista para armazenar os gradientes dos pesos da camada de saída
#     gradientes_camada_saida = []
#     # Lista para armazenar os gradientes dos pesos da camada escondida
#     gradientes_camada_escondida = []

#     # Passo 6: Retropropagação do erro na camada de saída
#     for indice, (resultado_esperado, resultado_obtido) in enumerate(zip(treino_resultado_esperado, resultado_camada_escondida)):
#         erro_camada_saida = resultado_esperado - resultado_obtido
#         delta_saida = erro_camada_saida * derivada(resultado_obtido)

#         # Atualiza pesos e bias da camada de saída
#         gradiente_pesos_saida = delta_saida * resultado_camada_escondida[indice]
#         gradiente_bias_saida = delta_saida
        
#         gradientes_camada_saida.append((gradiente_pesos_saida, gradiente_bias_saida))

#     # Passo 7: Retropropagação do erro na camada escondida
#     for indice_neuronio_escondido, pesos_neuronio in enumerate(pesos):
#         for gradiente_pesos_saida, gradiente_bias_saida in gradientes_camada_saida:
#             # Calcular o delta para a camada escondida
#             delta_escondida = np.dot(pesos_neuronio, gradiente_pesos_saida) * derivada(resultado_camada_escondida[indice_neuronio_escondido])
            
#             # Atualiza pesos e bias da camada escondida
#             gradiente_pesos_escondida = delta_escondida * resultado_camada_escondida[indice_neuronio_escondido]
#             gradiente_bias_escondida = delta_escondida
            
#             gradientes_camada_escondida.append((gradiente_pesos_escondida, gradiente_bias_escondida))
    
#     # Atualiza os pesos e bias com base nos gradientes calculados
#     # Isso seria feito após acumular todos os gradientes dos exemplos de treinamento
#     # A taxa de aprendizado determina o quanto atualizar

#     return gradientes_camada_saida, gradientes_camada_escondida