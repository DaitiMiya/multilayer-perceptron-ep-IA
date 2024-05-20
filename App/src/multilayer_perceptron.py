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


def treina_sem_parada_antecipada(train, resultado_esperado,tamanho_camadas_entrada,tamanho_camadas_escondida, tamanho_camada_saida, Bias_entrada_camada_escondida, Bias_entrada_camada_saida, max_epocas,taxa_aprendizado):
    ativacao, derivada = funcao_ativacao()
    epoca_atual = 0
    resultados_por_epoca = []
    condicao_de_parada = False
    pesos_camada_escondida = inicializando_pesos(tamanho_camadas_entrada, tamanho_camadas_escondida)
    pesos_camada_saida = inicializando_pesos(tamanho_camadas_escondida,tamanho_camada_saida)
    
    resultado_camada_escondida = []
    resultado_camada_saida = []
    resultado_somatoria_erro = []
    resultado_acuracia = []
    #Feedforward
    # while epoca_atual < 100:
    #     print(f'----EPOCA ATUAL {epoca_atual}----')
    #     for treinamento_atual, resultado_esperado_atual in zip(train,resultado_esperado):
    #         acuracia = 0
    #         somatorio_erro = 0
    #         total_previsoes = 0
    #         Ni_camada_escondida = np.dot(treinamento_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
    #         #60 saidas na ativacao camada escondida
    #         func_ativacao_camada_escondida = ativacao(Ni_camada_escondida)
    #         resultado_camada_escondida.append(func_ativacao_camada_escondida)

    #         Ni_camada_saida = np.dot(func_ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
    #         #26 saidas na ativacao camada saida 
    #         func_ativacao_camada_saida = ativacao(Ni_camada_saida)
    #         resultado_camada_saida.append(func_ativacao_camada_saida)
        
    #         # Erro da saída
    #         erro = resultado_esperado_atual - func_ativacao_camada_saida
    #         #media erro ao quadrado
    #         #zerar a cada epoca
    #         somatorio_erro += erro
    #         # Backpropagation
    #         #26 no delta
    #         delta_saida = erro * derivada(func_ativacao_camada_saida)
    #         # Correção de pesos da camada de saída
    #         #No outer criamos uma matriz de 60 linhas e 26 colunas para podermos ajustar
    #         pesos_camada_saida_correcao = taxa_aprendizado * np.outer(func_ativacao_camada_escondida, delta_saida)
   
    #         #pesos_camada_saida contem array de 26 pesos
    #         pesos_camada_saida += pesos_camada_saida_correcao

    #         # Cálculo do delta da camada escondida
    #         #pesos_camada_saida.T tem 26x60
    #         delta_escondida = np.dot(delta_saida, pesos_camada_saida.T) * derivada(func_ativacao_camada_escondida)
            
    #         # Correção de pesos da camada escondida
    #         pesos_camada_escondida_correcao = taxa_aprendizado * np.outer(treinamento_atual, delta_escondida)
    #         pesos_camada_escondida += pesos_camada_escondida_correcao

    #         resultado_somatoria_erro.append(somatorio_erro)
    #         resultado_acuracia.append(acuracia)
      
    #     # Calculando a acurácia (supondo uma tarefa de classificação binária)
    #     acuracia = (total_previsoes - somatorio_erro) / total_previsoes
        
    #     print(f'ERRO MÉDIO: {somatorio_erro / total_previsoes}')
    #     print(f'ACURÁCIA: {acuracia}')
    #     # Verificação da condição de parada (aqui deve ser ajustado conforme seu critério)
    #     #Neste exemplo, a condição de parada é baseada na soma do erro
    #     epoca_atual += 1
    while epoca_atual < max_epocas and not condicao_de_parada:
        somatorio_erro = 0
        acertos = 0
        total = len(train)

        for treinamento_atual, resultado_esperado_atual in zip(train, resultado_esperado):
            # Feedforward
            Ni_camada_escondida = np.dot(treinamento_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
            ativacao_camada_escondida = ativacao(Ni_camada_escondida)

            Ni_camada_saida = np.dot(ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
            ativacao_camada_saida = ativacao(Ni_camada_saida)

            # Calculo do erro
            erro = resultado_esperado_atual - ativacao_camada_saida
            somatorio_erro += np.mean(erro**2)

            # Avaliar acurácia
            indice_predito = np.argmax(ativacao_camada_saida)
            if resultado_esperado_atual[indice_predito] == 1:
                acertos += 1

            # Backpropagation
            delta_saida = erro * derivada(ativacao_camada_saida)
            delta_escondida = np.dot(delta_saida, pesos_camada_saida.T) * derivada(ativacao_camada_escondida)

            # Atualizações de pesos (SGD)
            correcao_saida = taxa_aprendizado * np.outer(ativacao_camada_escondida, delta_saida)
            correcao_escondida = taxa_aprendizado * np.outer(treinamento_atual, delta_escondida)
            
            pesos_camada_saida += correcao_saida
            pesos_camada_escondida += correcao_escondida

        erro_medio = somatorio_erro / total
        acuracia = acertos / total
        resultados_por_epoca.append((erro_medio, acuracia))
        if(erro_medio <= 0.01):
            condicao_de_parada = True
        print(f'Epoca {epoca_atual}, Erro Médio {erro_medio}, Acurácia {acuracia}, Acertos {acertos}')

        epoca_atual += 1
    return resultado_camada_escondida, resultado_camada_saida

def treina_com_parada_antecipada(train, resultado_esperado, validacao, resultado_validacao_esperada,tamanho_camadas_entrada,tamanho_camadas_escondida, tamanho_camada_saida, Bias_entrada_camada_escondida, Bias_entrada_camada_saida, max_epocas,taxa_aprendizado):
    ativacao, derivada = funcao_ativacao()
    epoca_atual = 0
    erro_validacao_anterior = 100000   
    resultados_por_epoca = []
    condicao_de_parada = False
    pesos_camada_escondida = inicializando_pesos(tamanho_camadas_entrada, tamanho_camadas_escondida)
    pesos_camada_saida = inicializando_pesos(tamanho_camadas_escondida,tamanho_camada_saida)
    
    resultado_camada_escondida = []
    resultado_camada_saida = []
    resultado_somatoria_erro = []
    resultado_acuracia = []

    while epoca_atual < max_epocas and not condicao_de_parada:
        somatorio_erro = 0
        somatorio_erro_validacao = 0
        acertos = 0
        acertos_validacao = 0
        total = len(train)

        for treinamento_atual, resultado_esperado_atual in zip(train, resultado_esperado):
            # Feedforward
            Ni_camada_escondida = np.dot(treinamento_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
            ativacao_camada_escondida = ativacao(Ni_camada_escondida)

            Ni_camada_saida = np.dot(ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
            ativacao_camada_saida = ativacao(Ni_camada_saida)

            # Calculo do erro
            erro = resultado_esperado_atual - ativacao_camada_saida
            somatorio_erro += np.mean(erro**2)

            # Avaliar acurácia
            indice_predito = np.argmax(ativacao_camada_saida)
            if resultado_esperado_atual[indice_predito] == 1:
                acertos += 1

            # Backpropagation
            delta_saida = erro * derivada(ativacao_camada_saida)
            delta_escondida = np.dot(delta_saida, pesos_camada_saida.T) * derivada(ativacao_camada_escondida)

            # Atualizações de pesos (SGD)
            correcao_saida = taxa_aprendizado * np.outer(ativacao_camada_escondida, delta_saida)
            correcao_escondida = taxa_aprendizado * np.outer(treinamento_atual, delta_escondida)
            
            pesos_camada_saida += correcao_saida
            pesos_camada_escondida += correcao_escondida

        erro_medio = somatorio_erro / total
        acuracia = acertos / total

        if(epoca_atual % 10 == 0):
            for validacao_atual, resultado_validacao_atual in zip(validacao, resultado_validacao_esperada):
                validacao_Ni_camada_escondida = np.dot(validacao_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
                validacao_ativacao_camada_escondida = ativacao(validacao_Ni_camada_escondida)

                validacao_Ni_camada_saida = np.dot(validacao_ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
                validacao_ativacao_camada_saida = ativacao(validacao_Ni_camada_saida)
                indice_predito_validacao = np.argmax(validacao_ativacao_camada_saida)
                
                if resultado_validacao_atual[indice_predito_validacao] == 1:
                    acertos_validacao += 1

                # Calculo do erro
                erro_validacao = resultado_esperado_atual - ativacao_camada_saida
                somatorio_erro_validacao += np.mean(erro_validacao**2)
            erro_medio_validacao = somatorio_erro_validacao / total
            acuracia_validacao = acertos / total
            print(f'Epoca {epoca_atual}, Erro Medio Anterior {erro_validacao_anterior} Erro Médio Validacao {erro_medio_validacao}, Acuracia Validacao {acuracia_validacao}, Acertos Validacao {acertos_validacao}')

            if(erro_validacao_anterior < erro_medio_validacao):
                print(f'Saida antecipada, erro de validacao: ')
                condicao_de_parada = True
            else:
                erro_validacao_anterior = erro_medio_validacao
        resultados_por_epoca.append((erro_medio, acuracia))

        epoca_atual += 1
    return resultado_camada_escondida, resultado_camada_saida