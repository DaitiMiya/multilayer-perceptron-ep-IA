import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

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
       
        resultado_somatoria_erro.append(erro_medio)
        if(erro_medio <= 0.0001):
            condicao_de_parada = True
        print(f'Epoca {epoca_atual}, Erro Médio {erro_medio}, Acurácia {acuracia}, Acertos {acertos}')

        epoca_atual += 1
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(epoca_atual)), resultado_somatoria_erro, label='Acurácia Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Erro medio')
    plt.title('Erro medio por Época')
    plt.legend()
    plt.show()
    return pesos_camada_escondida, pesos_camada_saida

def treina_com_parada_antecipada(train, resultado_esperado, validacao, resultado_validacao_esperada,tamanho_camadas_entrada,tamanho_camadas_escondida, tamanho_camada_saida, Bias_entrada_camada_escondida, Bias_entrada_camada_saida, max_epocas,taxa_aprendizado):
    ativacao, derivada = funcao_ativacao()
    epoca_atual = 0

    erro_medio_validacao_anterior = 100000   
    resultados_por_epoca = []
    condicao_de_parada = False
    pesos_camada_escondida = inicializando_pesos(tamanho_camadas_entrada, tamanho_camadas_escondida)
    pesos_camada_saida = inicializando_pesos(tamanho_camadas_escondida,tamanho_camada_saida)

    resultado_camada_escondida = []
    resultado_camada_saida = []
    resultado_somatoria_erro = []
    acuracias_treinamento = []
    acuracias_validacao = []
    contador_paciente = 0
    while epoca_atual < max_epocas and not condicao_de_parada:
        erro_validacao = 0
        somatorio_erro = 0
        acertos = 0
        acertos_validacao = 0

        total = len(train)
        total_validaao = len(validacao)
        for treinamento_atual, resultado_esperado_atual in zip(train, resultado_esperado):
            # Feedforward
            # print()
            # print(f'TREINAMENTO ATUAL: {treinamento_atual}')
            Ni_camada_escondida = np.dot(treinamento_atual , pesos_camada_escondida) + Bias_entrada_camada_escondida
            # print(f'NEURONIO CAMADA ESCONDIDA: {Ni_camada_escondida}')
            ativacao_camada_escondida = ativacao(Ni_camada_escondida)
            # print(f'ATIVACAO CAMADA ESCONDIDA: {ativacao_camada_escondida}')
            Ni_camada_saida = np.dot(ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
            # print()
            # print(f'NEURONIO CAMADA SAIDA: {ativacao_camada_escondida}')
            ativacao_camada_saida = ativacao(Ni_camada_saida)
            # print(f'ATIVACAO CAMADA SAIDA: {ativacao_camada_escondida}')

            # Calculo do erro
            erro = resultado_esperado_atual - ativacao_camada_saida
            # print()
            # print(f'ERRO: {erro}')
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
        resultado_somatoria_erro.append(erro_medio)
        acuracia = acertos / total
        for validacao_atual, resultado_validacao_atual in zip(validacao, resultado_validacao_esperada):
            validacao_Ni_camada_escondida = np.dot(validacao_atual, pesos_camada_escondida) + Bias_entrada_camada_escondida
            validacao_ativacao_camada_escondida = ativacao(validacao_Ni_camada_escondida)

            validacao_Ni_camada_saida = np.dot(validacao_ativacao_camada_escondida, pesos_camada_saida) + Bias_entrada_camada_saida
            validacao_ativacao_camada_saida = ativacao(validacao_Ni_camada_saida)
            indice_predito_validacao = np.argmax(validacao_ativacao_camada_saida)
            
            if resultado_validacao_atual[indice_predito_validacao] == 1:
                acertos_validacao += 1

            # Calculo do erro
            erro_validacao += mse(resultado_validacao_atual,validacao_ativacao_camada_saida)
            
        erro_medio_validacao = erro_validacao / total_validaao
        acuracia_validacao = acertos_validacao / total_validaao
        print(f'Epoca {epoca_atual}, Erro Medio  {erro_medio_validacao} Erro Medio Anterior {erro_medio_validacao_anterior}, Acuracia Validacao {acuracia_validacao}, Acertos Validacao {acertos_validacao}')
        acuracias_treinamento.append(acuracia)
        acuracias_validacao.append(acuracia_validacao)
        epoca_atual += 1
        
        #paciencias
        #erro mais
        if((erro_medio_validacao_anterior) < erro_medio_validacao):
            contador_paciente += 1
            print(f'--CONTADOR: {contador_paciente}--')
        else:
            erro_medio_validacao_anterior = erro_medio_validacao
            contador_paciente = 0

        if(contador_paciente == 30): break
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(epoca_atual)), acuracias_treinamento, label='Acurácia Treinamento')
    plt.plot(list(range(epoca_atual)), acuracias_validacao, label='Acurácia Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.title('Acurácia de Treinamento e Validação por Época')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(epoca_atual)), resultado_somatoria_erro, label='Acurácia Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Erro medio')
    plt.title('Erro medio por Época')
    plt.legend()
    plt.show()
    return pesos_camada_escondida, pesos_camada_saida

def matriz_confusao(teste_predito, teste_esperado):
    matriz = np.zeros((26,26), dtype=int)
    acertos = 0
    index = 0

    for predito, esperado in zip(teste_predito, teste_esperado):
        indice_predito = np.argmax(predito)
        indice_desejado = np.argmax(esperado)
        matriz[indice_desejado][indice_predito] += 1
        
    for linha in matriz:
        acertos+=linha[index]
        index += 1
        print(linha)
    print(acertos)
    return matriz


def teste_multilayer_perceptron(pesos_camada_escondida, pesos_camada_saida, matriz_teste, teste_resultado_esperado, bias_entrada_camada_escondida, bias_saida_camada_escondida, taxa_aprendizado):
    ativacao, derivada = funcao_ativacao()
    somatorio_erro = 0
    acertos = 0
    total = len(matriz_teste)
    resultado_predito = []

    for treinamento_atual, resultado_esperado_atual in zip(matriz_teste, teste_resultado_esperado):
        # Feedforward
        Ni_camada_escondida = np.dot(treinamento_atual, pesos_camada_escondida) + bias_entrada_camada_escondida
        ativacao_camada_escondida = ativacao(Ni_camada_escondida)

        Ni_camada_saida = np.dot(ativacao_camada_escondida, pesos_camada_saida) + bias_saida_camada_escondida
        ativacao_camada_saida = ativacao(Ni_camada_saida)

        # Calculo do erro
        erro = resultado_esperado_atual - ativacao_camada_saida
        somatorio_erro += np.mean(erro**2)

        # Avaliar acurácia
        indice_predito = np.argmax(ativacao_camada_saida)
        predito = np.zeros(26, dtype=int)
        predito[indice_predito] = 1
        resultado_predito.append(predito)
        if resultado_esperado_atual[indice_predito] == 1:
            acertos += 1
            
    acuracia_teste= acertos / total
    matriz = matriz_confusao(resultado_predito, teste_resultado_esperado)
    print(f'ACURACIA: {acuracia_teste}')
    return True
    