from Functions import funcions_data as func_data
import numpy as np
from multilayer_perceptron import MultiLayerPerceptron, treina_com_parada_antecipada, treina_sem_parada_antecipada, teste_multilayer_perceptron

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)  
    dictionary = {'TamanhoCamadaEntrada':120, 'TamanhoCamadaEscondida':30, 'TamanhoCamadaSaida':26,
                'Epocas':5000, 'TaxaDeAprendizado':0.02,'BiasCamadaSaida':1, 
                'BiasCamadaEntrada':1}
    train,resultado_treino_esperado, teste, teste_resultado_esperado, validacao, resultado_validacao_esperada = func_data.get_train_test()

    Perceptron = MultiLayerPerceptron(dictionary)

    #inicializando os pesos
    # pesos_entrada_camada_escondida, pesos_saida_camada_escondida = inicializando_pesos(Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida,Perceptron.tamanho_camadas_saida)

    # pesos_camada_escondida_sem_pa, pesos_camada_saida_sem_pa = treina_sem_parada_antecipada(train, resultado_treino_esperado,Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida, Perceptron.tamanho_camadas_saida,Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.maximo_epocas, Perceptron.taxa_aprendizado)
    pesos_camada_escondida_com_pa, pesos_camada_saida_com_pa = treina_com_parada_antecipada(train, resultado_treino_esperado,validacao,resultado_validacao_esperada, Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida, Perceptron.tamanho_camadas_saida,Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.maximo_epocas, Perceptron.taxa_aprendizado)
    
    # teste_multilayer_perceptron(pesos_camada_escondida_sem_pa, pesos_camada_saida_sem_pa, teste, teste_resultado_esperado, Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.taxa_aprendizado)
    teste_multilayer_perceptron(pesos_camada_escondida_com_pa, pesos_camada_saida_com_pa, teste, teste_resultado_esperado, Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.taxa_aprendizado)
