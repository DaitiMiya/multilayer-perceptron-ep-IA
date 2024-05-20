from Functions import funcions_data as func_data
import numpy as np
from multilayer_perceptron import MultiLayerPerceptron, inicializando_pesos, treina_com_parada_antecipada

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)  
    dictionary = {'TamanhoCamadaEntrada':120, 'TamanhoCamadaEscondida':25, 'TamanhoCamadaSaida':26,
                'Epocas':200, 'TaxaDeAprendizado':0.02,'BiasCamadaSaida':1, 
                'BiasCamadaEntrada':1}
    train,resultado_treino_esperado, test, teste_resultado_esperado = func_data.get_train_test()
    validacao, resultado_validacao_esperada = func_data.get_validation_data()
    
    Perceptron = MultiLayerPerceptron(dictionary)


   
    #inicializando os pesos
    # pesos_entrada_camada_escondida, pesos_saida_camada_escondida = inicializando_pesos(Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida,Perceptron.tamanho_camadas_saida)

    x,y = treina_com_parada_antecipada(train, resultado_treino_esperado,validacao,resultado_validacao_esperada, Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida, Perceptron.tamanho_camadas_saida,Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.maximo_epocas, Perceptron.taxa_aprendizado)