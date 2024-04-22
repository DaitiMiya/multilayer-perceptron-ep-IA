from Functions import funcions_data as func_data
from multilayer_perceptron import MultiLayerPerceptron, inicializando_pesos

if __name__ == "__main__":
        
    dictionary = {'TamanhoCamadaEntrada':120, 'TamanhoCamadaEscondida':60, 'TamanhoCamadaSaida':26,
                'Epocas':700, 'TaxaDeAprendizado':0.3,'BiasCamadaSaida':1, 
                'BiasCamadaEntrada':1}
    train,test = func_data.get_train_test()
    Perceptron = MultiLayerPerceptron(dictionary)
    #inicializando os pesos
    pesos_entrada_camada_escondida, pesos_saida_camada_escondida = inicializando_pesos(Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida,Perceptron.tamanho_camadas_saida)
    