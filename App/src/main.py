#######################################################################
#                Inteligencia Artificial - ACH2016                    #
#                                                                     #
#  Gandhi Daiti Miyahara 11207773                                     #
#  Lucas Tatsuo Nishida 11208270                                      #
#  Juan Kineipe 11894610                                              #
#  Leonardo Ken Matsuda Cancela 11207665                              #
#  João de Araújo Barbosa da Silva 11369704                           #
#                                                                     #
#######################################################################

from Functions import functions_data as func_data
import numpy as np
from multilayer_perceptron import MultiLayerPerceptron, treina_sem_validacao_cruzada, treina_com_validacao_cruzada, teste_multilayer_perceptron
from Functions.functions_write import write_pesos_finais_camada_Escondida_txt, write_pesos_finais_camada_Saida_txt,write_arch_rede_mlp
import json 

if __name__ == "__main__":
    print()
    print("INTEGRANTES\n")
    print("--------------------------------------------------------------------------")
    print(" - Gandhi Daiti Miyahara - NUSP 11207773")
    print(" - Lucas Tatsuo Nishida - NUSP 11208270")
    print(" - Juan Kineipe - NUSP 11894610")
    print(" - Leonardo Ken Matsuda Cancela - NUSP 11207665")
    print(" - João de Araújo Barbosa da Silva - NUSP 11369704")
    print("--------------------------------------------------------------------------\n")

    saida = False
    
    while(not saida):
        print("\nMultilayer Perceptron (MLP)\n")


        rede_escolhida = int(input("Qual rede deseja escolher? (1/2) \n 1 - Sem validacao Cruzada \n 2 - Com validacao Cruzada \n 3 - Sair\n"))
        match rede_escolhida:
            case 1:
                print("\n1 - Sem validacao Cruzada\n")
            case 2:
                print("\n2 - Com validacao Cruzada\n")
            case 3:
                print("\nSaindo do programa... Obrigado!\n")
                saida = True
                break

        camada_escondida = int(input("Digite o numero de neuronios na camada escondida: \n"))
        taxa_aprendizado = float(input("Digite a taxa de aprendizado(Exemplo: 0.5): \n"))
        num_epocas = int(input("Digite o numero de epocas máximo: \n"))
        
        dictionary = {'TamanhoCamadaEntrada':120, 'TamanhoCamadaEscondida':camada_escondida, 'TamanhoCamadaSaida':26,
                'Epocas':num_epocas, 'TaxaDeAprendizado':taxa_aprendizado,'BiasCamadaSaida':1, 
                'BiasCamadaEntrada':1}
        
        Perceptron = MultiLayerPerceptron(dictionary)
        write_arch_rede_mlp(json.dumps(dictionary))
        print("\nGerando conjuntos de treinamento, teste e validacao(se necessario)...\n")
        train,resultado_treino_esperado, teste, teste_resultado_esperado, validacao, resultado_validacao_esperada = func_data.get_train_test()
        print(f"Dados de treinamento: Total de {len(train)} dados de treinamento\n")
        print(f"Dados de teste: Total de {len(teste)} dados de teste\n")
       
        if(rede_escolhida == 1):
            print("Dados de Validacao: não utilizado\n")
            print()
            print("Iniciando treinamento...\n")
            pesos_camada_escondida_sem_pa, pesos_camada_saida_sem_pa = treina_sem_validacao_cruzada(train, resultado_treino_esperado,Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida, Perceptron.tamanho_camadas_saida,Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.maximo_epocas, Perceptron.taxa_aprendizado)
            write_pesos_finais_camada_Escondida_txt(pesos_camada_escondida_sem_pa)
            write_pesos_finais_camada_Saida_txt(pesos_camada_saida_sem_pa)
            print("\nSalvando pesos em arquivos...\n")
            
            print("\nIniciando os testes...\n")
            teste_multilayer_perceptron(pesos_camada_escondida_sem_pa, pesos_camada_saida_sem_pa, teste, teste_resultado_esperado, Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.taxa_aprendizado)
        elif(rede_escolhida == 2):
            print(f"Dados de Validacao: Total de {len(validacao)}\n")
            print()
            print("Iniciando treinamento...\n")
            pesos_camada_escondida_com_pa, pesos_camada_saida_com_pa = treina_com_validacao_cruzada(train, resultado_treino_esperado,validacao,resultado_validacao_esperada, Perceptron.tamanho_camadas_entrada,Perceptron.tamanho_camadas_escondida, Perceptron.tamanho_camadas_saida,Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.maximo_epocas, Perceptron.taxa_aprendizado)
            write_pesos_finais_camada_Escondida_txt(pesos_camada_escondida_com_pa)
            write_pesos_finais_camada_Saida_txt(pesos_camada_saida_com_pa)
            print("\nIniciando os testes...\n")
            teste_multilayer_perceptron(pesos_camada_escondida_com_pa, pesos_camada_saida_com_pa, teste, teste_resultado_esperado, Perceptron.bias_entrada_camada_escondida, Perceptron.bias_saida_camada_escondida, Perceptron.taxa_aprendizado)
            