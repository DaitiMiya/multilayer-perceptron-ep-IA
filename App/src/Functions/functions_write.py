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
def write_pesos_iniciais_camada_Escondida_txt(pesos):
    f = open("Resultados/pesos_iniciais(camadaEscondida).txt", "w")
    for peso_iniciais in pesos:
        f.write(f"{peso_iniciais}\n")
    f.close()
    
def write_pesos_iniciais_camada_Saida_txt(pesos):
    f = open("Resultados/pesos_iniciais(camadaSaida).txt", "w")
    for peso_iniciais in pesos:
        f.write(f"{peso_iniciais}\n")
    f.close()

def write_pesos_finais_camada_Escondida_txt(pesos):
    f = open("Resultados/pesos_finais(camadaEscondida).txt", "w")
    for peso_finais in pesos:
        f.write(f"{peso_finais}\n")
    f.close()

def write_pesos_finais_camada_Saida_txt(pesos):
    f = open("Resultados/pesos_finais(camadaSaida).txt", "w")
    for peso_finais in pesos:
        f.write(f"{peso_finais}\n")
    f.close()

def write_arch_rede_mlp(dictionary):
    f = open("Resultados/hiperparametros(MLP).txt", "w")
    f.write(dictionary)
    f.close()


def write_erro_por_epoca(resultado_erro):
    epoca = 0
    f = open("Resultados/erroPorEpoca.txt", "w")
    for erro_medio in zip(resultado_erro):
        epoca+=1
        f.write(f'Epoca {epoca}, Erro Medio {erro_medio}+ \n')
    f.close()


def write_resultado_rede(resultado_rede):
    f = open("Resultados/Resultado_teste.txt", "w")
    for resultado_teste in resultado_rede:
        f.write(str(resultado_teste) + '\n')
    f.close()