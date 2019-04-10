'''
MLBank v.1
---------------
Testes de algortimos de Ml na base:
https://archive.ics.uci.edu/ml/datasets/bank+marketing

@author: Thiago Serra F. Carvalho (thiagonce at gmail.com)
'''
import time
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


print("---------------------------------------------------------------")
print(">> O arquivo bank-full.csv deve estar na mesma pasta do projeto! ")
print("---------------------------------------------------------------")
caminho = "bank-full.csv"


class ML:

    #semente:   usada pela funcao train_test_split para gerar um numero aleatorio de partida
    #           para distribuicao dos conjutos de treinamento e teste
    global semente

    '''
    metodo aguarde(tempo, mensagem)
        responsavel apenas por escrever na tela uma mensagem para sabermos
        em que fase está nossa execução
    '''
    def aguarde(self, tempo, texto):
        print("[i] " + texto + "...Aguardando...")
        time.sleep(tempo)

    '''
    metodo formatarTimer(start, end)
        exibe o tempo decorrido a partir de um tempo incial e final
    '''
    def formatarTimer(self, start,end):
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.5f}".format(int(hours),int(minutes),seconds)


    '''
    metodo core()
        nosso programa principal que ira executar a carga e tratamento dos dados e
        criacao dos cojuntos de treinamento e teste
    '''
    def core(self):

        #semente: setando um valor inicial aleatório...um dado de partida apenas.
        self.semente = 5

        #percValidacao: percentual do nosso dataset que sera usado para validacao do modelo
        percValidacao = 0.20

        self.aguarde(2, "1 - Lendo o arquivo csv")
        #dataset: vamos amazenar os dados do arquivo csv, usamos o pandas para ler o arquivo
        dataset = pd.read_csv(caminho, delimiter=';')

        #vetor: vamos extrair do dataset somente os valores
        vetor = dataset.values

        # Nosso arquivo possui 17 colunas
        # As colunas no slice do Python sempre começam do 'Zero'
        # Assim, vamos guardar os dados (as primerias 16 colunas) no vetor X
        # Tecnicamente estamos separando as entidades dos resultados (coluna 17 indica se o cliente aceitou ou não: y/n)
        X = vetor[:,0:-1]   # entidades
        Y = vetor[:,16]     # respostas (y/n) -> lembre-se do conceito do artigo anterior de aprendizagem supervisionada


        self.aguarde(2, "2 - Convertando classes em números")
        #Como os algoritmos só trabalham com números, não podemos informar estado civil, profissao, escolaridade... no formato texto
        #Usaremos então um encoder (LabelEncoder do sklearn) para converter cada classe em número
        fitData = LabelEncoder()

        #Vamos percorrer as 16 colunas (* como a primeira e idade, ja e um numero inteiro) e
        #Fazer traducao do texto para um numero
        #O LabelEncoder vai fazer, por exemplo,  'male' em 1 e 'female' em 2... 'married' em 1, 'single' em 2, 'divorced' em 3...
        #acho que voce ja entendeu.... :-)
        for x in range(16):
            X[:, x] = fitData.fit_transform(X[: ,x])

        self.aguarde(2, "3 - Criando conjuntos de treino e teste")
        #Agora vamos a divisão.
        #Vamos criar 4 conjuntos para rodar nossos modelos (usando o model_selection do sklearn):
        #
        #  treino_X = conjunto de treino para os modelos (contendo 80% dos registros do nosso arquivo) = caracteristicas ou features
        #  validacao_X = conjunto de respostas (y/n) para os dados de treino = classes / labels
        #  treino_Y = conjunto de dados para teste do aprendizado (contendo 20% dos registros do nosso arquivo)
        #  validacao_Y = conjunto de respostas (y/n) para massa de testes = classes / labels
        treino_X, validacao_X, treino_Y, validacao_Y = model_selection.train_test_split(X, Y, test_size=percValidacao, random_state=self.semente)

        self.aguarde(2, "4 - Testando os modelos com os dados do problema")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #chama o metodo que ira rodar os dados
        self.testarDesempenho(treino_X, treino_Y)
        #metodo que roda apenas o modelo escolhido
        self.aguarde(2, "5 - Rodando o melhor modelo para predizer os resultados e verificar acertos")
        self.rodarModelo(treino_X, treino_Y, validacao_X, validacao_Y)

    '''
    metodo aprender(treino_X, treino_Y)
        responsavel por rodar os modelos escolhidos na nossa base de treinamento
    '''
    def testarDesempenho(self, treino_X, treino_Y):
        self.semente = 5
        scoring = 'accuracy'
        resultados = []
        algor = []

        #modelos: matriz contendo descricao e chamada para cada modelo que sera testado
        #Cada modelo foi brevemente explicado no texto...
        modelos = []
        modelos.append(('Gaussian NB', GaussianNB()))
        modelos.append(('Decision Tree Classifier', DecisionTreeClassifier()))
        modelos.append(('K-Neighbors Classifier', KNeighborsClassifier()))
        #modelos.append(('SVM', SVC(gamma='auto'))) (no artigo eu explico porque nao rodamos este algoritmo)
        modelos.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
        modelos.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
        modelos.append(('MLP Classifier', MLPClassifier()))
        modelos.append(('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()))

        for descricao, modelo in modelos:
            # Para fazer o teste de cada modelo, vamos dividir nosso conjunto em 10 subconjuntos (ou particoes = fold)
            kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state=self.semente)

            # A validacao cruzada serve para nos indicar como o modelo reage a um dado que ele desconhece
            # Os dados de treinamento estao em X e, os dados desconhecidos pelo modelo estao em Y
            # essa funcao vai retornar uma matriz de pontuacoes do estimador para cada execucao da validacao cruzada
            start = time.time()
            cross = model_selection.cross_val_score(modelo, treino_X, treino_Y, cv=kfold, scoring=scoring)
            resultados.append(cross)
            algor.append(descricao)
            # Vamos mostrar entao para cada modelo, a media dessas pontuacoes e tambem o desvio padrao para sabermos
            # o quanto cada um esta  generalizando (ou seja, sua aderencia ao problema)
            txtResultado = "%s: %f (%f)" % (descricao, cross.mean(), cross.std())
            print(txtResultado)
            print('[i] Tempo decorrido: ', self.formatarTimer(start, time.time()))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")


    '''
    metodo rodarModelo(treino_X, treino_Y, validacao_X, validacao_Y)
        vamos agora tentar predizer os resultados e verificar a acuracia do modelo
    '''
    def rodarModelo(self, treino_X, treino_Y, validacao_X, validacao_Y):
        print("-----------------------------------------------------")
        print("Obtivemos o melhor resultado neste conjunto com o modelo")
        print(" >> Logistic Regression ")
        print("Vamos agora testar esse modelo fazendo algumas predicoes!")
        print("-----------------------------------------------------")
        # Classificador KNeighbors implementando os k-vizinhos mais próximos votar
        rede = LogisticRegression(solver='liblinear', multi_class='ovr')
        rede.fit(treino_X, treino_Y)
        predictions = rede.predict(validacao_X)
        print("-----------------------------------------------------")
        self.aguarde(1, "Acurácia: " + str(accuracy_score(validacao_Y, predictions)*100) + "%" )
        self.aguarde(1, "Matriz de confusão")
        print(confusion_matrix(validacao_Y, predictions))
        self.aguarde(1, "Relatório de uso do modelo")
        print(classification_report(validacao_Y, predictions))
        print("-----------------------------------------------------")

# nosso programinha...
# :-)
print("----------------------- Machine Learning - Testes v.1")
print("-----------------------------------------------------")
print("-------------------------------------------- INICIO -")
inicio = ML()
inicio.core()
print("----------------------------------------------- FIM -")
