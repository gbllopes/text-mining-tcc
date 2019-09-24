# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer, stem
from nltk.corpus import stopwords
import unicodedata
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def pre_processar(data_set):
    new_data = []
    
    for frase in data_set:
        frase = frase.strip()

        if frase != '':

            new_str = unicodedata.normalize('NFD', frase.lower() ).encode('ASCII', 'ignore').decode('UTF-8')       

            dlist = tokenizer.tokenize(new_str)

            dlist = list(set(dlist).difference(stopword_set))
            
            for s in range(len(dlist)):
                dlist[s] = stemmer.stem(dlist[s])
            
            new_data.append(dlist)
    return new_data

def dividir_base(dados):
    quantidade_total = len(dados)
    percentual_treino = 0.75
    treino = []
    validacao = []

    for indice in range(0, quantidade_total):
        if indice < quantidade_total * percentual_treino:
            treino.append(dados[indice])
        else:
            validacao.append(dados[indice])

    return treino, validacao

#Função que treina o modelo Doc2Vec
def treinar_modelo(tagged_data):
    max_epochs = 100 # Número de iterações sobre o corpus.
    vec_size = 20 # Dimensão dos vetores de recursos.
    alpha = 0.025 # Taxa de apredizagem inicial.

    model = Doc2Vec(
        vector_size=vec_size,
        alpha=alpha, 
        min_alpha=0.00025, # #taxa setada para queda de apredizagem a medida que o treinamento progride
        min_count=1,# Descarta do aprendizado palavras com frequência total menor que o valor setado.
        window = 20,
        dm =1) # Define o algoritmo de treinamento. Se dm = 1 , 'memória distribuída' (PV-DM) é usada. Caso contrário, o pacote distribuído de palavras (PV-DBOW) é empregado.

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)

        model.alpha -= 0.0002

        model.min_alpha = model.alpha
    return model

def gerar_vetor(model, wordVector):
    return model.infer_vector(wordVector, steps=1000, alpha=0.01)

# Gera o classificador ( Naive Bayes )
def gerar_classificador(model, depressivas, nao_depressivas):
    array =  [[gerar_vetor(model, depressiva), 1] for depressiva in depressivas]
    array += [[gerar_vetor(model, nao_depressiva), 0] for nao_depressiva in nao_depressivas]

    random.shuffle(array)

    train_array = []
    train_labels = []

    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])
        
    # Treina Naive Bayes
    classificador = GaussianNB()
    classificador.fit(train_array, train_labels)

    return classificador

def imprime_metricas(vetor_esperado, vetor_resultados):
    precisao = metrics.precision_score(vetor_esperado, vetor_resultados)
    print("Taxa de precisão: {:6.2f}%".format(precisao * 100))

    acuracia = metrics.accuracy_score(vetor_esperado, vetor_resultados)
    print("Taxa de acurácia: {:6.2f}%".format(acuracia * 100))

if __name__ == '__main__':
    tokenizer = RegexpTokenizer(r'\w+')

    stopword_set = set(stopwords.words('portuguese'))

    stemmer = stem.RSLPStemmer()

    data_set = open('depressivas.txt', 'r')
    depressiva = pre_processar(data_set)
    data_set.close()

    data_set = open('nao_depressivas.txt', 'r')
    nao_depressiva = pre_processar(data_set)
    data_set.close()

    treino_depressiva, validacao_depressiva = dividir_base(depressiva)
    treino_nao_depressiva, validacao_nao_depressiva = dividir_base(nao_depressiva)

    # Cria a label para o vetor. Esta é a forma que o doc2vec aceita
    tagged_data = [TaggedDocument(words=linha, tags=['0','NÃO_DEPRESSIVA_'+str(index)]) for index, linha in enumerate(treino_nao_depressiva)]
    tagged_data += [TaggedDocument(words=linha, tags=['1','DEPRESSIVA_'+str(index)]) for index, linha in enumerate(treino_depressiva)]

    # Inicializa e treina modelo
    # model = treinar_modelo(tagged_data)
    # model.save("doc2vec.model")
    # print("Model Saved")

    # Carrega o modelo já treinado.
    model= Doc2Vec.load("doc2vec.model")

    # Gera classificador
    classificador_naive_bayes       = gerar_classificador(model, treino_depressiva, treino_nao_depressiva)

    vetores_frases_depressivas = []
    for frase in validacao_depressiva:
        vetores_frases_depressivas.append(gerar_vetor(model, frase))

    vetores_frases_nao_depressivas = []
    for frase in validacao_nao_depressiva:
        vetores_frases_nao_depressivas.append(gerar_vetor(model, frase))

    vetor_resultado_esperado = [1 for i in range(len(vetores_frases_depressivas))]
    vetor_resultado_esperado += [0 for i in range(len(vetores_frases_nao_depressivas))]

    #Atribuição dos valores aos vetores 
    vetor_naive_bayes = []
    for frase in vetores_frases_depressivas:
        resultado = classificador_naive_bayes.predict_proba([frase])
        if resultado[0][0] < resultado[0][1]:
            vetor_naive_bayes.append(1)
        else:
            vetor_naive_bayes.append(0)
    for frase in vetores_frases_nao_depressivas:
        resultado = classificador_naive_bayes.predict_proba([frase])
        if resultado[0][0] < resultado[0][1]:
            vetor_naive_bayes.append(1)
        else:
            vetor_naive_bayes.append(0)

    print("\n\n####  Naive Bayes  ####")
    imprime_metricas(vetor_resultado_esperado, vetor_naive_bayes)

    while(1):
        frase = input('Favor informar frase a ser testada: ')
        print('\nFrase a ser testada: \"' + frase + '\"')
        frase = pre_processar([frase])
        print('\nFrase após tratamento:')
        print(frase)
        vetor = gerar_vetor(model, frase[0])
        print('Vetor gerado a partir da frase:')
        print(vetor)
        resultado = classificador_naive_bayes.predict_proba([vetor])
        print('\nResultado: ')
        print(' * {:6.2f}% de chance da frase possuir característica depressiva'.format(resultado[0][1]     * 100))
        print(' * {:6.2f}% de chance da frase não possuir característica depressiva'.format(resultado[0][0] * 100))
        print('\n * Frase com característica depressiva' if resultado[0][0] < resultado[0][1] else '\n * Frase sem característica depressiva!')

        if (input('\n\nDeseja testar uma nova frase?(sim)(não)\n') == 'não'):
            break
