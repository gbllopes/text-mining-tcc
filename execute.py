import random
import unicodedata
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning,  module='gensim')
import csv
import nltk
from nltk import RegexpTokenizer, stem
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import GaussianNB
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')

stop_words = nltk.corpus.stopwords.words('portuguese')

#redutor da palavra ao seu radical
stemmer = stem.RSLPStemmer()

def preProcessarTexto(texto):
    texto_limpo = []

    for linha in texto:
        linha = linha.lower().strip()

        if linha != '':

          linha = unicodedata.normalize('NFD', linha).encode('ASCII', 'ignore').decode('UTF-8')

          listaDeTokens = tokenizer.tokenize(linha)

          listaDeTokens = list(set(listaDeTokens).difference(stop_words))

          for index in range(len(listaDeTokens)):
              listaDeTokens[index] = stemmer.stem(listaDeTokens[index])

          texto_limpo.append(listaDeTokens)

    return texto_limpo


def divisorHoldUp(dados):
    qtde_total = len(dados);
    dados_treino = []
    dados_validacao = []
    for index in range(qtde_total):
        if index < qtde_total * 0.75:
            dados_treino.append(dados[index])
        else:
            dados_validacao.append(dados[index])
    return dados_treino, dados_validacao


def treinar_modelo(documentos):
    max_epochs =  1000 # num de iteracões que vão percorrer o corpus.
    vec_size = 500  #Dimensão dos vetores de recursos.
    alpha = 0.01 #Setando taxa de apredizagem inicial.

    model = Doc2Vec(
        vector_size = vec_size,
        alpha = alpha,
        min_alpha = 0.00025, #taxa setada para queda de apredizagem a medida que o treinamento progride,
        min_count = 1, #Descarta do aprendizado palavras com frequência total menor que o valor setado.
        window = 20,
        dm = 1  #Define o algoritmo de treinamento. se dm == 1, 'memória distribuída' é utilizada, se não,  o pacote de palavras (PV-DBOW) é empregado.
    )

    # Cria o vocabulário
    model.build_vocab(documentos)

    for epoch in range(max_epochs):
        model.train(documentos,
                    total_examples = model.corpus_count,
                    epochs=model.iter
                    )
        model.alpha -= 0.0002

        model.min_alpha = model.alpha
    return model

def criarVetorWord2Vec(model, vetorPalavra):
    return model.infer_vector(vetorPalavra, steps=1000, alpha=0.01)


def classificar_naive_bayes(model, depressivas, nao_depressivas):
    #concatenando em tabela
    array = [[criarVetorWord2Vec(model, depressiva), 1] for depressiva in depressivas]
    array += [[criarVetorWord2Vec(model, nao_depressiva), 0] for nao_depressiva in nao_depressivas]

    # separador de array e label(teste)
    treino_array = []
    treino_labels = []

    for index in range(len(array)):
        treino_array.append(array[index][0])
        treino_labels.append(array[index][1])

    print(treino_array)
    # Função que treina o classificador Naive Bayes
    classificacao = GaussianNB()
    classificacao.fit(treino_array, treino_labels)

    return classificacao


data_set = open('./depressivas.txt', 'r', encoding="utf8")
depressivas = preProcessarTexto(data_set)
data_set.close()

data_set = open('./nao-depressivas.txt', 'r', encoding="utf8")
nao_depressivas = preProcessarTexto(data_set)
data_set.close()




treino_depressivas, validacao_depressivas = divisorHoldUp(depressivas)
treino_nao_depressivas, validacao_nao_depressivas = divisorHoldUp(nao_depressivas)

#Constroi uma label para cada vetor, com o formato que o algoritmo doc2vec aceita.
#documentos = [TaggedDocument(words=linha, tags=['0','NÃO_DEPRESSIVAS_'+str(index)]) for index, linha in enumerate(treino_nao_depressivas)]
#documentos += [TaggedDocument(words=linha, tags=['1','DEPRESSIVAS_'+str(index)]) for index, linha in enumerate(treino_depressivas)]

#executa treinamento
'''
model = treinar_modelo(documentos)
model.save('teste.model')
print("modelo salvo")
'''

model = Doc2Vec.load('d2v.model')
classificacao = classificar_naive_bayes(model, treino_depressivas, treino_nao_depressivas)


#### Precisão do Algoritmo ####

array_frases_depressivas = []
for frase in validacao_depressivas:
    array_frases_depressivas.append(criarVetorWord2Vec(model, frase))

print(array_frases_depressivas)

'''array_frases_nao_depressivas = []
for frase in validacao_nao_depressivas:
    array_frases_nao_depressivas.append(criarVetorWord2Vec(model, frase))
    
array_naive_bayes = []
for frase in array_frases_depressivas:
    precisao = '''


















