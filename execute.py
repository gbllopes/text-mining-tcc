import unicodedata
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning,  module='gensim')
import csv
import nltk
from nltk import RegexpTokenizer, stem
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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


def treinar_modelo(tagged_data):
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
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples = model.corpus_count,
                    epochs=model.epochs
                    )
        model.alpha -= 0.0002

        model.min_alpha = model.alpha
    return model






data_set = open('./depressivas.txt', 'r', encoding="utf8")
depressivas = preProcessarTexto(data_set)
data_set.close()

data_set = open('./nao-depressivas.txt', 'r', encoding="utf8")
nao_depressivas = preProcessarTexto(data_set)
data_set.close()




treino_depressivas, validacao_depressivas = divisorHoldUp(depressivas)
treino_nao_depressivas, validacao_nao_depressivas = divisorHoldUp(nao_depressivas)


#Constroi uma label para cada vetor, com o formato que o algoritmo doc2vec aceita.
#tagged_data = [TaggedDocument(words=linha, tags=['0','NÃO_DEPRESSIVAS_'+str(index)]) for index, linha in enumerate(treino_nao_depressivas)]
#tagged_data += [TaggedDocument(words=linha, tags=['1','DEPRESSIVAS_'+str(index)]) for index, linha in enumerate(treino_depressivas)]

#executa treinamento
    #model = treinar_modelo(tagged_data)
    #model.save('d2v.model')
    #print("modelo salvo")

model = Doc2Vec.load('d2v.model')


def criarVetorWord2Vec():



def classificador_naive_bayes(model, treino_depressivas, treino_nao_depressivas):
    array = [[criarVetorWord2Vec()]]



teste = float(input("testando:"))
print("{:6.2f}% de chance ".format(teste * 100))














