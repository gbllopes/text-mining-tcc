import unicodedata
import csv
import nltk
from nltk import RegexpTokenizer, stem

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




data_set = open('./depressivas.txt', 'r', encoding="utf8")
depressivas = preProcessarTexto(data_set)
data_set.close()

data_set = open('./nao-depressivas.txt', 'r', encoding="utf8")
nao_depressivas = preProcessarTexto(data_set)
data_set.close()

print(depressivas)

teste = input("Insira uma frase: \n")
print(preProcessarTexto([teste]))
