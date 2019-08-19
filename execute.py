import unicodedata

import nltk
from nltk import RegexpTokenizer, stem

tonkenizer = RegexpTokenizer(r'\w+')

stop_words = nltk.corpus.stopwords.words('portuguese')

#redutor da palavra ao seu radical
stemmer = stem.RSLPStemmer()

def preProcessarTexto(texto):
    novo_texto = []

    for frase in texto:
        frase = frase.strip()

        if frase != '':
            frase_lower = unicodedata.normalize('NFD', frase.lower()).encode('ASCII', 'ignore').decode('UTF-8')
            print(frase_lower)


preProcessarTexto(input("Digite uma frase \n"))
