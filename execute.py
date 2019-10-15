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
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import os.path

def pre_process(data_set):
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('portuguese'))
    stemmer = stem.RSLPStemmer()
    new_data = []

    for phrase in data_set:
        phrase = phrase.strip()
        if phrase != '':
            new_str = unicodedata.normalize('NFD', phrase.lower() ).encode('ASCII', 'ignore').decode('UTF-8')       
            dlist = tokenizer.tokenize(new_str)
            dlist = list(set(dlist).difference(stopword_set))
            print(dlist)
            for s in range(len(dlist)):
                dlist[s] = stemmer.stem(dlist[s])
            new_data.append(dlist)
    return new_data

def share_base(data):
    amount_total = len(data)
    percentage_train = 0.75
    train = []
    validation = []
    for index in range(0, amount_total):
        if index < amount_total * percentage_train:
            train.append(data[index])
        else:
            validation.append(data[index])
    return train, validation

#Função que treina o modelo Doc2Vec
def train_model(tagged_data):
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

def generate_vector(model, wordVector):
    return model.infer_vector(wordVector, steps=1000, alpha=0.01)

# Gera o classificador ( Naive Bayes )
def generate_classifier(model, depressive, non_depressive):
    array =  [[generate_vector(model, depressive), 1] for depressive in depressive]
    array += [[generate_vector(model, non_depressive), 0] for non_depressive in non_depressive]
    random.shuffle(array)
    train_array = []
    train_labels = []
    for index in range(len(array)):
        train_array.append(array[index][0])
        train_labels.append(array[index][1])
    # Treina Naive Bayes
    classifier = GaussianNB()
    classifier.fit(train_array, train_labels)
    return classifier

def print_metrics(vector_expected, vector_results):
    precision = metrics.precision_score(vector_expected, vector_results)
    print("Taxa de precisão: {:6.2f}%".format(precision * 100))

    accuracy = metrics.accuracy_score(vector_expected, vector_results)
    print("Taxa de acurácia: {:6.2f}%".format(accuracy * 100))

# Treina um novo modelo caso necessário, carrega e retorna
def load_model(name_model, tagged_data):
    if os.path.exists(name_model+'.model'):
        model = Doc2Vec.load(name_model+'.model')
        print('Modelo carregado')
    else:
        model = train_model(tagged_data)
        model.save(name_model+'.model')
        print('Novo modelo treinado e salvo')
    return model

def get_dataset(name_dataset):
    data_set = open(name_dataset+'.txt', 'r')
    return data_set

if __name__ == '__main__':
    data_set = get_dataset('depressive')
    depressive = pre_process(data_set)
    data_set.close()

    data_set = get_dataset('non_depressive')
    non_depressive = pre_process(data_set)
    data_set.close()

    train_depressive, validation_depressive = share_base(depressive)
    train_non_depressive, validation_non_depressive = share_base(non_depressive)

    # Gera o rótulo para o vetor, padrão atribuição do Doc2Vec
    tagged_data = [TaggedDocument(words=linha, tags=['0','NÃO_DEPRESSIVA_'+str(index)]) for index, linha in enumerate(train_non_depressive)]
    tagged_data += [TaggedDocument(words=linha, tags=['1','DEPRESSIVA_'+str(index)]) for index, linha in enumerate(train_depressive)]

    model = load_model('Depression_Model', tagged_data)

    # Gera classificador
    classifier_naive_bayes = generate_classifier(model, train_depressive, train_non_depressive)

    vectors_phrase_depressive = []
    for phrase in validation_depressive:
        vectors_phrase_depressive.append(generate_vector(model, phrase))

    vectors_non_depressive_phrase = []
    for phrase in validation_non_depressive:
        vectors_non_depressive_phrase.append(generate_vector(model, phrase))

    vector_result_expected = [1 for i in range(len(vectors_phrase_depressive))]
    vector_result_expected += [0 for i in range(len(vectors_non_depressive_phrase))]

    #Atribuição dos valores aos vetores 
    vector_naive_bayes = []
    for phrase in vectors_phrase_depressive:
        result = classifier_naive_bayes.predict_proba([phrase])
        if result[0][0] < result[0][1]:
            vector_naive_bayes.append(1)
        else:
            vector_naive_bayes.append(0)
    for phrase in vectors_non_depressive_phrase:
        result = classifier_naive_bayes.predict_proba([phrase])
        if result[0][0] < result[0][1]:
            vector_naive_bayes.append(1)
        else:
            vector_naive_bayes.append(0)

    print("\n\n####  Naive Bayes  ####")
    print_metrics(vector_result_expected, vector_naive_bayes)

    while(1):
        phrase = input('Informe uma frase para classificação: ')
        print('\nFrase informada: \"' + phrase + '\"')
        phrase = pre_process([phrase])
        print('\nFrase após tratamento:')
        print(phrase)
        vector = generate_vector(model, phrase[0])
        print('Vetor gerado:')
        print(vector)
        result = classifier_naive_bayes.predict_proba([vector])
        print('\nResultado: ')
        print(' * {:6.2f}% de chance da frase possuir características depressivas'.format(result[0][1]     * 100))
        print(' * {:6.2f}% de chance da frase não possuir características depressivas'.format(result[0][0] * 100))
        print('\n * frase com características depressivas' if result[0][0] < result[0][1] else '\n * frase sem características depressivas!')

        if (input('\n\nDeseja classificar outra frase?(sim)(não)\n') == 'não'):
            break
