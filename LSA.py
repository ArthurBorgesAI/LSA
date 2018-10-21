import pandas
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#definição do conjunto de stopwords
StopWords = set(stopwords.words('english'))
StopWords.update([])
#recebendo corpus
raw_corpus = pandas.read_csv('corpus_movies.csv',low_memory=True)
#modelando a entrada para ser utilizada pelo programa
raw_overview_corpus = raw_corpus['overview'].fillna('')
vectorizer = TfidfVectorizer(stop_words = StopWords, use_idf=True, ngram_range=(1,3))
#criando a matriz Term Frequency - Inverse Document Frequency
TfIdf_matrix = vectorizer.fit_transform(raw_overview_corpus)
#print(TfIdf_matrix.shape) ==> (45466, 2374466) para corpus_movies.csv

#decomposição da matriz tfidf nas matrizes USV pelo SVD
SVD = TruncatedSVD(n_components=25, n_iter=100)
SVD.fit(TfIdf_matrix)

#mapeamento dos conceitos(lista_conceitos[]) em conjunto de termos(lista_termos_ordenado)
lista_termos = vectorizer.get_feature_names()
lista_conceitos = []
for i,conceitos in enumerate(SVD.components_):
    termos_q_definem_conceito = zip(lista_termos,conceitos)
    lista_termos_ordenado = sorted(termos_q_definem_conceito, key=lambda x: x[1], reverse=True) [0:10]
    lista_conceitos.append(lista_termos_ordenado)

#impressão dos termos que constituem cada conceito
for i,instancia in enumerate(lista_conceitos):
    print ("conceito " + str(i))
    for elemento in instancia:
        print(elemento[0])

#exportando modelo para um arquivo CSV

    #criando o dataframe
lista_de_listas = [lista for lista in lista_conceitos]
Saida_fitted_to_csv = {'concept': [i for i in range(0,len(lista_conceitos))],
                    'listas_de_termos': [[e[0] for e in lista ] for lista in lista_conceitos],
                    'listas_de_valores_de_termos': [[e[1] for e in lista ] for lista in lista_conceitos]}
data_to_file = pandas.DataFrame(Saida_fitted_to_csv, columns = ['concept','listas_de_termos','listas_de_valores_de_termos'])
print("Data frame em csv")
print(data_to_file)
#salvando o dataframe em um arquivo.csv
export_csv = data_to_file.to_csv (r'concepts.csv', index = None, header=True)
