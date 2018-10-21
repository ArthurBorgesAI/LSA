import pandas
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

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
SVD = TruncatedSVD(n_components=25)
matriz_documento_conceito = SVD.fit_transform(TfIdf_matrix)


#mapeamento dos conceitos(lista_conceitos[]) em conjunto de termos(lista_termos_ordenado)
lista_termos = vectorizer.get_feature_names()
lista_conceitos = []
for i,conceitos in enumerate(SVD.components_):
    termos_q_definem_conceito = zip(lista_termos,conceitos)
    lista_termos_ordenado = sorted(termos_q_definem_conceito, key=lambda x: x[1], reverse=True) [0:10]
    lista_conceitos.append(lista_termos_ordenado)




def Filmes_similares(entrada_usuario,matriz_documento_conceito):
    distancia = euclidean_distances(entrada_usuario.reshape(1,-1),matriz_documento_conceito)
    pares = enumerate(distancia[0])
    similares_ordenados = sorted(pares, key = lambda item: item[1])[0:5]
    return similares_ordenados












#exportando modelo para um arquivo CSV

    #criando o dataframe
lista_de_listas = [lista for lista in lista_conceitos]
Saida_fitted_to_csv = {'concept': [i for i in range(0,len(lista_conceitos))],
                    'listas_de_termos': [[e[0] for e in lista ] for lista in lista_conceitos],
                    'listas_de_valores_de_termos': [[e[1] for e in lista ] for lista in lista_conceitos]}
data_to_file = pandas.DataFrame(Saida_fitted_to_csv, columns = ['concept','listas_de_termos','listas_de_valores_de_termos'])

data_to_file.to_csv (r'concepts.csv', index = None, header=True)


#printando o overview das 5 recomendações
entrada_usuario = "A film about love sex drugs"
entrada_transformada = SVD.transform(vectorizer.transform([entrada_usuario]))[0]
recomendacoes = Filmes_similares(entrada_transformada,matriz_documento_conceito)
id_documento,similaridade = recomendacoes[0]
id_documento1,similaridade = recomendacoes[1]
id_documento2,similaridade = recomendacoes[2]
id_documento3,similaridade = recomendacoes[3]
id_documento4,similaridade = recomendacoes[4]

print(raw_overview_corpus[id_documento])
print(raw_overview_corpus[id_documento1])
print(raw_overview_corpus[id_documento2])
print(raw_overview_corpus[id_documento3])
print(raw_overview_corpus[id_documento4])