import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

def LSA(raw_overview_corpus,vectorizer,SVD):
	#criando a matriz Term Frequency - Inverse Document Frequency
    TfIdf_matrix = vectorizer.fit_transform(raw_overview_corpus)
	#print(TfIdf_matrix.shape) ==> (45466, 2374466) para corpus_movies.csv
	#decomposição da matriz tfidf nas matrizes USV (U, sigma,) pelo SVD
    U = SVD.fit_transform(TfIdf_matrix)
    return U

def gera_doc_MatrizTermoConceito(vectorizer,SVD):
	lista_termos = vectorizer.get_feature_names()
	lista_conceitos = []
	for i,conceitos in enumerate(SVD.components_):
	    termos_q_definem_conceito = zip(lista_termos,conceitos)
	    lista_termos_ordenado = sorted(termos_q_definem_conceito, key=lambda x: x[1], reverse=True) [0:10]
	    lista_conceitos.append(lista_termos_ordenado)

	Saida_fitted_to_csv = {'concept': [i for i in range(0,len(lista_conceitos))],
	                    'listas_de_termos': [[e[0] for e in lista ] for lista in lista_conceitos],
	                    'listas_de_valores_de_termos': [[e[1] for e in lista ] for lista in lista_conceitos]}
	data_to_file = pandas.DataFrame(Saida_fitted_to_csv, columns = ['concept','listas_de_termos','listas_de_valores_de_termos'])
	data_to_file.to_csv (r'concepts.csv', index = None, header=True)


def Filmes_similares(entrada_usuario,matriz_documento_conceito):
    distancia = euclidean_distances(entrada_usuario.reshape(1,-1),matriz_documento_conceito)
    pares = enumerate(distancia[0])
    similares_ordenados = sorted(pares, key = lambda item: item[1])[0:5]
    return similares_ordenados

def main():
    # recebendo corpus
    raw_corpus = pandas.read_csv('corpus_movies.csv', low_memory=True)
    raw_overview_corpus = raw_corpus['overview'].fillna('')
    # definição do conjunto de stopwords
    StopWords = set(stopwords.words('english'))
    StopWords.update(['overview'])
    # instanciando objetos
    vectorizer = TfidfVectorizer(stop_words=StopWords, use_idf=True, ngram_range=(1, 3))
    SVD = TruncatedSVD(n_components=35)
    #texto escrito pelo usuário
    entrada_usuario = "A comedy movie with friends travelling together and romance in background"

    U = LSA(raw_overview_corpus,vectorizer,SVD)
    gera_doc_MatrizTermoConceito(vectorizer,SVD)
    entrada_transformada = SVD.transform(vectorizer.transform([entrada_usuario]))[0]
    recomendacoes = Filmes_similares(entrada_transformada,U)
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

if __name__ == "__main__":
    main()