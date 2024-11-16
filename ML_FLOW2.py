import pandas as pd
import numpy as np
import language_tool_python
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
#import fasttext ## embedings
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.decomposition import PCA

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import stanza
import re
from nltk.corpus import stopwords
import en_core_web_sm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

#"" Número de stopwords en nltk
stop_nltk = stopwords.words('english')
print("nltk :",len(stop_nltk))

## Número de stopwords en spacy
nlp = en_core_web_sm.load()
stop_spacy = nlp.Defaults.stop_words
print("spacy:", len(stop_spacy))
stop_todas = list(stop_spacy.union(set(stop_nltk)))

from nltk.stem import PorterStemmer

def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text): #  gensim.utils.simple_preprocess tokeniza el texto
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import spacy
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ML_FLOW_PARCIAL2:
    def __init__(self):
        self.path = "/content/drive/MyDrive/QUIZ1_3CORTE_IA/SegundoPunto/"

    def load_data(self):
        self.datos = pd.read_pickle('processed_q.pkl')

        return self.datos

    def preprocessing(self):
        data = self.datos.get(['essay_text','essay_id','processed_text'])

        def lemmatize_stemming(text):
            ps = PorterStemmer()
            return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

        def preprocess(text):
            result = []
            for token in gensim.utils.simple_preprocess(text): #  gensim.utils.simple_preprocess tokeniza el texto
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    result.append(lemmatize_stemming(token))
            return result

        data["essay_text"] = data["processed_text"].copy()
        data = data.drop(["processed_text"],axis = 1)

        documents = data

        doc_sample = documents[documents['essay_id'] == 10].values[0][0]
        words = []
        for word in doc_sample.split(' '):
            words.append(word)
        processed_docs = documents['essay_text'].map(preprocess)
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=500)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        bow_doc_0 = bow_corpus[0]
        for i in range(len(bow_doc_0)):
            print("Word {} (\"{}\") appears {} time.".format(bow_doc_0[i][0],
            dictionary[bow_doc_0[i][0]],bow_doc_0[i][1]))

        ## TF IDF
        from gensim import corpora, models
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        from pprint import pprint
        for doc in corpus_tfidf:
            pprint(doc)
            break

        ## LDA sin TFIDF
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
        lda_model.print_topics(-1)
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
        for index, score in sorted(lda_model[bow_corpus[0]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 8)))
        lda_model[bow_corpus[0]] # clúster y probabilidades
        ind_without_tfidf = lda_model[bow_corpus]
        y = 0 # individuo

        lista = [ind_without_tfidf[y][x][1] for x in range(2)]
        topicos = list(np.arange(10))
        topicos[np.argmax(lista)] # tópico
        ind_without_tfidf[y]

        y = 0
        lista = [ind_without_tfidf[y][x][1] for x in range(len(ind_without_tfidf[y]))]
        ind_without_tfidf[y][np.argmax(lista)][0]

        topics_wo = []
        for y in range(self.datos.shape[0]):
            if len(ind_without_tfidf[y]) > 0:
                valid_sublist = [sublist for sublist in ind_without_tfidf[y] if len(sublist) > 1]
                if len(valid_sublist) > 0:
                    max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                    topics_wo.append(valid_sublist[max_index][0])
                else:
                    topics_wo.append(None)
            else:
                topics_wo.append(None)

        data["topic"] = topics_wo

        ## LDA CON TF IDF
        corpus_tfidf[0]
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=10, workers=4)
        for idx, topic in lda_model_tfidf.print_topics(-1):
              print('Topic: {} Word: {}'.format(idx, topic))
        for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
                print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 8)))
        ind_with_tfidf = lda_model_tfidf[bow_corpus] # Todos los individuos
        ind_with_tfidf[0]

        topics_with =  []
        for y in range(data.shape[0]):
            if len(ind_with_tfidf[y]) > 0:
                valid_sublist = [sublist for sublist in ind_with_tfidf[y] if len(sublist) > 1]
                if len(valid_sublist) > 0:
                    max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                    topics_wo.append(valid_sublist[max_index][0])
                else:
                    topics_wo.append(None)
            else:
                topics_wo.append(None)

        # Inicializamos la lista vacía
        topics_with = []

        # Iteramos sobre cada documento en `ind_with_tfidf`
        for doc in ind_with_tfidf:
            if len(doc) > 0:  # Aseguramos que no esté vacío
                # Obtenemos el índice del tema con mayor probabilidad
                max_index = max(doc, key=lambda x: x[1])[0]
                topics_with.append(max_index)
            else:
                topics_with.append(None)
        data["topic_tfidf"] = topics_with
        data_LDA = data.copy()

        ##Bert
        data = pd.read_pickle('processed_q.pkl')
        data = data.get(['essay_text','essay_id','processed_text'])

        embeddings_df = pd.read_pickle('embeddings.pkl')
        embeddings_pca_df = np.load('embeddings_pca.npy')
        df_pca_df = pd.read_pickle('df_pca.pkl')

        # Definir el rango de número de clusters que quieres probar
        n_max = 10  # Puedes ajustar este valor según lo necesites
        silhouette_coefficients = []

        # Normalizar los embeddings PCA para que estén escalados entre -1 y 1 (opcional pero recomendado)
        #from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_pcs = scaler.fit_transform(embeddings_pca_df)

        # Probar diferentes números de clusters
        for k in range(2, n_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(scaled_pcs)
            score = silhouette_score(scaled_pcs, kmeans.labels_)
            silhouette_coefficients.append(score)

        # Clustering de los embeddings con K-Means
        kmeans = KMeans(n_clusters=8, random_state=0).fit(embeddings_df)
        data['cluster'] = kmeans.labels_

        ## FAST TEXT
        doc_embedding = pd.read_csv('Doc_Embedding_300_NLP_Encuestas_q.csv',index_col=0)

        pca = PCA(n_components=30, random_state=0)
        pcs = pca.fit_transform(doc_embedding.values)

        scaler = StandardScaler()
        scaled_pcs = scaler.fit_transform(pcs)

        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []
        kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}

        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, n_max):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(scaled_pcs)
            score = silhouette_score(scaled_pcs, kmeans.labels_)
            silhouette_coefficients.append(score)

        cf = pd.DataFrame({'clúster':range(2, n_max),'silueta':silhouette_coefficients})

        K_ = 8
        km = KMeans(n_clusters=K_, random_state=0)
        km.fit_transform(scaled_pcs)
        cluster_labels = km.labels_
        cluster_labels = pd.DataFrame(cluster_labels, columns=['Grupo'])

        data["FAST"] = km.labels_

        return data

    def train_model(self):

        datos_f = pd.read_pickle('data_final.pkl')

        #LDA SIN TF-IDF

        data_final_copia = datos_f.copy()
        Clase = 1
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 1 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        col_met = ['Modelo', 'Tema', 'accuracy', 'roc_auc', 'f1score']
        metricas = pd.DataFrame(columns=col_met)
        metricas.loc[0] = ['SIN TF-IDF', 'tema 1', accuracy, roc_auc, f1_score]

        data_final_copia = datos_f.copy()
        Clase = 2
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 2 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[1] = ['SIN TF-IDF', 'tema 2', accuracy, roc_auc, f1score]

        Clase = 3
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 4 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[2] = ['SIN TF-IDF', 'tema 3', accuracy, roc_auc, f1score]

        Clase = 4
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[3] = ['SIN TF-IDF', 'tema 4', accuracy, roc_auc, f1score]

        Clase = 5
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[4] = ['SIN TF-IDF', 'tema 5', accuracy, roc_auc, f1score]

        Clase = 6
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 5 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[5] = ['SIN TF-IDF', 'tema 6', accuracy, roc_auc, f1score]

        Clase = 7
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 7 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[6] = ['SIN TF-IDF', 'tema 7', accuracy, roc_auc, f1score]

        Clase = 8
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 3 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic"]!=prediccion,"topic"] = -1
        data_final_copia.loc[data_final_copia["topic"]==prediccion,"topic"] = 1
        data_final_copia.loc[data_final_copia["topic"]==-1,"topic"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[7] = ['SIN TF-IDF', 'tema 8', accuracy, roc_auc, f1score]

        #LDA CON TF-IDF

        Clase = 1
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 6 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[8] = ['TF-IDF', 'tema 1', accuracy, roc_auc, f1score]

        Clase = 2
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 7 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[9] = ['TF-IDF', 'tema 2', accuracy, roc_auc, f1score]

        Clase = 3
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 5 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[10] = ['TF-IDF', 'tema 3', accuracy, roc_auc, f1score]

        Clase = 4
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[11] = ['TF-IDF', 'tema 4', accuracy, roc_auc, f1score]

        Clase = 5
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 1 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[12] = ['TF-IDF', 'tema 5', accuracy, roc_auc, f1score]

        Clase = 6
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 2 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[13] = ['TF-IDF', 'tema 6', accuracy, roc_auc, f1score]

        Clase = 7
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 3 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[14] = ['TF-IDF', 'tema 7', accuracy, roc_auc, f1score]

        Clase = 8
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 3 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["topic_tfidf"]!=prediccion,"topic_tfidf"] = -1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==prediccion,"topic_tfidf"] = 1
        data_final_copia.loc[data_final_copia["topic_tfidf"]==-1,"topic_tfidf"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["topic_tfidf"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[15] = ['TF-IDF', 'tema 8', accuracy, roc_auc, f1score]

        #BERT

        data_final_copia = datos_f.copy()
        Clase = 1
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 4 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[16] = ['BERT', 'tema 1', accuracy, roc_auc, f1score]

        Clase = 2
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 6 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[17] = ['BERT', 'tema 2', accuracy, roc_auc, f1score]

        Clase = 3
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 7 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[18] = ['BERT', 'tema 3', accuracy, roc_auc, f1score]

        Clase = 4
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 2 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[19] = ['BERT', 'tema 4', accuracy, roc_auc, f1score]

        Clase = 5
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[20] = ['BERT', 'tema 5', accuracy, roc_auc, f1score]

        Clase = 6
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 3 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[21] = ['BERT', 'tema 6', accuracy, roc_auc, f1score]

        Clase = 7
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 1 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[22] = ['BERT', 'tema 7', accuracy, roc_auc, f1score]

        Clase = 8
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["cluster"]!=prediccion,"cluster"] = -1
        data_final_copia.loc[data_final_copia["cluster"]==prediccion,"cluster"] = 1
        data_final_copia.loc[data_final_copia["cluster"]==-1,"cluster"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["cluster"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[23] = ['BERT', 'tema 8', accuracy, roc_auc, f1score]

        #FAST TEXT

        Clase = 1
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 4 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[24] = ['FAXT TEST', 'tema 1', accuracy, roc_auc, f1score]

        Clase = 2
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 6 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[25] = ['FAXT TEST', 'tema 2', accuracy, roc_auc, f1score]

        Clase = 3
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 1 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[26] = ['FAXT TEST', 'tema 3', accuracy, roc_auc, f1score]

        Clase = 4
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 7 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[27] = ['FAXT TEST', 'tema 4', accuracy, roc_auc, f1score]

        Clase = 5
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 0 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[28] = ['FAXT TEST', 'tema 5', accuracy, roc_auc, f1score]

        Clase = 6
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 3 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[29] = ['FAXT TEST', 'tema 6', accuracy, roc_auc, f1score]

        Clase = 7
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 2 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[30] = ['FAXT TEST', 'tema 7', accuracy, roc_auc, f1score]

        Clase = 8
        data_final_copia = datos_f.copy()
        data_final_copia.loc[data_final_copia["tema"]!=Clase,"tema"] = -1
        data_final_copia.loc[data_final_copia["tema"]==Clase,"tema"] = 1
        data_final_copia.loc[data_final_copia["tema"]==-1,"tema"] = 0


        prediccion = 2 # ser cuidadosos y definir la clase de la predicción como la categoría con mayor cantidad de casos predichos con respecto a la clase original
        data_final_copia.loc[data_final_copia["FAST"]!=prediccion,"FAST"] = -1
        data_final_copia.loc[data_final_copia["FAST"]==prediccion,"FAST"] = 1
        data_final_copia.loc[data_final_copia["FAST"]==-1,"FAST"] = 0

        ytrue = list(data_final_copia["tema"])
        yest = list(data_final_copia["FAST"])

        roc_auc = roc_auc_score(ytrue,yest)
        accuracy = accuracy_score(ytrue,yest)
        f1score = f1_score(ytrue,yest)

        metricas.loc[31] = ['FAXT TEST', 'tema 8', accuracy, roc_auc, f1score]

        metricas = metricas.sort_values(by='roc_auc', ascending=False)

        metricas.to_csv('TABLA METRICAS.csv')

        prom_modelo = metricas.groupby('Modelo')['roc_auc'].mean()
        mejor_modelo = prom_modelo.idxmax()  # Nombre del modelo con el mejor AUC
        auc = prom_modelo.max()       # Valor máximo de la media AUC

        return mejor_modelo, auc

    def ML_FLOW(self):
        try:
            # Paso 1: Cargar datos
            self.datos = self.load_data()

            # Paso 2: Preprocesamiento
            data = self.preprocessing()

            # Paso 3: Entrenamiento del modelo
            mejor_modelo, auc = self.train_model()

            mensaje = f"El mejor modelo es {mejor_modelo} con una media de AUC_ROC de {auc:.2f}"

            return {'success':True,'message':mensaje}
        except Exception as e:
            return {'success':False,'message':str(e)}