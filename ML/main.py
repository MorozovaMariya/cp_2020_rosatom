from werkzeug.wrappers import Request, Response
import pprint
import pandas as pd
import os
import nltk
nltk.download('punkt')
import gensim.models as models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import spatial

import urllib.parse as urlparse
from urllib.parse import parse_qs

path = os.getcwd()
dataset_path = path + '/data/data.csv'
model_path = path + '/modeles/model_word2vec'

#Загрузка данных
data = pd.read_csv(dataset_path, header = 0, usecols=['mdmcodenomen','nomen'], sep = ',')

def tokenize(text):
    #  Разбивает текст на токены
    tokens = nltk.word_tokenize(text.lower())
    return tokens

def preprocess_data(sentences):
    # Делает препроцессинг для всех предложений, возвращает лист препроцесснутых слов
    result = list()
    for sentence in sentences:
        result.append(tokenize(sentence))
    return result

def avg_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_answ(q):
    # Возвращает ответ
    #q={"id_request":"5243c597-af1a-4267-b30a-71ecd69951ec","mdmcodenomen":"e80f28f0-1ae7-11eb-85b0-00232453feb3","nomen":"Двутавр 10Б1 ГОСТ 26020/Ст3сп5 ГОСТ 535"}
    #q={"id_request":"5243c597-af1a-4267-b30a-71ecd69951ec","mdmcodenomen":"e80f28f0-1ae7-11eb-85b0-00232453feb3","nomen":"Двутавр 20Б1 СТО АСЧМ 20-93 09Г2С-15 дл.12,0"}
    parsed = urlparse.urlparse(q)
    id_request = parse_qs(parsed.query)['id_request'][0]
    mdmcodenomen = parse_qs(parsed.query)['mdmcodenomen'][0]
    nomen = parse_qs(parsed.query)['nomen'][0]
    
    print(q)
    print( 'Запрос на обработку от 1С:', '\n', 'id_request = ', id_request, '\n', 'mdmcodenomen = ',  mdmcodenomen, '\n', 'nomen = ', nomen)

    #Загрузка данных
    data_load = pd.read_csv(dataset_path, header = 0, usecols=['mdmcodenomen','nomen'])
    len(data_load)
    print( 'Данные для модели загружены')

    data_load.loc[len(data_load)] = [mdmcodenomen, nomen]
    print( 'Обработка данных...')

    # Подготовка данных
    data_for_train_model = preprocess_data(data_load['nomen'])

    #Обучение модели на подготовленных данных
    model = models.Word2Vec(data_for_train_model, min_count=1)
    model.save(model_path)

    index2word_set = set(model.wv.index2word)


    # Векторизация текста
    a = []
    i = len(data_load)-1
    #print(data_load['nomen'][i]) # наш тестовый образец
    for i in range(len(data_load)):
        d = avg_vector(data_load['nomen'][i], model=model, num_features=100, index2word_set=index2word_set)
        a.append(d)
    
    # Вычисление меры схожести заданной строки
    simil = []
    h = (len(a))-1
    for i in range(len(data_load)):
        similarity = int((100.0 - spatial.distance.cosine(a[h],a[i])*100))
        print('%:', similarity, data_load['nomen'][i], 'Запрос: ', data_load['nomen'][h]) #####
        simil.append(similarity)
    # Добавление процентов
    data_load.loc[:, 'proc'] = simil    

    #Формирование ответа
    data_load.drop(data_load.index[[len(data_load)-1]], inplace=True)
    data_answer = data_load[['mdmcodenomen','proc']].sort_values(by='proc', ascending=[False])[:5]

    dd = data_answer.to_json(orient = "records")
    rr = ''
    rr = '{"id_request": "' + id_request + '", "content": ' + dd + '}'

    status = 200
    answ = rr  
    print('\n', answ, ', status = ',status)   
    return status, answ



@Request.application
def application(request):
    #print request
    #pp = pprint.PrettyPrinter(indent=4) #
    #q = request.get_data(False, True, False) 
    q = request.get_data(False, False, False)
    d = request.url
    #q = request.get_json() 500
    #pp.pprint(q)
    status, answ = get_answ(d)
    #status = 200
    return Response(answ, status=status)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    #run_simple('localhost', 5000, application)
    run_simple('10.0.0.6', 5000, application) 
  
