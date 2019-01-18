

import pandas as pd
import numpy as np
from pandas import DataFrame as df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

def tfidf_process(feature):
    tfidf_vec = TfidfVectorizer(binary=False,decode_error='ignore',
                                stop_words='english', max_df=0.4, min_df=0.005) 
    # print(tfidf_vec)
    vec = tfidf_vec.fit_transform(feature) 

    voc = tfidf_vec.vocabulary_  
    res = sorted(voc.items(), key=lambda d:d[1])
    idf_value = tfidf_vec.idf_ 

    voc_idf_list = []
    num = 0
    for r in res:
        tmp = list(r)
        tmp[1] = idf_value[num]
        num += 1
        voc_idf_list.append(tmp)   

    with open('voc_idf_list.txt', 'w') as fw:
        for v in voc_idf_list:
            fw.write(v[0]+'\t'+str(v[1])+'\n')
    

    arr = vec.todense()
    arr = np.array(arr)
    return arr

def main():
    data = pd.read_csv('shuffled-full-set-hashed.csv', header=None)
    new_data = data.dropna() 

    labels = new_data.iloc[:,0]  
    feature = new_data.iloc[:, 1]  
    print('read done...')
    feature_vectors = tfidf_process(feature) 
    print('tfidf done...')
    
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=0)

    clf = MultinomialNB(alpha=0.01) 
    clf.fit(X_train, y_train)
    print('model done...') 
    scores = clf.score(X_test, y_test)  
    print(scores)
    joblib.dump(clf, "train_model.m")  

if __name__ == '__main__':
    main()
