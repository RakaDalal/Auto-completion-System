
import json
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import math
import re
import nltk
from collections import defaultdict
from collections import OrderedDict
import operator
from scipy import optimize
from sklearn import metrics
import numpy as np


# This method reads the json file to extract the input_data
def read_data(filename):
    processed_data=[]
    with open(filename) as f:
        data=json.load(f)
    dataframe = pd.DataFrame(data)

    for i in range(0, len(dataframe["Issues"])):
        IssueId = dataframe["Issues"][i]["IssueId"]
        CompanyGroupId = dataframe["Issues"][i]["CompanyGroupId"]
        messages = dataframe["Issues"][i]["Messages"]
        for item in messages:
            text = (item["Text"])
            IsFromCustomer = (item["IsFromCustomer"])
            tup=(IssueId, CompanyGroupId, text, IsFromCustomer)
            processed_data.append(tup)


    return (processed_data) 

# This method preprocesses the extracted input_data and also creating a sentence dictionary    
def data_preprocessing(processed_data):
    salutations=["hi", "hello", "hey"]
    data = [];sentence_dict=defaultdict(list)
    for row in processed_data:
        text = row[2];flag = str(row[3])
        sent_text = nltk.sent_tokenize(text)
        for sentence in sent_text:
            sentence=re.sub(' +',' ',sentence)
            if len(sentence.split(" "))>1:
                sentence_without_first_word=sentence.split(" ",1)[1]
                remove=[word for word in sentence_without_first_word.split(" ") if word[0].isupper()]
                for item in remove:
                    if item!="I" and item!="Bye" and item!="Morning" and item!="Evening" and item!="AM" and item!="PM":
                        sentence=sentence.replace(item,"") 
            sentence=re.sub(' +',' ',sentence)
            sentence=sentence.lower()
            data.append(sentence)
            if flag == "False":
                sentence2=sentence.split(" ")
                if sentence2[0] in salutations:
                    sentence=sentence.replace(sentence2[0],"")
                    sentence=sentence.strip(" ")
                    sentence=sentence.lstrip(",")
                key=' '.join(sentence.split()[:3])
                sentence_dict[key].append(sentence)
    for key in sentence_dict:
        sentence_dict[key]=list(set(sentence_dict[key]))   
    return (data, sentence_dict)

# This method splits the data into training set and development set
def split_into_training_and_development(data):
    train_data = data[:(len(data)*4)/5]
    development_data = data[(len(data)*4)/5:]
    return (train_data,development_data)

# This method creates a vocabulary of words
def create_vocab(data):
    vocabulary={}
    vectorizer = CountVectorizer()
    vectorizer_fit=vectorizer.fit_transform(data)
    words = (vectorizer.get_feature_names())
    count = (vectorizer_fit.toarray().sum(axis=0))
    for i in range(0,len(words)):
        vocabulary[words[i]]=count[i]    
    return vocabulary


# This methods finds 20 least frequent words and marks them as out of vocabulary(oov)
def least_common_words(vocabulary):
    oov=[]
    count=0
    for key, value in sorted(vocabulary.iteritems(), key=lambda (k,v): (v,k)):
        count+=1
        if count<=20:
            oov.append(key)
    return oov

# This method creates unigram, bigram and trigram from the data provided and stores them in separate dictionaries along with the counts
def ngramscount(data,oov):
    unigram_dict={};bigram_dict={};trigram_dict={}
    for conversation in data:
        conversation2=conversation
        for item in conversation.split(" "):
            if item in oov:
                try:
                    conversation2=str.replace(str(conversation2),str(item),"oov")
                except:
                    continue
        unigrams = ngrams(conversation2.split(" "),1)
        for item in unigrams:
            if item in unigram_dict:
                unigram_dict[item]+=1
            else:
                unigram_dict[item]=1
        bigrams = ngrams(conversation2.split(" "),2)
        for item in bigrams:
            if item in bigram_dict:
                bigram_dict[item]+=1
            else:
                bigram_dict[item]=1
        trigrams = ngrams(conversation2.split(" "),3)
        for item in trigrams:
            if item in trigram_dict:
                trigram_dict[item]+=1
            else:
                trigram_dict[item]=1

    if (u'oov',u'oov') not in bigram_dict:
        bigram_dict[(u'oov',u'oov')]=1

    if (u'oov',u'oov',u'oov') not in trigram_dict:
        trigram_dict[(u'oov',u'oov',u'oov')]=1

    return (unigram_dict, bigram_dict, trigram_dict)

# This method stores data in form of ngrams in separate lists and also creates sets (one each for unigram, bigram and trigram)
def counting(data):
    unigram_list_total=[];bigram_list_total=[];trigram_list_total=[];unigram_set=[];bigram_set=[];trigram_set=[]
    for conversation in data:
        unigrams = ngrams(conversation.split(" "),1)
        unigram_list=[]
        for item in unigrams:
            unigram_list.append(item)
            unigram_set.append(item)
        unigram_list_total.append(unigram_list)
        bigrams = ngrams(conversation.split(" "),2)
        bigram_list=[]
        for item in bigrams:
            bigram_list.append(item)
            bigram_set.append(item)
        bigram_list_total.append(bigram_list)
        trigrams = ngrams(conversation.split(" "),3)
        trigram_list=[]
        for item in trigrams:
            trigram_list.append(item)
            trigram_set.append(item)
        trigram_list_total.append(trigram_list)
    unigram_set=set(unigram_set)
    bigram_set=set(bigram_set)
    trigram_set=set(trigram_set)
    tup=(unigram_list_total, bigram_list_total, trigram_list_total, unigram_set, bigram_set, trigram_set)
    return (tup)

# This method creates a dictionary for unigram discounted probabilities   
def unigram_prob(unigram_dict, beta, epsilon, discount1):
    unigram_dict_prob={}
    total=sum(unigram_dict.values())
    for key in unigram_dict:
        if unigram_dict[key]>epsilon:
            upper=unigram_dict[key]-discount1
            lower=total
            prob=upper/float(lower)
        else:
            prob=beta/float(len(unigram_dict))
        unigram_dict_prob[key]=prob
    unigram_dict_prob = OrderedDict(sorted(unigram_dict_prob.items(), key=operator.itemgetter(1), reverse=True))
    return unigram_dict_prob

# This method creates a dictionary for bigram discounted probabilities
def bigram_prob(bigram_dict, unigram_dict, epsilon, discount2):
    bigram_dict_prob=defaultdict(list)
    for key in bigram_dict:
        if bigram_dict[key]>epsilon:
            upper=bigram_dict[key]-discount2
            lower=unigram_dict[(key[0],)]
            prob=upper/float(lower)
        else:
            prob=-1
        bigram_dict_prob[(key[0],)].append((key[1],prob))

    for key in bigram_dict_prob:
        bigram_dict_prob[key]=sorted(bigram_dict_prob[key], key=lambda x: x[1], reverse=True) 
    return bigram_dict_prob

# This method creates a dictionary for trigram discounted probabilities
def trigram_prob(trigram_dict, bigram_dict, epsilon, discount3):
    trigram_dict_prob=defaultdict(list)
    for key in trigram_dict:
        if trigram_dict[key]>epsilon:
            upper=trigram_dict[key]-discount3
            lower=bigram_dict[(key[0],key[1])]
            prob=upper/float(lower)
        else:
            prob=-1
        trigram_dict_prob[(key[0],key[1])].append((key[2],prob))

    for key in trigram_dict_prob:
        trigram_dict_prob[key]=sorted(trigram_dict_prob[key], key=lambda x: x[1], reverse=True) 
    return trigram_dict_prob


# This method computes the beta for backoff     
def backoff_beta(unigram_dict,unigram_set,epsilon,discount1):
    total=sum(unigram_dict.values())
    c_y_d1=0
    c_y=0
    unigram_set=list(unigram_set)
    unigram_set.append((u'oov',))
    unigram_set=set(unigram_set)
    for item in unigram_set:
        if item in unigram_dict and unigram_dict[item]>epsilon:
            c_y_d1+=(unigram_dict[item]-discount1)
        else:
            c_y+=1
    beta=c_y_d1/float(total)
    beta=1-beta
    beta=beta/float(c_y)
    return (beta)

# This method computes the alpha for backoff 
def backoff_alpha(unigram_dict,bigram_dict,bigram_set,epsilon,discount1,discount2):
    
    alpha_utility={}
    bigram_set=list(bigram_set)
    bigram_set.append((u'oov',u'oov'))
    bigram_set=set(bigram_set)
    for item in bigram_set:
        (x,y)=item
        z2=(y,)
        z1=(x,)
        c_x_y=0
        c_y_d1=0
        c_y=0
        if item in bigram_dict and bigram_dict[item]>epsilon:
            c_x_y=(bigram_dict[item]-discount2)
        else:
            if z2 in unigram_dict and unigram_dict[z2]>epsilon:
                c_y_d1=(unigram_dict[z2]-discount1)
            else:
                c_y=1
        tup=(c_x_y,c_y_d1,c_y)            
        if z1 in alpha_utility:
            actual_tup=alpha_utility[z1]
            tup2=(actual_tup[0]+tup[0],actual_tup[1]+tup[1],actual_tup[2]+tup[2])
            alpha_utility[z1]=tup2
        else:
            alpha_utility[z1]=tup
    return (alpha_utility)

# This method computes the gamma for backoff 
def backoff_gamma(unigram_dict,bigram_dict,trigram_dict,trigram_set,epsilon,discount1,discount2,discount3):
    
    gamma_utility={}
    trigram_set=list(trigram_set)
    trigram_set.append((u'oov',u'oov',u'oov'))
    trigram_set=set(trigram_set)
    for item in trigram_set:
        (x,y,z)=item; t3=(z,);t2=(y,z);t1=(x,y);c_x_y_z=c_y_z=c_z_d1=c_z=0
        if item in trigram_dict and trigram_dict[item]>epsilon:
            c_x_y_z=(trigram_dict[item]-discount3)
        else:
            if t2 in bigram_dict and bigram_dict[t2]>epsilon:
                c_y_z=(bigram_dict[t2]-discount2)
            else:
                if t3 in unigram_dict and unigram_dict[t3]>epsilon:
                    c_z_d1=(unigram_dict[t3]-discount1)
                else:
                    c_z=1
        tup=(c_x_y_z,c_y_z,c_z_d1,c_z)            
        if t1 in gamma_utility:
            actual_tup=gamma_utility[t1]
            tup2=(actual_tup[0]+tup[0],actual_tup[1]+tup[1],actual_tup[2]+tup[2],actual_tup[3]+tup[3])
            gamma_utility[t1]=tup2
        else:
            gamma_utility[t1]=tup
    return (gamma_utility)

# This method performs the alpha gamma computation
def alpha_gamma_computation(alpha_utility, item1, gamma_utility, item2, total, c_y, c_x_y, beta, unigram_dict):
    (c_x_y_dash,c_y_d1,c_y_dash)=alpha_utility[item1]
    alpha_lower= (c_y_d1/float(total)) + (c_y_dash*(beta/float(len(unigram_dict))))
    if (alpha_lower==0):
        alpha_lower=0.01
    alpha=(1-(c_x_y_dash/float(c_y)))/(float(alpha_lower))
    (c_x_y_z,c_y_z,c_z_d1,c_z)=gamma_utility[item2]
    gamma_lower=(c_y_z/float(c_y))+(alpha*(c_z_d1/float(total)))+(alpha*(c_z*(beta/float(len(unigram_dict)))))
    gamma=(1-(c_x_y_z/float(c_x_y)))/float(gamma_lower)

    return (gamma, alpha)

# This method computes the perplexity for backoff trigram model
def perplexity(discount_parameters,unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,epsilon): 
    total=sum(unigram_dict.values())
    beta=backoff_beta(unigram_dict,unigram_set,epsilon,discount_parameters[0]) 
    alpha_utility= backoff_alpha(unigram_dict,bigram_dict,bigram_set,epsilon,discount_parameters[0],discount_parameters[1])      
    gamma_utility= backoff_gamma(unigram_dict,bigram_dict,trigram_dict,trigram_set,epsilon,discount_parameters[0],discount_parameters[1],discount_parameters[2])
    value=0
    count=0
    for i in range(0, len(trigram_list)):
        for j in range(0, len(trigram_list[i])):
            if trigram_list[i][j] in trigram_dict and trigram_dict[trigram_list[i][j]]>epsilon:
                upper=trigram_dict[trigram_list[i][j]]-discount_parameters[2]
                lower=bigram_dict[bigram_list[i][j]]
                prob=upper/float(lower)
            else:
                if bigram_list[i][j] in bigram_dict:
                    c_x_y=bigram_dict[bigram_list[i][j]]
                else:
                    c_x_y=bigram_dict[(u'oov',u'oov')] 
                if unigram_list[i][j+1] in unigram_dict:
                    c_y=unigram_dict[unigram_list[i][j+1]]
                else:
                    c_y=unigram_dict[(u'oov',)]
                (gamma, alpha)=alpha_gamma_computation(alpha_utility, unigram_list[i][j+1], gamma_utility, bigram_list[i][j], total, c_y, c_x_y, beta, unigram_dict)
                if bigram_list[i][j+1] in bigram_dict and bigram_dict[bigram_list[i][j+1]]>epsilon:
                    upper=bigram_dict[bigram_list[i][j+1]]-discount_parameters[1]
                    lower=unigram_dict[unigram_list[i][j+1]]
                    prob=gamma*(upper/float(lower))
                else:
                    if unigram_list[i][j+2] in unigram_dict and unigram_dict[unigram_list[i][j+2]]>epsilon:
                        upper=unigram_dict[unigram_list[i][j+2]]-discount_parameters[0]
                        lower=total
                        prob=gamma*alpha*(upper/float(lower))
                    else:
                        prob=gamma*alpha*(beta/float(len(unigram_dict)))
            try:    
                value+=math.log(prob)
            except:
                print (x,prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

# This method performs optimization of parameters
def optimization(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,epsilon):
    bnds=((0.1,0.4),(0.1,0.4),(0.1,0.4))
    res = optimize.minimize(perplexity,0.2*np.ones(3),args=(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,epsilon), method='L-BFGS-B',bounds=bnds,options={'maxiter': 10, 'disp': True})
    return (res.x)
    
