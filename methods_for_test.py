import re

# This method is used to find the first word to predict
def finding_first_predicted_word(possible_ngrams, last_word, prediction, predicted_result, count):
    for item in possible_ngrams:
        if item[0].startswith(last_word):
            prediction.append(item[0])
            predicted_result+=" "+item[0]
            count+=1
            break
    return (prediction, predicted_result, count)

# This method finds the next word to be predicted
def finding_next_word(value, word, prob_final, next_word, constant):
    prob=constant*value
    if prob>prob_final:
        prob_final=prob
        next_word=word
    return (prob_final, next_word)

# This method handles if the input consists of more than three words
def more_than_three_words(sentence_dict,text):
    possible_sents=[]
    key=' '.join(text.split()[:3])
    if key in sentence_dict:
        possible_sents=[item for item in sentence_dict[key] if item.startswith(text)==True]
    return possible_sents

# This method is for preprocessing the input
def preprocessing_test_data(prediction, predicted_result, text, sentence_dict, salut, counter_limit, external_flag, unigram_dict_prob, count, most_frequent_unigram ):
    last_word="";prediction_begin="";prediction_next=[];flag=0
    if len(prediction)>3:
        result=more_than_three_words(sentence_dict,text)
        if len(result)>0:
            predicted_result=result
            flag=1
        else:
            text2=""
            size=len(prediction)
            counter=0
            for item in text.split(" "):
                if (size-counter)>3:
                    prediction_begin+=" "+item
                else:
                    prediction_next.append(item)
                    text2+=" "+item
                counter+=1
            prediction=prediction_next
            text=text2
            external_flag=1
    if len(prediction)==1:
        predicted_result=""
        prediction=[]
        (prediction, predicted_result, count)=finding_first_predicted_word(unigram_dict_prob, text, prediction, predicted_result, 0)
        if count==0:
            prediction.append(most_frequent_unigram[0])
            predicted_result=most_frequent_unigram[0]
            count+=1
    if len(prediction)>=2 and flag==0:
        prediction=[]
        text=text.strip()
        predicted_result=(text.rsplit(' ', 1)[0]).strip()
        last_word=(text.rsplit(' ', 1)[1]).strip()
        text=predicted_result
        for item in text.split(" "):
            prediction.append(item)
        count=0

    return (external_flag, prediction_begin, prediction, predicted_result, count, last_word, flag)

# This method handles salutations
def salutations_handling(text, most_frequent_word):
    salutations=["hi", "hello", "hey"]
    salut="";prediction=[]
    text=text.strip();text=text.lower()
    text2=text.split(" ")
    if (text2[0].rstrip(",")) in salutations:
        try:
            predicted_result=(text.split(" ",1)[1]).strip()
        except:
            predicted_result=most_frequent_word
        salut=text.split(" ",1)[0]
        text=predicted_result
    else:
        predicted_result=text
    for item in text.split(" "):
        prediction.append(item)
    
    return (text, predicted_result, prediction, salut)

# This method uses trigram for prediction
def prediction_using_trigrams(prediction, predicted_result, last_word, index, count, trigram_dict_prob, prob_final, next_word):
    possible_trigrams=[];flag=0;indicator=0
    if (prediction[index], prediction[index+1]) in trigram_dict_prob:
        if count==0:
            possible_trigrams=trigram_dict_prob[(prediction[index], prediction[index+1])]
            (prediction, predicted_result, count)=finding_first_predicted_word(possible_trigrams, last_word, prediction, predicted_result, count)
            if count!=0:
                indicator=1
            else:
                flag=1
        else:
            possible_trigrams.append(trigram_dict_prob[(prediction[index], prediction[index+1])][0])
            if possible_trigrams[0][1]>-1:
                (prob_final, next_word)=finding_next_word(possible_trigrams[0][1], possible_trigrams[0][0], prob_final, next_word, 1)
            else:
                flag=1
    else:
        flag=1    
    return (indicator, flag, count, prediction, predicted_result, prob_final, next_word)

# This method uses bigram for prediction                
def prediction_using_bigrams(prediction, predicted_result, last_word, index, count, bigram_dict_prob, prob_final, next_word, constant):
    possible_bigrams=[];indicator=0;flag=0;flag3=0
    if (prediction[index],) in bigram_dict_prob:
        if count==0:
            possible_bigrams=bigram_dict_prob[(prediction[index],)]
            (prediction, predicted_result, count)=finding_first_predicted_word(possible_bigrams, last_word, prediction, predicted_result, count)
            if count!=0:
                indicator=1
            else:
                flag=1
                flag3=1
        else:
            possible_bigrams.append(bigram_dict_prob[(prediction[index],)][0])
            if possible_bigrams[0][1]>-1:
                (prob_final, next_word)=finding_next_word(possible_bigrams[0][1], possible_bigrams[0][0], prob_final, next_word, constant)
            else:
                flag=1
                flag3=1
    else:
        flag=1
        flag3=1

    return (indicator, flag, flag3, count, prediction, predicted_result, prob_final, next_word)

# This method is for alpha computation
def alpha_computation(prediction, index, alpha_utility, beta, unigram_dict, total): 
    if (prediction[index+1],) in unigram_dict:
        c_y=unigram_dict[(prediction[index+1],)]
    else:
        c_y=unigram_dict[(u'oov',)]
    try:
        (c_x_y_dash,c_y_d1,c_y_dash)=alpha_utility[(prediction[index+1],)]
    except:
        (c_x_y_dash,c_y_d1,c_y_dash)=alpha_utility[(u'oov',)]
    alpha_lower= (c_y_d1/float(total)) + (c_y_dash*(beta/float(len(unigram_dict))))
    if (alpha_lower==0):
        alpha_lower=0.01
    alpha=(1-(c_x_y_dash/float(c_y)))/(float(alpha_lower))

    return (alpha, c_y)

# This method is for gamma computation
def gamma_computation(prediction, index, bigram_dict, unigram_dict, total, gamma_utility, alpha_utility, beta):
    (alpha, c_y) = alpha_computation(prediction, index, alpha_utility, beta, unigram_dict, total)

    if (prediction[index],prediction[index+1]) in bigram_dict:
        c_x_y=bigram_dict[(prediction[index],prediction[index+1])]
    else:
        c_x_y=bigram_dict[(u'oov',u'oov')] 
    try:
        (c_x_y_z,c_y_z,c_z_d1,c_z)=gamma_utility[(prediction[index],prediction[index+1])]
    except:
        (c_x_y_z,c_y_z,c_z_d1,c_z)=gamma_utility[(u'oov',u'oov')]
    try:
        gamma_lower=(c_y_z/float(c_y))+(alpha*(c_z_d1/float(total)))+(alpha*(c_z*(beta/float(len(unigram_dict)))))
    except:
        gamma_lower=0.01
    try:
        gamma=(1-(c_x_y_z/float(c_x_y)))/float(gamma_lower)
    except:
        gamma=0.01

    return (gamma, alpha)

# This method is used for prediction
def backoff_predict(total,unigram_dict,bigram_dict,trigram_dict,unigram_dict_prob,bigram_dict_prob,trigram_dict_prob,most_frequent_unigram,frequency,epsilon,discount1,discount2,discount3,beta,alpha_utility,gamma_utility,sentence_dict,text):
    counter_limit=6;index=-1;external_flag=0;count=1;flag3=0
    (text, predicted_result, prediction, salut)=salutations_handling(text, most_frequent_unigram[0])
    (external_flag, prediction_begin, prediction, predicted_result, count, last_word, flag)=preprocessing_test_data(prediction, predicted_result, text, sentence_dict, salut, counter_limit, external_flag, unigram_dict_prob, count, most_frequent_unigram )
    if flag == 1:
        index=8
    while (index<7):
        if len(prediction)==3 and external_flag==0:
            if predicted_result.strip() in sentence_dict:
                predicted_result=(sentence_dict[predicted_result.strip()])
                break
        index+=1
        prob_final=0;next_word=""
        if len(prediction)==2:
            index=0
        if len(prediction)>1:
            (indicator, flag1, count, prediction, predicted_result, prob_final, next_word)=prediction_using_trigrams(prediction, predicted_result, last_word, index, count, trigram_dict_prob, prob_final, next_word)
            
            if count!=0 and indicator==1:
                continue
        if len(prediction)==1 or flag1==1:
            if len(prediction)==1:
                (indicator, flag2, flag3, count, prediction, predicted_result, prob_final, next_word)=prediction_using_bigrams(prediction, predicted_result, last_word, index, count, bigram_dict_prob, prob_final, next_word, 1)
                if count!=0 and indicator==1:
                    continue
            if  len(prediction)>1 or flag2==1 :
                if flag3 !=1:
                    (gamma, alpha)=gamma_computation(prediction, index, bigram_dict, unigram_dict, total, gamma_utility, alpha_utility, beta)

                    (indicator, flag3, flag4, count, prediction, predicted_result, prob_final, next_word)=prediction_using_bigrams(prediction, predicted_result, last_word, index+1, count, bigram_dict_prob, prob_final, next_word, gamma)
                    if count!=0 and indicator==1:
                        continue
                else:
                    if count!=0:
                        (prob_final, next_word)=finding_next_word(frequency, most_frequent_unigram[0], prob_final, next_word, 1)
                    if count==0:
                        (prediction, predicted_result, count)=finding_first_predicted_word(unigram_dict_prob, last_word, prediction, predicted_result, count)
                        if count!=0:
                            continue
                        else:
                            prediction.append(most_frequent_unigram[0])
                            predicted_result+=" "+most_frequent_unigram[0]
                            count+=1
                            continue
        prediction.append(next_word)
        predicted_result+=" "+next_word
        if re.match('^[A-Z a-z][^?!.]*[?.!]$', predicted_result):
            break

    return (salut, prediction_begin, predicted_result, counter_limit)
