from methods_for_training import *
import pickle
import gzip
import sys

def main():
	# This is the path where the model will be stored 
	path_to_model="./"
	input_file=sys.argv[1]
	processed_data=read_data(input_file)
	(data, sentence_dict)=data_preprocessing(processed_data)
	total_data=data
	(train_data,development_data)=split_into_training_and_development(data)
	vocabulary=create_vocab(train_data)
	oov=least_common_words(vocabulary)
	(unigram_dict, bigram_dict, trigram_dict)=ngramscount(train_data,oov)
	(unigram_list, bigram_list, trigram_list, unigram_set, bigram_set, trigram_set) = counting(development_data)

	# optimizing parameters
	discount_parameters=optimization(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,0)
	
	vocabulary=create_vocab(total_data)
	oov=least_common_words(vocabulary)
	(unigram_dict, bigram_dict, trigram_dict)=ngramscount(total_data,oov)
	(unigram_list, bigram_list, trigram_list, unigram_set, bigram_set, trigram_set) = counting(total_data)
	beta=backoff_beta(unigram_dict,unigram_set,0,discount_parameters[0]) 
	alpha_utility= backoff_alpha(unigram_dict,bigram_dict,bigram_set,0,discount_parameters[0],discount_parameters[1])
	gamma_utility= backoff_gamma(unigram_dict,bigram_dict,trigram_dict,trigram_set,0,discount_parameters[0],discount_parameters[1],discount_parameters[2]) 
	unigram_dict_prob=unigram_prob(unigram_dict, beta, 0, discount_parameters[0])
	bigram_dict_prob=bigram_prob(bigram_dict, unigram_dict, 0, discount_parameters[1])
	trigram_dict_prob=trigram_prob(trigram_dict, bigram_dict, 0, discount_parameters[2])
	most_frequent_unigram = max(unigram_dict_prob.iteritems(), key=operator.itemgetter(1))[0]
	frequency = max(unigram_dict_prob.iteritems(), key=operator.itemgetter(1))[1]
	total=sum(unigram_dict.values())

	# creating the model
	model={}
	model["total"]=total
	model["dictionaries"]=(unigram_dict,bigram_dict,trigram_dict,unigram_dict_prob,bigram_dict_prob,trigram_dict_prob)
	model["most_frequent_unigram"]=(most_frequent_unigram,frequency)
	model["parameters"]=(0,discount_parameters[0],discount_parameters[1],discount_parameters[2])
	model["beta"]=beta
	model["alpha"]=alpha_utility
	model["gamma"]=gamma_utility
	model["sentence_dictionary"]=sentence_dict

	# storing the serialized model
	f = gzip.open(path_to_model+'model.pklz','wb')
	pickle.dump(model,f)
	f.close()

	print ("Training Done")

if __name__=='__main__' :
    main()



