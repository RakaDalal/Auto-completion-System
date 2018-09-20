from flask import Flask, jsonify
from flask import request
import os
from methods_for_test import *
import pickle
import gzip
import json

# This is the path where the model is stored 
path_to_model="./"
f = gzip.open(path_to_model+'model.pklz','rb')
model = pickle.load(f)
f.close()
print ("reading done")

# This method is used for returning the predictions in json format
def printing_results_server(salut, prediction_begin, predicted_result, counter_limit):
	result=[]
	if type(predicted_result)==list:
		counter=0
		for item in predicted_result:
			output=re.sub(' +',' ',((salut+" "+prediction_begin+" "+item).strip()).capitalize())
			result.append(output)
			counter+=1
			if counter==counter_limit:
				break
	else:
		output=re.sub(' +',' ',((salut+" "+prediction_begin+" "+predicted_result).strip()).capitalize())
		result.append(output)
	data={}
	data["Completions"] = result
	return jsonify(data)

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'    

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/')

@app.route('/<user_input>', methods=['GET'])
def home(user_input):
	input_to_model = request.args.get('q')
	global model
	(salut, prediction_begin, predicted_result, counter_limit)=backoff_predict(model["total"], model["dictionaries"][0],model["dictionaries"][1],model["dictionaries"][2],model["dictionaries"][3],model["dictionaries"][4],model["dictionaries"][5],model["most_frequent_unigram"][0],model["most_frequent_unigram"][1],model["parameters"][0],model["parameters"][1],model["parameters"][2],model["parameters"][3],model["beta"],model["alpha"],model["gamma"],model["sentence_dictionary"],input_to_model)
	result=printing_results_server(salut, prediction_begin, predicted_result, counter_limit)
	return result
	

if __name__=='__main__' :
    app.run()
	

