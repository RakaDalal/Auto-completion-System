# Auto-completion-System

The project involves building an auto-completion system from sample conversations between customers and customer service representatives. The main aim is to speed up the customer service
representatives' responses by suggesting sentence completions. I have used a trigram Katz’s back-off model for this. I have splitted the data into training set (80%) and development set (20%) and used the
development set to tune the parameters. I have used the optimize function from scipy for that. After training, I have saved the model on disk as a pickle file(model.pklz). During testing, the model is simply
loaded from disk to run the predictions.

# Prerequisites:

All the packages required for this project to run successfully are listed in the requirements.txt file, which can be installed issuing the following command:

pip install –r requirements.txt

# How to run the system:

Download the entire folder and run,
1. python training.py <input_file>
2. python server.py

<input_file> is the json file containing sample conversations that you want the model to train on.

Sample tests:
1. curl http://localhost:5000/autocomplete?q=What+is+y
2. curl http://localhost:5000/autocomplete?q=how+ca
3. curl http://localhost:5000/autocomplete?q=when+w
