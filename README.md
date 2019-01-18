# HeavyWater_ML_Solution

Requirement:

Anaconda python 3.7
Flask

Operating System:
Ubuntu

How to run it:

this program has already running on AWS EC2 server. If this program is going to be executed in localhost, please make sure the local device has Anaconda python 3.7 and flask. Running with this command:
Python app.py
or
FLASK_APP=app.py flask run
Then go to localhost:5000. 
Input the word into the search box and click predict button. The result will be shown below.

Implement Details:

Training:
This training document has over 20 million words. After removing the repeating words, this document still has over 1 million distinguish words. Therefore, it is necessary to use tf-idf to filter the words.
If the tf-idf value of words is between 0.003 and 0.4, it will be considered as characteristic word. After filtering, around 3700 words were selected as characteristic word.
Multinomial Bayesian was used for training model. The training data was divided by 8:2. The accuracy is between 75% to 80%.
Note: To running the training, make sure the local device has csv file. The command will be:
python model.py
It will output train_model.m and voc_idf_list.txt. 
Warning: Make sure the local devices has larger than 10GB RAM memory and training will take a lot of time.

Prediction:
After training, we can predict the result by inputing the words. The running method on localhost has been mentioned before. 

Note:
You can also running the prediction in the backend by uncommenting the line words and inputting command line:
python predict_result.py
The result will output on the terminal.

Deploying:
This project has been once deployed on AWS EC2 Linux server. 

Testing:
1.Put the word into inputbox and click predict, the result will shown under inputbox. 
2.Put the command line GET like this:
GET (your_ip):(your_port)/predict?words=2ee1e0de2738
And the terminal will return the result:
{
  "result":"POLICY CHANGE"
}

Other notice:
Sometimes port with specific number will be used. If that happends, just change to another port. 
