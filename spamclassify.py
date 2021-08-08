from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

df = pd.read_table('SMSSpamCollection.txt',names=['label','sms_message'])

# Output printing out first 5 rows
df['label'] = df.label.map({'ham':0,'spam':1})



X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
count_vector.get_feature_names()
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)
msg_text= ['free gift']
test_msg = count_vector.transform(msg_text)
pred = naive_bayes.predict(test_msg)
if pred == [0]:
    print(msg_text," is ham")
else:
    print(msg_text," is spam")    

##Model Evaluation

print('Accuracy score: ', format(accuracy_score(predictions,y_test)))
print('Precision score: ', format(precision_score(predictions,y_test)))
print('Recall score: ', format(recall_score(predictions,y_test)))
print('F1 score: ', format(f1_score(predictions,y_test)))