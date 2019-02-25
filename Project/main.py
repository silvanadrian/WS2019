import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

questions = pd.read_csv("data/train_dataset.csv", header=None, encoding="iso-8859-1", sep=";")

# Name columns
questions.columns = ['id', 'question', 'answer', 'topic']


science = questions[questions['topic'] == 'science-technology']
kids = questions[questions['topic'] == 'for-kids']
games = questions[questions['topic'] == 'video-games']
sports = questions[questions['topic'] == 'sports']
music = questions[questions['topic'] == 'music']


#Science Questions
print(len(science.index))
science.loc[:, 'question_length'] = science['question'].apply(len)
science.loc[:, 'answer_length'] = science['question'].apply(len)
print(np.mean(science['question_length']))
print(np.std(science['question_length']))

# Kids Questions
print(len(kids.index))

# Game questions
print(len(games.index))
print(len(sports.index))
print(len(music.index))


# Classifier

questions['question'] = questions['question'].str.replace('[^\w\s]','')
questions['answer'] = questions['answer'].str.replace('[^\w\s]','')

train, validate, test = np.split(questions.sample(frac=1), [int(.8 * len(questions)), int(.9 * len(questions))])

y = train[['topic']]

vectorizer = CountVectorizer(analyzer="word", max_features=8913)
vectorizer.fit(train['question'], train['answer'])
train_data_features = vectorizer.transform(train['question']).toarray() + vectorizer.transform(train['answer']).toarray()

forest = RandomForestClassifier(n_estimators = 150)
forest = forest.fit(train_data_features, y['topic'])


test_data_features = vectorizer.transform(test['question']).toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id":test["id"], "topic":result})
output.to_csv('final.csv', index=False, quoting=3)

generated_questions = pd.read_csv("data/sample_crowdsourcing.tsv", header=None, encoding="utf-8", sep="\t")
generated_questions.columns = ['id', 'question', 'answer', 'difficulty', 'opinion', 'factuality']
generated_questions.groupby(['question'])

print(generated_questions.groupby(['question'])['question'].count())
