import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
import nltk
from nltk.corpus import stopwords
import pickle


def retrain():
    stop_words = set(stopwords.words('english'))
    le = LE()
    tfv = TfidfVectorizer(min_df=1, stop_words='english')

    def cleanup(sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [w for w in word_tok if not w in stop_words]

        return ' '.join(stemmed_words)

    data = pd.read_csv("./Data/Retrieval_model_data.csv", encoding='utf-8')
    questions = data['question'].values
    print("===>Retrieval_model_data.csv loaded successfully.../")
    X = []

    for question in questions:
        X.append(cleanup(question))

    tfv.fit(X)  # transforming the questions into tfidf vectors
    le.fit(data['class'])  # label encoding the classes as 0, 1, 2, 3

    X = tfv.transform(X)
    y = le.transform(data['class'])

    #trainx, testx, trainy, testy = tts(X, y, test_size=.3, random_state=42)

    model = SVC(kernel='linear')
    model.fit(X, y)

    filename = 'Retrieval_model.pkl'
    pickle.dump(model, open(filename, 'wb'))

    print("===>Model Retrained and Pickled successfully with :", model.score(X, y), "% accuracy")


if __name__ == "__main__":
    retrain()
