import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pickle
from nltk.stem import WordNetLemmatizer
import csv
from sklearn.svm import SVC
import torch
from transformers import GPT2Tokenizer
import transformers
from TTS.api import TTS
import pyaudio
import wave
import speech_recognition as sr
import warnings

warnings.filterwarnings("ignore")

print("===>Loading models. Please wait.../")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model from disk
filename = 'Retrieval_model.pkl'
Retrieval_model = pickle.load(open(filename, 'rb'))

# load the Tokenizer from disk
filename = 'GPT2_tokenizer.pkl'
tokenizer = pickle.load(open(filename, 'rb'))

# load the model from disk
filename = 'GPT2_model.pkl'
Generative_model = pickle.load(open(filename, 'rb'))
Generative_model = Generative_model.to(device)

# TTS model loading id cuda available
if device == "cuda":
    tts = TTS(model_path="checkpoint_1032000.pth", config_path="config.json", progress_bar=True, gpu=True)
else:
    tts = TTS(model_path="checkpoint_1032000.pth", config_path="config.json", progress_bar=True, gpu=False)

QUESTIONSET = list()
ANSWERSET = list()
CLASSSET = list()

print("===>Models Loaded Successfully!\n\n")

text_list = []


def ASR():
    r = sr.Recognizer()
    mic = sr.Microphone()
    audio_list = []
    # o = Au_ext
    # create an empty list to store untranscribed audio samples
    untranscribed_audio_list = []

    print("Listening")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0)
        audio = r.listen(source)
    x = f"{audio}"
    last_four_chars = x[-4:]
    try:
        if x == "A6D0":
            print(last_four_chars)
        else:
            # try to transcribe the audio
            print("start Transcribing")
            text = r.recognize_google(audio, language="en-in")
            if text is None:
                # if unable to transcribe, add to the transcribed audio list
                untranscribed_audio_list.append(audio)
            else:
                # if transcribed successfully, add to the audio list
                audio_list.append(audio)
                if "A6D0" in last_four_chars:
                    audio_list[-1].pop()
                    k = 1

    except sr.UnknownValueError:
        print("Could not understand audio")
        # if unable to transcribe, add to the un-transcribed audio list
        untranscribed_audio_list.append(audio)

    # remove un-transcribed audio samples from the audio list
    audio_list = [audio for audio in audio_list if audio not in untranscribed_audio_list]

    # transcribe the remaining audio samples in the audio list
    # text_list=[]
    for audio in audio_list:
        text = r.recognize_google(audio, language="en-in")
        if text is None:
            print("Could not understand audio")
            text_list[len(text_list)].pop()
            print('cleared')
        else:
            text_list.append(text)

    if text_list:
        print(text_list[-1])
        return str(text_list[-1])
    else:
        print("##############")
        return "empty"


def play(text1):
    # Run TTS

    tts.tts_to_file(
        text=text1,
        file_path="out.wav")

    # !usr/bin/env python
    # coding=utf-8

    # define stream chunk
    chunk = 1024

    # open a wav format music
    f = wave.open(r"out.wav", "rb")

    # instantiate PyAudio
    p = pyaudio.PyAudio()

    #     # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

        # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()


def cleanup(sentence):
    stop_words = set(stopwords.words('english'))
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [w for w in word_tok if not w in stop_words]

    return ' '.join(stemmed_words)


def load_data():
    le = LE()

    # stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=1, stop_words='english')

    data = pd.read_csv("./Data/Retrieval_model_data.csv")
    questions = data['question'].values

    X = []

    for question in questions:
        X.append(cleanup(question))

    tfv.fit(X)  # transforming the questions into tfidf vectors
    le.fit(data['class'])  # label encoding the classes as 0, 1, 2, 3

    X = tfv.transform(X)
    y = le.transform(data['class'])

    return data, X, y, tfv, le


# noinspection SpellCheckingInspection
def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))

    ixarr.sort()

    ixs = []
    for i in ixarr[-5:]:
        ixs.append(i[1])

    return ixs[::-1]


# Function to extract the proper nouns
def ClassExtractor(text):
    lemmatizer = WordNetLemmatizer()

    print('Class EXTRACTED :')

    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in words]
        tagged = nltk.pos_tag(lemmatized_tokens)
        for (word, tag) in tagged:
            if tag == 'NN':
                print(word)
                return word
            if tag == 'VBG':
                print(word)
                return word
            if tag == 'VBN':
                print(word)
                return word
            if tag == 'VBP':
                print(word)
                return word
            if tag == 'VBZ':
                print(word)
                return word
            if tag == 'NNP':
                print(word)
                return word
            if tag == 'VB':
                print(word)
                return word
            else:
                print("No class created as no POS found!")
                return "NoClass"


# insert to csv i.e., database
def write_csv(question: list(), answer: list(), _class: list()):
    # Open CSV file in append mode
    # Create a file object for this file
    with open('./Data/Retrieval_model_data.csv', 'a', newline='', encoding='utf-8') as f_object:
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get an object of DictWriter

        fieldnames = ['question', 'answer', 'class']  # Define the column names
        writer = csv.DictWriter(f_object, fieldnames=fieldnames)

        # Pass the dictionary as an argument to the Writerow()
        for i in range(len(question)):
            writer.writerow({'question': question[i], 'answer': answer[i], 'class': _class[i]})

        # Close the file object
        f_object.close()


def retrain():
    write_csv(QUESTIONSET, ANSWERSET, CLASSSET)
    stop_words = set(stopwords.words('english'))
    le = LE()
    tfv = TfidfVectorizer(min_df=1, stop_words='english')

    def clean(sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [w for w in word_tok if not w in stop_words]

        return ' '.join(stemmed_words)

    data = pd.read_csv("./Data/Retrieval_model_data.csv", encoding='utf-8')
    questions = data['question'].values
    print("===>Retrieval_model_data.csv loaded successfully.../")
    X = []

    for question in questions:
        X.append(clean(question))

    tfv.fit(X)  # transforming the questions into tfidf vectors
    le.fit(data['class'])  # label encoding the classes as 0, 1, 2, 3

    X = tfv.transform(X)
    y = le.transform(data['class'])

    model = SVC(kernel='linear')
    model.fit(X, y)

    file = 'Retrieval_model.pkl'
    pickle.dump(model, open(file, 'wb'))

    print("===>Model Retrained and Pickled successfully with :", round(model.score(X, y), 2) * 100, "% accuracy")


# noinspection PyUnresolvedReferences
def generative_model_inference(userinput: str, tokenizer: GPT2Tokenizer,
                               model: transformers.AutoModelForCausalLM) -> str:
    # create a prompt in compliance with the one used during training without the answer part
    prompt = f'{"<|startoftext|>"}question: {userinput}\nanswer:'
    # generate tokens
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    input_ids = input_ids.to(device)
    # predict response (answer)
    gt_len = len(userinput.split()) + 1
    response = model.generate(input_ids,
                              do_sample=True,
                              top_k=1,
                              min_new_tokens=gt_len,
                              max_new_tokens=64,  # 128
                              repetition_penalty=10.0,
                              length_penalty=-0.1,
                              pad_token_id=tokenizer.eos_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              top_p=1.0)
    # decode the predicted tokens into texts
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    answer = response_text.split('answer: ')[-1]

    contains_incomplete_sent = True
    if answer.endswith('.'):
        contains_incomplete_sent = False

    sents = answer.split('. ')
    if contains_incomplete_sent:
        sents.pop()

    cleaned_ans = '. '.join(sents).strip()
    cleaned_ans = cleaned_ans + '.'

    output = cleaned_ans.replace("the question. ", "")

    return output


# noinspection SpellCheckingInspection,PyUnusedLocal
def get_response(usrText):
    data, X, y, tfv, le = load_data()

    while True:

        if usrText.lower() == "bye":
            return "Bye"

        GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey", "hiii", "hii", "yo",
                           "Assalam o Alaikum"]

        a = [x.lower() for x in GREETING_INPUTS]

        sd = ["Thanks", "Welcome"]

        d = [x.lower() for x in sd]

        am = ["OK"]

        c = [x.lower() for x in am]

        t_usr = tfv.transform([cleanup(usrText.strip().lower())])
        class_ = le.inverse_transform(Retrieval_model.predict(t_usr))

        questionset = data[data['class'].values == class_]

        cos_sims = []
        for question in questionset['question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)

            cos_sims.append(sims)

        ind = cos_sims.index(max(cos_sims))

        b = [questionset.index[ind]]

        if usrText.lower() in a:
            return "Hi, I'm Nova representing Visiblity Bots!\U0001F60A"

        if usrText.lower() in c:
            return "Ok...Alright!\U0001F64C"

        if usrText.lower() in d:
            return "My pleasure! \U0001F607"

        if max(cos_sims) > 0.6:  # set this parameter values
            a = data['answer'][questionset.index[ind]] + "   "
            return a

        if 0.6 >= max(cos_sims) > 0.2:  # set this parameter values
            a = generative_model_inference(usrText, tokenizer, Generative_model)
            classes = ClassExtractor(usrText)
            QUESTIONSET.append(usrText)
            ANSWERSET.append(a)
            CLASSSET.append(classes)

            # write_csv(usrText, a, ClassExtractor(usrText))
            return a

        if 0.2 >= max(cos_sims) > 0.1:  # set this parameter values
            inds = get_max5(cos_sims)
            print(inds)

            b = "(1)" + data['question'][questionset.index[0]]
            c = "(2)" + data['question'][questionset.index[1]]
            d = "(3)" + data['question'][questionset.index[2]]
            e = "(4)" + data['question'][questionset.index[3]]
            f = "(5)" + data['question'][questionset.index[4]]

            return "I didn't understand \nDid you mean these Questions----->\n" + b + '\n' + c + '\n' + d + '\n' + e + '\n' + f


        elif max(cos_sims) == [[0.]]:
            return "sorry!. This query is out of scope. You can ask me regarding banking queries related to: Accounts, Investments, Funds, etc. or you can call to customer support 0301 7144499 \U0001F615"


if __name__ == "__main__":
    play('Hello i am NOVA calling from Visibility Bots. How can i help you?')
    while True:
        user_input = ASR()  # (input(">> User:"))
        if user_input == "bye":
            break
        if user_input == "empty":
            continue
        if user_input is None:
            continue
        if user_input == "":
            continue
        else:
            # print("Bot: {}".format(get_response(user_input)))
            reply = get_response(user_input)
            # torch.cuda.empty_cache()
            play(reply)
            # torch.cuda.empty_cache()

    retrain()
