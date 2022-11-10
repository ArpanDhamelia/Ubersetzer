


**Pretrained Translator for Python Package Index**


# **Index**



|**Chapter No.**|**Contents**|**Page No.**|
| :- | :-: | :- |
|1|Introduction|1|
||1.1 Background||
||1.2 Scope of the project||
||1.3 Project Objectives||
||1.4 Problem Statement||
|2|Literature Review|3|
|3|System Description|4|
||3.1 System block diagram||
||3.2 Architectural details||
|4|Timeline chart|6|
|5|Implementation|7|
||5.1 Dataset details||
||5.2 Working of project||
||5.3 Code||
|6|Results and Discussions|13|
|7|Conclusion and Future Scope|16|
||References|17|

**List of Abbreviations**



|**Sr. No.**|**Abbreviation**|**Full Form**|
| :- | :- | :-: |
|1.|RNN|Recurrent Neural Network|
|2.|POS|Product of Sum|
|3.|PIP|preferred installer program|
|4.|MT|Machine Translation|
|5.|NLP|Natural Language Processing|

# **List of Figures**



|**Fig. No.**|**Figure Name**|**Page No.**|
| :- | :-: | :- |
|3.1.1|System Design|4|
|3.2.1|Architectural Design|5|
|4.1.1|`     `Gantt Chart|6|
|5.2.1|Attributions Details|7|
|5.2.2|Tokenization|8|
|6.1|Loss Epoch Graph|13|
|6.2|Uploading Pre trained translator model to Python Package Index|13|
|6.3|Importing Translator Model through Python Package Index|14|
|6.4|Translated Output|15|

**List of Tables**



|**Table. No.**|**Table Name**|**Page No.**|
| :- | :-: | :- |
|2.1|Paper Review|3 |


# **CHAPTER 1 INTRODUCTION**

1. **Background**

`	`Breaking language barriers through machine translation (MT) is one of the most important ways to bring people together. Translator models are one of the most frequently used systems, a typical MT systems require building separate AI models for each language and each task, but this approach doesn’t scale effectively 

`	`One of the biggest hurdles of building a translator model is curating large volumes of quality sentence pairs (also known as parallel sentences) for arbitrary translation directions. It’s a lot easier to find translations for Chinese to English and English to French, than, say, French to Chinese. What’s more, the volume of data required for training grows quadratically with the number of languages that we support. For instance, if we need 10M sentence pairs for each direction, then we need to mine 1B sentence pairs for 10 languages and 100B sentence pairs for 100 languages. 

`	`Python package index repository for python is the largest machine learning libraries in python, but it does not have ready to use pre-trained model in its repository, the goal of this project is to create a pre-trained translator model which removes the need for training the neural network in order to have a functional translator through python package index repository.

1. **Scope of the Project**

Integration of pre-trained models is guaranteed in pip – Pip Repository is an open platform which enables anyone to upload a package. Thus enabling the pre trained translator model to be accessed by the open source community through “pip install”. Hence the developed pretrained translator model is exported to python package index repository only.

Integration of a pre-trained model is uncertain in sklearn library as it is controlled by the community. The pull request needs to be verified by the Project Contributors of sklearn on Github and then merged into the main branch of python package index. Hence the uncertainty of getting accepted into the sklearn exists.


The translator model is capable of translating only German to English. Dataset consists only of German to English translated sentences, hence the RNN model was only trained to translate sentences from German to English. 

Due to limitations on upload size in pip, the translation package size is only 100MB, which restricts the training of the translator model. Due to the dataset's lack of language variation, contextual performance suffers. The translator only has the ability to translate from German to English.


1. **Project Objectives**
- To build an 1-1 Translator model using a neural network.
- To export the trained model as a class in python package index.
- To deploy ready to use translator model in Python Package Index as an open source contribution.

1. **Problem Statement**

To develop a neural network-based pre-trained translator model for the Python Package Index that will allow users to skip the model training phase, thus increasing the versatility of Python Package Index.
` `PAGE 13
17![](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.002.png)
# **CHAPTER 2**

## **LITERATURE REVIEW**
##

|**Sr No.**|**Title**|**Paper Review**|**Research Gaps Identified**|
| :-: | :-: | :-: | :-: |
|[1]|K. Jiang and X. Lu, "Natural Language Processing and Its Applications in Machine Translation: A Diachronic Review," 2020 IEEE 3rd International Conference of Safe Production and Informatization (IICSPI), 2020, pp. 210-214, doi: 10.1109/IICSPI51290.2020.9332458.|Understanding use of NLP for translators.|<p>- The rapid advancements in natural language processing provides strong support for machine translation research. </p><p>- This paper reviews the history and progress of NLP research at home and abroad. </p><p>- The author also discusses the relationship between machine translation and human translation in the age of artificial intelligence, and visualizes the future prospect of machine translation.</p>|
|[2]|J. -U. Bang, M. -K. Lee, S. Yun and S. -H. Kim, "Improving End-To-End Speech Translation Model with Bert-Based Contextual Information," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 6227-6231, doi: 10.1109/ICASSP43922.2022.9746117.|How BERT helps in improving end to end speech translation.|<p>- Conventional end-to-end speech translation primarily designed to handle single utterances. </p><p>- This paper introduces a context encoder that extracts contextual information from previous translation results. </p><p>- Contextual information contributes to processing of unclearly spoken utterances as well as ambiguity caused by pronouns and homophones.</p>|
|[3]|M. Wolfel, M. Kolss, F. Kraft, J. Niehues, M. Paulik and A. Waibel, "Simultaneous machine translation of german lectures into english: Investigating research challenges for the future," 2008 IEEE Spoken Language Technology Workshop, 2008, pp. 233-236, doi: 10.1109/SLT.2008.4777883.|<p>Simultaneous machine translation of German lectures into English: Investigating research challenges for the future.</p><p></p>|<p>- A large vocabulary and strong variations in speaking style make lecture translation a challenging task. </p><p>- In many cases, human interpreters are prohibitively expensive or simply not available. </p><p>- We report our progress in building an end-to-end system and analyze its performance in terms of objective and subjective measures.</p>|
Table 2.1: Paper Review
# **CHAPTER 3**

**System Description**


1. **System block diagram**

![https://lh5.googleusercontent.com/HTH5bRaLhmG73g3\_RlaOQ9INz9yquTEw26kxmTkrtVdVA5gA3czkhZdOvEnCYt8w9utp\_l21XXZmkLgKWB4qE3QXE3vqHpEe86\_mK0322jmzZG6Z4HJJEaJUUVxo7z\_FR2x659g9C2MYf-Bb8plzvTANx3kxXczl0TTzTwpDOa\_mx\_\_SxZ4DPKf7oYk8IekceOyyKg](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.003.png)

Fig 3.1.1 System Design

The input sequence is split into a sequence of words which is fed into the RNN model to determine its context, meaning and determine its equivalent English word to form a meaningful English sentence. The spatial content of each individual frame is less significant than the temporal dynamics that connect the data whenever there is a succession of data. In a RNN the information cycles through a loop. When it makes a decision, it considers the current input and also what it has learned from the inputs it received previously.

RNNs are able to accurately forecast what will happen next because of their internal memory, which helps them to retain key details about the input they received. For sequential data such as time series, speech, text, financial data, audio, video, weather, and many more types, they are the algorithm of choice. Compared to other algorithms, recurrent neural networks can develop a far deeper grasp of a sequence and its environment.





1. **Architectural details**

![https://lh6.googleusercontent.com/LhivdBEUqWegvdGcQZc0Qe\_eXQ7I6JoaatU36o5OfkaBFTpMOfzDgvseX9gyriT8-XLVM1SIuC-vDAza8YQyTQ97QF06OjEifB3A1TY3PbnKe1XilNQFCjPkKSItcjjOQl9\_YXnVrrv\_6XCyZtXD4fHZnHVcPcQl5Teo1wAjj-HozobFkUciDUrZNSsm2CZmS8zaXg](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.004.png)

Figure 3.2.1 Architectural Design

The German sentence *“Det Hund ist glucklich”* is forwarded to encoder W. Encoder POS enables RNN to understand the context of the sentence. Hence the result of both the encoders is combined in order to determine the meaning of the sentence, before being decoded into its equivalent English sentence.



# **CHAPTER 4**

**TIMELINE CHART**

![](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.005.png)

Figure 4.1.1: Gantt chart

The figure 4.1.1 shows a Gantt chart which consists of 10 activities and 5 activity descriptors. It shows the progress and the completion percentage of each activity.
# **CHAPTER 5**

## **IMPLEMENTATION**

1. **Dataset Details**

The dataset has been extracted from *tatoeba.org.* The file was created on 22-09-2022. The dataset consists of German words and their respective English translations. The dataset also has 3 column heads and 1,50,000 rows containing sentence pairs of German-English words. To lower the number of errors, only sentences by native speakers and proofread sentences have been included. For the non-English language, the following (possibly wrong) assumptions have been made:

Assumption 1: Sentences written by native speakers can be trusted.

Assumption 2: Contributors to the Tatoeba Project are honest about what their native language is.

1. **Working of the project**

![https://cdn.discordapp.com/attachments/996807239027396675/1029424531196092446/unknown.png](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.006.png)

Figure 5.2.1: Attribution Details

The figure shows attribution details of the dataset used for developing the pertained translator model, it shows the english sentence and equivalent german sentence and the contribution details.

![https://cdn.discordapp.com/attachments/996807239027396675/1029424799119835136/unknown.png](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.007.png)

Figure 5.2.2: Tokenization

Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. One can think of a token as parts like a word is a token in a sentence, and a sentence is a token in a paragraph. It significantly affects the remainder of the pipeline. Tokenization is the process of dividing unstructured data and natural language text into units of data that can be regarded as discrete pieces. One can directly utilize a document's token occurrences as a vector to represent the document. This instantly converts a text document or unstructured string into a numerical data format appropriate for machine learning. They can also be directly employed by a computer to initiate helpful answers and actions. They could also be employed as features in a machine learning pipeline to initiate more complicated actions or judgments.



1. **Code**

**Training NLP Model**

import string

import re

from numpy import array, argmax, random, take

import numpy as np

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.sequence import pad\_sequences

from tensorflow.keras.models import load\_model

from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

\# function to read raw text file

def read\_text(filename):

`    `# open the file

`    `file = open(filename, mode='rt', encoding='utf-8')

`    `# read all text

`    `text = file.read()

`    `file.close()

`    `return text

\# split a text into sentences

def to\_lines(text):

`    `sents = text.strip().split('\n')

`    `sents = [i.split('\t') for i in sents]

`    `return sents

data = read\_text("deu.txt")

deu\_eng = to\_lines(data)

deu\_eng = array(deu\_eng)

deu\_eng = deu\_eng[:100000,:]

deu\_eng

\# Remove punctuation 

deu\_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu\_eng[:,0]]

deu\_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu\_eng[:,1]]

\# convert to lowercase

for i in range(len(deu\_eng)):

`    `deu\_eng[i,0] = deu\_eng[i,0].lower()



`    `deu\_eng[i,1] = deu\_eng[i,1].lower()

\# empty lists

eng\_l = []

deu\_l = []

\# populate the lists with sentence lengths

for i in deu\_eng[:,0]:

`    `eng\_l.append(len(i.split()))

for i in deu\_eng[:,1]:

`    `deu\_l.append(len(i.split()))

print(max(deu\_l))

print(max(eng\_l))

\# function to build a tokenizer

def tokenization(lines):

`    `tokenizer = Tokenizer()

`    `tokenizer.fit\_on\_texts(lines)

`    `return tokenizer

\# prepare english tokenizer

eng\_tokenizer = tokenization(deu\_eng[:, 0])

eng\_vocab\_size = len(eng\_tokenizer.word\_index) + 1

eng\_length = 8

print('English Vocabulary Size: %d' % eng\_vocab\_size)

\# prepare Deutch tokenizer

deu\_tokenizer = tokenization(deu\_eng[:, 1])

deu\_vocab\_size = len(deu\_tokenizer.word\_index) + 1

deu\_length = 8

print('Deutch Vocabulary Size: %d' % deu\_vocab\_size)

\# encode and pad sequences

def encode\_sequences(tokenizer, length, lines):

`    `# integer encode sequences

`    `seq = tokenizer.texts\_to\_sequences(lines)

`    `# pad sequences with 0 values

`    `seq = pad\_sequences(seq, maxlen=length, padding='post')

`    `return seq

import pickle

with open('eng\_tokenizer.pickle', 'wb') as handle:

`    `pickle.dump(eng\_tokenizer, handle, protocol=pickle.HIGHEST\_PROTOCOL)

with open('deu\_tokenizer.pickle', 'wb') as handle:

`    `pickle.dump(deu\_tokenizer, handle, protocol=pickle.HIGHEST\_PROTOCOL)

from sklearn.model\_selection import train\_test\_split

train, test = train\_test\_split(deu\_eng, test\_size=0.05, random\_state = 1)

\# prepare training data

trainX = encode\_sequences(deu\_tokenizer, deu\_length, train[:, 1])

trainY = encode\_sequences(eng\_tokenizer, eng\_length, train[:, 0])

\# prepare validation data

testX = encode\_sequences(deu\_tokenizer, deu\_length, test[:, 1])

testY = encode\_sequences(eng\_tokenizer, eng\_length, test[:, 0])

\# build NMT model

def build\_model(in\_vocab, out\_vocab, in\_timesteps, out\_timesteps, units):

`    `model = Sequential()

`    `model.add(Embedding(in\_vocab, units, input\_length=in\_timesteps, mask\_zero=True))

`    `model.add(LSTM(units))

`    `model.add(RepeatVector(out\_timesteps))

`    `model.add(LSTM(units, return\_sequences=True))

`    `model.add(Dense(out\_vocab, activation='softmax'))

`    `return model

model = build\_model(deu\_vocab\_size, eng\_vocab\_size, deu\_length, eng\_length, 512)

rms = optimizers.RMSprop(learning\_rate=0.001)

model.compile(optimizer=rms, loss='sparse\_categorical\_crossentropy')

#@title Default title text

filename = 'arpanKaModel'

checkpoint = ModelCheckpoint(filename, monitor='val\_loss', verbose=1, save\_best\_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 

`          `epochs=30, batch\_size=512, 

`          `validation\_split = 0.2,

`          `callbacks=[checkpoint], verbose=1)

model = load\_model('arpanKaModel')

preds = np.argmax(model.predict(testX),axis=-1)

def get\_word(n, tokenizer):

`    `for word, index in tokenizer.word\_index.items():

`        `if index == n:

`            `return word

`    `return None

\# convert predictions into text (English)

preds\_text = []

for i in preds:

`    `temp = []

`    `for j in range(len(i)):

`        `t = get\_word(i[j], eng\_tokenizer)

`        `if j > 0:

`            `if (t == get\_word(i[j-1], eng\_tokenizer)) or (t == None):

`                `temp.append('')

`            `else:

`                `temp.append(t)



`        `else:

`            `if(t == None):

`                `temp.append('')

`            `else:

`                `temp.append(t)            



`    `preds\_text.append(' '.join(temp))

pred\_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds\_text})

pd.set\_option('display.max\_colwidth', 200)

pred\_df.sample(15)






**Trained Model Code:**

from tensorflow.keras.models import load\_model

from tensorflow.keras.preprocessing.sequence import pad\_sequences

import numpy as np

import pickle

import os

import string

from pathlib import Path

class eng\_deu:

`    `BASE\_DIR = Path(\_\_file\_\_).resolve().parent

`    `def \_\_init\_\_(self) -> None:

`        `self.model = load\_model(os.path.join(self.BASE\_DIR,'arpanKaModel'))

`        `with open(os.path.join(self.BASE\_DIR,'deu\_tokenizer.pickle'), 'rb') as handle:

`            `self.deu\_tokenizer = pickle.load(handle)

`        `with open(os.path.join(self.BASE\_DIR,'eng\_tokenizer.pickle'), 'rb') as handle:

`            `self.eng\_tokenizer = pickle.load(handle)

`    `def encode\_sequences(self,tokenizer, length, lines):

`        `seq = tokenizer.texts\_to\_sequences(lines)

`        `seq = pad\_sequences(seq, maxlen=length, padding='post')

`        `return seq

`    `def translate(self, sentence):

`        `sentence = sentence.translate(str.maketrans('', '', string.punctuation))

`        `sentence = sentence.lower()

`        `sentence = np.array([sentence])

`        `testX = self.encode\_sequences(self.deu\_tokenizer, 8, sentence)

`        `preds = np.argmax(self.model.predict(testX), axis=-1)

`        `preds\_text = []

`        `for i in preds:

`            `temp = []

`            `for j in range(len(i)):

`                `t = self.get\_word(i[j], self.eng\_tokenizer)

`                `if j > 0:

`                    `if (t == self.get\_word(i[j-1], self.eng\_tokenizer)) or (t == None):

`                        `temp.append('')

`                    `else:

`                        `temp.append(t)

`                `else:

`                    `if(t == None):

`                        `temp.append('')

`                    `else:

`                        `temp.append(t)

`            `preds\_text.append(' '.join(temp))

`        `return preds\_text[0]

`    `def get\_word(self,n, tokenizer):

`        `for word, index in tokenizer.word\_index.items():

`            `if index == n:

`                `return word

`        `return None
# **CHAPTER 6**

**RESULTS AND DISCUSSIONS**


![https://cdn.discordapp.com/attachments/996807239027396675/1029425128142024734/unknown.png](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.008.png)

Figure 6.1: loss-epoch graph

The above graph is the representation between validation loss and train loss slopes. X-axis represents epoch and Y-axis represents value loss. It tells us that as the epoch keeps on increasing the loss decreases till it becomes stagnant after a stage. It follows the ex graph.

![](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.009.png)

Figure 6.2: Uploading Pre trained translator model to Python Package Index

The figure 6.2 shows the process of uploading the pre-trained translator model to python package index. After building the translator package into an archive file, the package is ready to be exported to the python package index. This is done via ‘python -m twine upload –repository pypi dist/\*’ which starts uploading distributions to upload.pypi.org. The package size is 86.8 MB.

![](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.010.png)

Figure 6.3: Importing Translator Model through Python Package Index

The figure 6.3 shows the command line interface running on an independent 3rd party system. This system is accessing the pre-trained translator model through python package index through the use of PIP command. The name of the pretrained translator model is called Ubersetzer which means translator in german. The model can be installed from anywhere in the world through the online repository of pip.


![https://cdn.discordapp.com/attachments/996807239027396675/1029425885801107477/unknown.png](Aspose.Words.794107e1-044c-40a2-b6bc-71444aa9bde3.011.png)

Figure 6.4: Translated Output

The figure 6.4 shows the translator output which is a comparison between the actual sentence and the predicted sentence. It shows the result of 15 random samples taken and tested against the translator model. The predicted column shows good contextual understanding of the english sentences but has limited performance when the sentences contain words which are outside the training vocabulary of the translator model. This can be further improved by increasing vocabulary variety in the dataset and increasing the training time of the translator model.
# **CHAPTER 7**

1. **Conclusion**

An encoder and decoder architecture with embedding and lstm layers was used to create a pre-trained translator model. This model was uploaded to the Python Package Index Repository, allowing for the use of a functional translator without the requirement to train a neural network.

This was done with the help of Tatoeba.org who provided the German to English sentence pairs. And with the help of RNN the translator model we were able to translate the German words to English. The project has been successfully added as a python package, which can be called using the PIP module. 

1. **Future Scope**

As the internet continues to spread throughout developing nations worldwide, translation services will eventually encompass more cultural backgrounds. The software must offer precise translations for less widely spoken dialects in addition to the top languages for translation. 

The RNN model accuracy can be significantly improved with higher quality databases, longer training period and a denser neural network. The current translator model is only capable of German to English translation which makes it unidirectional in nature, hence it can be further improved by making a bidirectional model.

Translator package size is limited to only 100MB due to restrictions in upload size in pip, which limits the training of the translator model. Contextual performance is negatively affected due to low vocabulary variety in the dataset. The translator is a one way translator, capable of translating only German to English.

Importing from python package index through PIP is the only way for accessing the pre-trained translator model from the internet, as integration into the main Sklearn branch in Github requires verification of pull requests from the Community Contributors/Admins. 



# **REFERENCES**
#
[1] K. Jiang and X. Lu, "Natural Language Processing and Its Applications in Machine Translation: A Diachronic Review," 2020 IEEE 3rd International Conference of Safe Production and Informatization (IICSPI), 2020, pp. 210-214, doi: 10.1109/IICSPI51290.2020.9332458.

[2] J. -U. Bang, M. -K. Lee, S. Yun and S. -H. Kim, "Improving End-To-End Speech Translation Model with Bert-Based Contextual Information," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 6227-6231, doi: 10.1109/ICASSP43922.2022.9746117.

[3] M. Wolfel, M. Kolss, F. Kraft, J. Niehues, M. Paulik and A. Waibel, "Simultaneous machine translation of german lectures into english: Investigating research challenges for the future," 2008 IEEE Spoken Language Technology Workshop, 2008, pp. 233-236, doi: 10.1109/SLT.2008.4777883.

[4] Stackoverflow, [online]. Available:

https://stackoverflow.com/questions/32839293/locally-modifying-a-python-library-sklearn-linux

(Accessed: August 08, 2022)
