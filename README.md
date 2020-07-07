# NLP Techniques with Shakespeare's Plays

One of the most interesting (if dubious) literary theories is that Shakespeare could not have written his plays. In this notebook I will use several modeling techniques to develop a model that can predict if text was or was not written by Shakespeare. 

### Technologies
* Pandas, Numpy, Scikit-learn, NLTK, XGBoost, Keras, Gensim, pyLDAvis, pprint, string, Regex, os, shutil, random, pickle, wordcloud, seaborn, matplotlib, pillow

## Data Process
I have outlined my data process below. The content of steps 2-5 on this process chart are separated into Notebooks 1-4.

![Data Process](/Images/DataProcess.PNG)


## Notebook 1: Data Cleaning
My data consists of 39 Shakespeare plays and plays written by other Elizabethan-era playwrights, all of which were downloaded as .txt files directly from Project Gutenberg. I am using the main historical body of 37 texts plus Two Noble Kinsmen and Edward III, which are typically partially attributed to Shakespeare. The other playwrights were chosen based on their writing contemporaneously to Shakespeare during the reign of Queen Elizabeth I and King James. 

### Cleaning Steps
* Spliting into test/train sets with 9 plays for each test class, 30 in the training Shakespeare class and 41 in the training non-Shakespeare class
* Removing stopwords, including Elizabethan-era stopwords such as dost and thou
* Removing editor's notes and any other prologues, legal appendices, and footnotes
* Tagging parts of speech so words can be lemmatized

### Data Representations
I am using two different representations of my data. The first is TF-IDF, which is a bag-of-words representation that creates a "feature" for every ngram in our corpus using term-frequency/inverse document freqency. The second is GloVe, which uses the global and local representations of words and their cooccurrence in order to generate a vector representation. 
* TF-IDF
* TF-IDF with 80% explained variance, using dimensionality reduction
* TF-IDF of Bigrams
* Word Embeddings using GloVe weights - 300d
 
## Notebook 2: Data Exploration
Initially I wanted to use EDA to ensure that there would be a reasonable expectation that a model would be able to differentiate between the plays, and wanted to understand better myself what those differences were. We can see from the below that the most common words in our training datasets are fairly different by class. 

![Word Frequency](/Images/Word_Frequency.png)

I used t-SNE to visualize high-dimensional data in 2-dimensions, transforming my TF-IDFs to view the divisions in classes for our unigrams.

![TF-IDF of Unigrams](/Images/Unigram_TFIDF.png)

One more interesting visualization is our corpus passed through a Word2Vec model, identifying word similarities based on our text visualized here by distance. I also used t_SNE to visualize this data, producing a data point for each of our common words. 

## Notebook 3: Supervised Classification Models

I used both TF-IDF and Word Embeddings with Random Forest, Naive Bayes, Support Vector Machines, XGBoost and RNNs to classify my data. The best performing models were my XGBoost and Recurrent Neural Network, both of which achieved a 100% accuracy, correctly classifying plays in the test data set. The XGBoost used the degault parameters and the full TF-IDF unigram data. The RNN is an LSTM(Long Short Term Memory) network, making it capable of retaining relevant information and forgetting irrelevant information.

## Notebook 4: Unsupervised Topic Modeling

I conducted unsupervised modeling by generating topics from my data. I initially generated 7 topics which had a coherence score (c_v) of .298, which did not improve by making changes to the number of topics. However, it is interesting to explore the pyLDAvis interactive chart, which enables us to view the distance of topics to one another, and to view the contributing tokens. 

![Topic Modeling pyLDAvis](/Images/Topic_Modeling.PNG)

## Conclusion 

My classification exercise was very successful given the size of the dataset, showing how well these processing and modeling techniques work even though we are working with Early Modern English and a dataset of limited size. There are a number of directions that can be taken with this project, such as the addition of data, the creation of recommendation systems for the non-Shakespeare dataset based on feedback and features on review sites, and experimentation with other pre-trained weights. 
