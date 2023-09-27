# Analysis-and-Classification-of-Customer-Reviews-in-Arabic

## Methodology
###	Data Preprocessing
The goal of this process is to prepare the data for model training, which varies depending on the type of data. Each type of data requires different processing methods. In this stage, we cleaned the data and applied appropriate processing operations, such as removing numbers, links, non-Arabic characters, punctuation marks, emojis, diacritics from Arabic letters, eliminating stop words, and then stemming the words, along with other processing operations.
We used Python for the entire coding process. In this stage, we relied on data processing modules such as pandas, numpy, nltk, re, and string. We converted the output column into numbers, performing a mapping operation where each negative value was assigned -1, and each positive value was assigned 1. We saved the clean data in a new file for use in the subsequent steps. We did not need the data augmentation process because our data is balanced. 
### Features Extraction
Feature extraction is an important and fundamental step before creating and training machine learning and deep learning models. This step involves converting text into numbers so that algorithms can understand, analyze, and classify texts. There are many algorithms involved in this process (as discussed in section Two). For machine learning, we used the tf-idf algorithm to extract features from the texts before training the various algorithms we used. TF-IDF is often used to represent documents as vectors in a high-dimensional space, where each term corresponds to a dimension, and the TF-IDF score for each term is the value along that dimension. These vectors can then be used for various text analysis tasks.
We also experimented with the word2Vec method along with unigram and bigram approaches for feature extraction from texts, but only with the SVM algorithm (both linear and rbf) because we did not observe a significant difference between feature extraction methods. 
For CNN and LSTM neural networks, we extracted features using the tokenizer and pad_sequence. We also used them with BILSTM neural networks, but with the addition of W2V.
### Machine Learning
After preparing the dataset and extracting the features, we divided it into 80% training data and 20% testing data. We applied several different algorithms with various feature extraction methods as follows: <br>
 	1. SVM (kernel ‘linear’, C=1) with Tf_Idf.
  <br>
 	2. SVM (kernel ‘linear’, C=1) with W2V and Ngrams.
  <br>
 	3. SVM (kernel ‘rbf’, C=0.5) with W2V and Ngrams. <strong> note: </strong> C in SVM refers to regularization.
  <br>
 	4. KNN (k=5) with Tf_Idf.
  <br>
 	5. Random Forest (number of trees are 100) with Tf_Idf.
  <br>
 	6. Decision Tree with Tf_Idf.
  <br>
 	7. Logistic Regression with Tf_Idf.
  <br>
### Deep Learning
For deep learning, we split the data into 80% training, 10% validation, and 10% testing. We created several models using neural networks: CNN, LSTM, and BILSTM.
  <br>
First model: This model consists of six layers in the following order: embedding, conv1d, global max pooling, dense (64), dropout (0.5), and dense (1) as an output layer with a sigmoid function. The model was trained with a learning rate (lr = 0.001) using the Adam optimizer and techniques like early stopping and reduce learning rate. It was trained for ten epochs (epochs = 10) with a batch size of 32.
  <br>
Second model: This model consists of three layers in the following order: embedding, LSTM (64 units with dropout = 0.2) and dense (1) as an output layer with a sigmoid function. The model was trained with a learning rate (lr = 0.001) using the Adam optimizer. It was trained for ten epochs (epochs = 10) with a batch size of 64.
  <br>
Third model: This model consists of three layers in the following order: embedding, BILSTM (64 units with dropout = 0.2) and dense (1) as an output layer with a sigmoid function. The model was trained with a learning rate (lr = 0.001) using the Adam optimizer. It was trained for ten epochs (epochs = 10) with a batch size of 64. We used learning rate scheduler to reduce learning rate within training.
 <br>
Fourth model: This model consists of eight layers in the following order: embedding, conv1d (256), max pooling (5), LSTM (128), BILSTM(64), dense (128), dropout (0.5), and dense (1) as an output layer with a sigmoid function. The model was trained with a learning rate (lr = 0.001) using the Adam optimizer and techniques like early stopping and reduce learning rate. It was trained for ten epochs (epochs = 10) with a batch size of 32.
 <br>
The last model: This model consists of three layers in the following order: embedding, BILSTM (128 units with dropout = 0.2) and dense (1) as an output layer with a sigmoid function. The model was trained with a learning rate (lr = 0.001) using the Adam optimizer. It was trained for ten epochs (epochs = 10) with a batch size of 32. We used learning rate scheduler to reduce learning rate within training.
