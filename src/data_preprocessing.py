import pandas as pd
import numpy as np
import gensim
import requests
import os
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler, LabelEncoder

def download_and_load_word2vec_model(url, model_path):
    if not os.path.exists(model_path):
        print("Downloading Persian Word2Vec model...")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    
    print("Loading model...")
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False, unicode_errors='ignore')
    print("Model loaded successfully.")
    return word2vec_model

def encode_comment(comment, model):
    tokens = comment.split()  # Simple tokenization; Persian might need a more sophisticated approach
    vectors = [model[token] for token in tokens if token in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Return a zero vector if no tokens are found
    return np.mean(vectors, axis=0)  # Average the vectors

def add_comment_embeddings(df, model):
    df['Comment_Embedding'] = df['Comment'].apply(lambda x: encode_comment(str(x), model))
    return df

def preprocess_data(config):
    train_df = pd.read_csv(config['data']['train_path']).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    test_df = pd.read_csv(config['data']['test_path']).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

    # Download and load the Word2Vec model
    word2vec_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz"
    word2vec_path = "cc.fa.300.vec.gz"
    word2vec_model = download_and_load_word2vec_model(word2vec_url, word2vec_path)

    # Add comment embeddings to the datasets
    train_df = add_comment_embeddings(train_df, word2vec_model)
    test_df = add_comment_embeddings(test_df, word2vec_model)

    # Convert the embeddings to a format suitable for modeling
    comment_embeddings_train = np.vstack(train_df['Comment_Embedding'].values)
    comment_embeddings_test = np.vstack(test_df['Comment_Embedding'].values)

    # Merge the embeddings back into the original DataFrame
    train_df = pd.concat([train_df.drop(columns=['Comment_Embedding']), pd.DataFrame(comment_embeddings_train)], axis=1)
    test_df = pd.concat([test_df.drop(columns=['Comment_Embedding']), pd.DataFrame(comment_embeddings_test)], axis=1)

    # Encode categorical features
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.drop(columns=['Label']))
    X_test = scaler.transform(test_df.drop(columns=['Label']))
    
    y_train = train_df['Label']
    y_test = test_df['Label']

    return X_train, X_test, y_train, y_test
