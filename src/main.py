print('Ready')

from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
embeddings = model.encode(sentences)
print(embeddings)