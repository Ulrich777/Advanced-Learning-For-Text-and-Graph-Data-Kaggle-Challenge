# Advanced-Learning-For-Text-and-Graph-Data-Kaggle-Challenge

In this challenge we were asked to predict continuous values associated with a graph. The rule obliged to address it 
like an NLP problem with a self-attention mecanism (HAN). Indeed a graph can be seen like a document by sampling many random walks out 
of it. However we proceded in two steps. We first enrich our node embeddings or word embeddings using a multi-layer graph convoluational 
network allowing us to have both node (word) and graph (doc) embeddings. Second we reduce our embeddings with classical PCA. 
At last we merge these node embeddings with the original ones and modify the initial HAN architecture to account for our learned 
documents embeddings.

At the end, we were ranked 4/46.
