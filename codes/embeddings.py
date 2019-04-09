from helper import load_data, tensorize_graphs, load_docs
from mlgcn import MLGCN
import numpy as np

path_to_data = "C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/for_kaggle_final/work/"

graphs, embds, ADJ_OPT, EMB_OPT, ADJ_TEST, EMB_TEST, targets = load_data(nmax=29, RESET=False, 
                                                                         path_to_data=path_to_data)

# train MLGCN
mlgcn = MLGCN(node_dim=13, graph_dim=[100, 50,50], dim_o=4, 
             nmax=29)

ALPHAS = [0.025, 0.01, 0.005, 0.001, 0.025]

for alpha in ALPHAS:

    mlgcn.fit([ADJ_OPT, EMB_OPT, targets.T], epochs=500, batch_size=128, keep_rate=1, 
              batch_norm=False, alpha=0.025)
    
    
ADJ, EMB = tensorize_graphs(graphs, embds, nmax=29)
del ADJ_OPT, EMB_OPT

# saving graph/doc embeddings
gl_embds = mlgcn.extract_graph_features(ADJ, EMB)
print(gl_embds.shape)
SAVE = True
if SAVE:
  print('saving')
  np.save(path_to_data + 'graph_mv.npy',gl_embds)


ml_embds = mlgcn.extract_node_features(ADJ, EMB)
print(ml_embds.shape)
SAVE = True
if SAVE:
  print('saving')
  np.save(path_to_data + 'ml_embds_mv.npy', ml_embds)









