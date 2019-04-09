import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/AlteGrad/for_kaggle_final")

import numpy as np
from sklearn.decomposition import PCA
from helper import load_obj, load_docs
from AttentionWithContext import AttentionWithContext
import sys
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense
from keras.layers import concatenate
from CyclicLR import CyclicLR
import pandas as pd

# saving graph/doc embeddings
path_to_data = 'fill me'



#####==================PCA===========================##############
gl_embds = np.save(path_to_data + 'graph_mv.npy')
k = 50
pca = PCA(n_components=k)
gl_pca = pca.fit_transform(gl_embds)
SAVE = True
if SAVE:
  np.save(path_to_data + 'gl_pca.npy', gl_pca)
del gl_embds


graphs = load_obj(path_to_data, 'graphs')
# Rearranging the embeddings into the initial order
SAVE = False
if SAVE:
  print('unstructued emveddings')
  ml_embds = np.load(path_to_data + 'ml_embds_mv.npy')
  print('original embeddings')
  embds = np.load(path_to_data + 'embeddings.npy') 
  
  print('building embeddings matrix...\n')
  embeddings_mv = np.zeros((embds.shape[0], ml_embds.shape[2]))
  for i, graph in enumerate(graphs):
    for j, node in enumerate(graph.nodes()):
      embeddings_mv[node] = ml_embds[i, j, :]
    
    sys.stderr.write('\rGraph: %d/%d' % (i+1, len(graphs)))
    sys.stderr.flush()
    
  print('Save....')
  del ml_embds
  #del embds
  np.save(path_to_data + 'embeddings_mv.npy', embeddings_mv)
  
else:
   embeddings_mv = np.load(path_to_data + 'embeddings_mv.npy')


# saving node/word embeddings
k = 50
_pca = PCA(n_components=k)
embds_pca = _pca.fit_transform(embeddings_mv)
if SAVE:
  print('saving')
  np.save(path_to_data + 'embds_pca.npy', embds_pca)
del embeddings_mv

docs = load_docs(BUILD_DOCS = False, pad_vec_idx = 1685894, num_walks = 5, 
              walk_length = 10, max_doc_size = 70, path_to_data=path_to_data)


EMBEDDINGS_MV = np.c_[embds, embds_pca]

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
train_idxs = [int(elt) for elt in train_idxs]
    
# create validation set
np.random.seed(12219)
idxs_select_train = np.random.choice(range(len(train_idxs)),size=int(len(train_idxs)*0.80),replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)),idxs_select_train)


docs_opt = docs[train_idxs]
doc_embds_train = gl_pca[train_idxs]

print('Loading all the targets......\n')
tgts = [0,1,2,3,]
targets = []

for tgt in tgts:
    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()
        targets.append(target)

######==========================FINAL MODEL ================####################"

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

def make_model(n_units, drop_rate, is_GPU, embeddings_mv, my_optimizer, nb_targets, doc_embds=None):


    # = = = = = defining architecture = = = = =

    sent_ints = Input(shape=(docs_opt.shape[2],))

    sent_wv = Embedding(input_dim=EMBEDDINGS_MV.shape[0],
                        output_dim=embeddings_mv.shape[1],
                        weights=[embeddings_mv],
                        input_length=docs_opt.shape[2],
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU)
    sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
    sent_encoder = Model(sent_ints,sent_att_vec_dr)

    doc_ints = Input(shape=(docs_opt.shape[1],docs_opt.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU)
    doc_att_vec,sent_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)
    _dense1 = Dense(50, activation='relu')(doc_att_vec_dr)
    _dense2 = Dense(30, activation='relu')(_dense1)
    if doc_embds is not None:
      enriched_docs = Input(shape=(doc_embds.shape[1],))
      dense1 = Dense(100, activation='relu')(enriched_docs)
      dense2 = Dense(100, activation='relu')(dense1)
      last_layer = concatenate([_dense2, dense2], axis=1)
      

      preds = Dense(units=nb_targets,
                  activation='linear')(last_layer)
    else:
      preds = Dense(units=nb_targets,activation='linear')(doc_att_vec_dr)
      
    if doc_embds is not None:
      model = Model([doc_ints, enriched_docs], preds)
    else:
      model = Model(doc_ints,preds)

    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer,
                  metrics=['mae'])

    return model

# = = = = = hyper-parameters = = = = =

N_UNITS = 50
DROP_RATE = 0.5
IS_GPU = True
MY_OPTIMIZER = 'adam'



BATCH_SIZE = 512
NB_EPOCHS = 15
MY_PATIENCE = 4
#=====================================

model = make_model(N_UNITS, DROP_RATE, IS_GPU, EMBEDDINGS_MV, MY_OPTIMIZER, 4, doc_embds=doc_embds_train)
reduce_lr = CyclicLR(base_lr=0.001, max_lr=0.01,step_size=2000.)
ckpt = ModelCheckpoint(path_to_data+'weights_all.h5', save_best_only=True, save_weights_only=True, verbose=1, monitor='val_loss', mode='min')
model.fit([docs_opt, doc_embds_train], targets.T, epochs=NB_EPOCHS, batch_size=512, validation_split=0.2, verbose=1, callbacks=[ckpt, reduce_lr]) 

#==============PREDICTIONS===============================#
with open(path_to_data + 'test_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()
    
test_idxs = [int(elt) for elt in train_idxs]
docs_test = docs[test_idxs,:,:]
doc_embds_test = gl_pca[test_idxs]

kaggle_han_mv = model.predict([docs_test, doc_embds_test])
predictions_han = np.hstack((kaggle_han_mv[:,0], kaggle_han_mv[:,1], kaggle_han_mv[:,2], kaggle_han_mv[:,3]))


ids = np.array(range(predictions_han.shape[0]))
predictions_han_mv = pd.DataFrame({'id':ids, 'pred':predictions_han})
predictions_han_mv.to_csv(path_to_data +'predictions.txt', index=False)




