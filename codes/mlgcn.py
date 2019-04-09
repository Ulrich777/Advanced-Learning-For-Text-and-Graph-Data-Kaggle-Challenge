from math import ceil
import tensorflow as tf
import numpy as np
import sys

class MLGCN():
    
    def __init__(self, node_dim=2, graph_dim=[3,3], dim_o=2, nmax=15):
        """
        Parameters of the model architecture
        
        """
        self.graph_dims = [node_dim] + graph_dim
        self.n_layers = len(graph_dim)
        self.dim_o = dim_o
        self.nmax = nmax
        self.scores = []
        self.scores_val = []
        
        self.is_built = False
        
    def build_model(self):
        epsi = 1e-04
     
        
        self.adjs = tf.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])
        self.targets = tf.placeholder(tf.float32, shape=[None, self.dim_o])
        self.alpha = tf.placeholder(tf.float32)
        
        self.A = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \
             for i in range(self.n_layers)}
        self.B = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \
             for i in range(self.n_layers)}
        self.W  = tf.Variable(tf.random_normal([sum(self.graph_dims[1:]), self.dim_o]))
        #self.W  = tf.Variable(np.random.normal(0,1,size=(sum(self.graph_dims[1:]), self.dim_o)))
        
        
        self.b = tf.Variable(tf.zeros([self.dim_o]))
        
        self.M, self.H, self.pM, self.G = {}, {}, {}, {}
        if self.batch_norm:
            self.m, self.var, self.nM = {}, {}, {}
            self.gamma = {i+1: tf.Variable(tf.ones([self.nmax,self.graph_dims[i+1]])) for i in range(self.n_layers)} 
        
        self.H[0] = tf.placeholder(tf.float32, shape=[None, self.nmax, self.graph_dims[0]])
        
        for i in range(1, self.n_layers+1):
        
            self.M[i] = tf.einsum('adc,adb->abc', self.H[i-1], self.adjs)
            self.pM[i] = tf.tensordot(self.M[i], self.A[i], (2, 1)) + tf.tensordot(self.H[i-1], self.B[i], (2, 1))

            if self.batch_norm:
                self.m[i], self.var[i] = tf.nn.moments(self.pM[i], 0)
                self.nM[i] = tf.multiply((self.pM[i] - self.m[i]) /  tf.sqrt(self.var[i] + epsi ) , self.gamma[i])  #*self.gamma[i]
                self.H[i] = tf.nn.sigmoid(self.nM[i])-0.5
                
            else:
                self.H[i] = tf.nn.sigmoid(self.pM[i])-0.5
            
            self.G[i] = tf.reduce_mean(self.H[i], 1)
            
        
        self.graph_features = tf.concat([self.G[i] for i in range(1, self.n_layers+1)], 1)
        self.G_out = tf.nn.dropout(self.graph_features, self.keep_rate)
        
        #self.G_out = tf.nn.dropout(tf.concat([self.G[i] for i in range(1, self.n_layers+1)], 1), self.keep_rate)
        
        self.new_features = tf.concat([self.H[i] for i in range(1, self.n_layers+1)], 2)

        Y_OUT = tf.matmul(self.G_out, self.W) + self.b
        #self.cost = tf.reduce_mean(tf.square(self.targets - Y_OUT))
        self.cost = tf.losses.mean_squared_error(self.targets, Y_OUT)
        
        self.predictions = Y_OUT
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.train = optimizer.minimize(self.cost)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def fit(self, data, epochs=20, batch_size=64, shuffle=True, data_val = None, 
            batch_norm=False, keep_rate=0.75, alpha=0.025):
      
      
        self.batch_norm = batch_norm
        self.keep_rate = keep_rate
        
        if not self.is_built:
            self.build_model()
            self.is_built = True
        
        
        adj = data[0]
        embds = data[1]
        y = data[2]
        
        if data_val is not None:
          adj_val = data_val[0]
          embds_val = data_val[1]
          y_val = data_val[2]
        minibatches = ceil(adj.shape[0]/batch_size)
        
        
        
        j = 0
        for i in range(epochs):
            INDS = np.array(range(len(adj)))
            
            if shuffle:
                idx = np.random.permutation(y.shape[0]) 
                INDS = INDS[idx]
                
            mini = np.array_split(INDS, minibatches)
            
            for inds in mini:
                j+=1
                sys.stderr.write('\rEpoch: %d/%d -- Iteration %d/%d ' % (i+1, epochs, j, epochs*minibatches))
                sys.stderr.flush()
                self.sess.run(self.train, feed_dict={self.adjs:adj[inds], self.H[0]:embds[inds], 
                                                self.targets:y[inds], self.alpha:alpha})
                
            cost = self.sess.run(self.cost, feed_dict={self.adjs:adj, self.H[0]:embds, 
                                                self.targets:y})
            self.scores.append(cost)
            
            if data_val is not None:
                cost_val = self.sess.run(self.cost, feed_dict={self.adjs:adj_val, self.H[0]:embds_val, 
                                                               self.targets:y_val})
                self.scores_val.append(cost_val)
            
        
        
    def predict(self, adj, embds):
        return self.sess.run(self.predictions, feed_dict={self.adjs:adj, self.H[0]:embds})
      
    def extract_node_features(self, adj, embds):
        return self.sess.run(self.new_features, feed_dict={self.adjs:adj, self.H[0]:embds})

    def extract_graph_features(self, adj, embds):
        return self.sess.run(self.graph_features, feed_dict={self.adjs:adj, self.H[0]:embds})
      
      
    def score(self, adj, embds,y):
        y_ = self.predict(adj, embds)
        return ((y_-y)**2).mean()       


