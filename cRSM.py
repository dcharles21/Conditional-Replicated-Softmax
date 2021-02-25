# cRSM code

# Load Libraries
import numpy as np
from preprocess import num_authors, data_train, data_test, vocab, Z_train, Z_test

# Sigmoid Function (Adjusted for exp overflow)
def sigmoid(x):   

    for b in x >= 0:
        if b:
            z = np.exp(-x)
            return 1/(1 + z)
        
        else: z = np.exp(x)
        return z/(1 + z)

# Exp for Softmax (Adjusted for exp overflow)
def exp_softmax(x):
    b = x.max()
    y = np.exp(x-b)

    return(y/y.sum())

# LogSumExp (Adjusted for exp overflow)
def LogSumExp(x):
    b = x.max()
    y = np.exp(x-b)

    return(b + np.log(np.exp(x-b).sum()))

# LogMeanExp(x)
def LogMeanExp(x):
  b = x.max()
  return b + np.log(np.exp(x-b).mean())

def train_cRSM(data, units, epochs, lr, momentum, Z, btsz):

        dictsize = len(vocab)

        # initilize weights
        w = 0.001 * np.random.randn(dictsize, units)
        b = 0.001 * np.random.randn(dictsize)
        a = np.zeros(units)

        # weight updates
        wu = np.zeros((dictsize, units))
        bu = np.zeros((dictsize))
        au = np.zeros((units))

        delta = lr/btsz
        batches = int(data.shape[0]/btsz)

        # Create Meta Weight Matrix and Meta Update Weight Matrix
        wm = 0.001 * np.random.randn(num_authors, units)
        wmu = np.zeros((num_authors, units))

        print("learning_rate: %f"%delta)
        print("updates per epoch: %s | total updates: %s"%(batches, batches*epochs))
        
        for epoch in range(epochs):
            print("Epoch", epoch)            

            for j in range(batches):
                start = j*btsz 

                # Set Visual Units to Training Data
                v1 = vt = data[start: start+btsz]

                # Set Meta Data Values to Training Data
                z = zt = Z[start: start+btsz]

                # Calculate Hidden Activations
                D = v1.sum(axis=1)                               
                h1 = np.dot(v1, w) + np.outer(D, a) + np.dot(z, wm)

                for i in range(0, len(h1)):
                    h1[i] = sigmoid(h1[i])               
                        
                for steps in range(0, 5):

                    # Calculate Hidden Activations
                    D = vt.sum(axis=1)                    
                    ht = np.dot(vt, w) + np.outer(D, a) + np.dot(z, wm)

                    for i in range(0, len(ht)):
                        ht[i] = sigmoid(ht[i])     
                            
                    # Sample hiddens
                    h_rand = np.random.rand(btsz, units)
                    h_sampled = np.array(h_rand < h1, dtype=int)

                    # Calculate visible activations
                    v2 = np.dot(h_sampled, w.T) + b

                    v2_pdf = [None]*len(v2)
                    for i in range(0, len(v2)):
                        v2_pdf[i] = exp_softmax(v2[i])  
                        
                    # Sample D times from Multinomial
                    v2 *= 0

                    for i in range(btsz):
                        v2[i] = np.random.multinomial(D[i], v2_pdf[i], size=1) 
                        
                    vt = v2
                    
                # Use Activations, Not Sampling Here
                h2 = np.dot(v2, w) + np.outer(D, a) + np.dot(z, wm)

                for i in range(0, len(h2)):                   
                        h2[i] = sigmoid(h2[i])

                # Compute Updates
                wu = wu * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                bu = bu * momentum + v1.sum(axis=0) - v2.sum(axis=0)
                au = au * momentum + h1.sum(axis=0) - h2.sum(axis=0)
                wmu = wmu * momentum + np.dot(z.T, h1) - np.dot(z.T, h2)

                # Update 
                w += wu * delta 
                b += bu * delta
                a += au * delta
                wm += wmu * delta

        return w, b, a, wm

def AIS_cRSM(data, document, M, Z, step, units):

    # Initialize Inverse Temperatures
    beta = np.arange(0, 1, 1/step)
    S = len(beta)    
    
    # Initialize Importance Weights
    logW = np.zeros(M)

    # Set Uniform Distribution p0
    v_units = data.shape[1]
    p0 = np.ones(v_units)/v_units

    # Sample V1 from p0
    D = np.ones(M)*data[document].sum()

    V = np.zeros((M, v_units))
    for r in range(0, M):
        V[r] = np.random.multinomial(D[r], p0, size=1)    
      
    # Calculate p0(V1) 
    logW += units*np.log(2)    
   
    for s in range(1, S):
        
        # Calculate ps(Vs)
        Wh = np.dot(V, w) + np.outer(D,a) + np.dot(Z[document], wm)
        expWh = np.exp(beta[s]*Wh)
        vbias = np.dot(V,b)
        
        # Add log ps(Vs) to importance weights
        logW += np.log(1 + expWh).sum(axis = 1) + beta[s]*vbias

        # Use Transition Function T(Vs -> Vs+1)
        
        # Sample from Hiddens
        h = beta[s]*np.dot(V, w) + beta[s]*np.outer(D, a) + beta[s]*np.dot(Z[document], wm) 
        
        for i in range(0, len(h)):
            h[i] = sigmoid(h[i])   

        h_rand = np.random.rand(M, units)
        h_s = np.array(h_rand < h, dtype=int)              

        # Sample from Visuals
        v = beta[s]*(np.dot(h_s, w.T) + b)   

        v_pdf = [None]*M
        for r in range(0, M):
            v_pdf[r] = exp_softmax(v[r])

        V *= 0                     
        for r in range(M):          
            V[r] = np.random.multinomial(D[r], v_pdf[r], size=1)  
            
        # Calculate ps(Vs+1)
        Wh = np.dot(V, w) + np.outer(D, a) + np.dot(Z[document], wm)
        expWh = np.exp(beta[s]*Wh)
        vbias = np.dot(V,b)

        # Subtract log ps(Vs+1) from importance weights
        logW -= np.log(1 + expWh).sum(axis = 1) + beta[s]*vbias

    # Calculate pS(VS)
    Wh = np.dot(V, w) + np.outer(D,a) + np.dot(Z[document], wm)
    expWh = np.exp(Wh)
    vbias = np.dot(V,b)

    # Add pS(VS) to importance weights
    logW += np.log(1 + expWh).sum(axis = 1) + vbias

    # from scipy.special import logsumexp 
    logZ = LogMeanExp(logW) + D[0]*LogSumExp(0.1*b) + units*np.log(2)
        
    return logZ

if __name__ == '__main__':

    import time
    start = time.time()
    (w, b, a, wm) = train_cRSM(data = data_train, units=50, epochs=3000, lr=0.01, momentum=0.5, Z = Z_train, btsz=100)
    end = time.time()
    print(end - start)

    # Find Top 10 Words Loop
    for i in range(0, 50):

        p = w[:, i]

        t = p.tolist()
        q = [0,0,0,0,0,0,0,0,0,0]    

        for i in range(len(q)):
            q[i] = t.index(max(t))
            t[t.index(max(t))] = 0
            print(vocab[q[i]])

        print('\n')

    # Find the Partition Function for Heldout documents
    num_test_docs = 50
    num_hiddens = units = 50

    # Randomly Sample from test set
    v = data_test
    Zv = Z_test

    logZ = np.zeros(num_test_docs)

    for i in range(0, num_test_docs):
        logZ[i] = AIS_cRSM(data = v, document = i, M = 100, Z = Z_test, step = 1000, units = 50)
