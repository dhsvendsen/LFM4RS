# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:58:56 2020
@author: Daniel & Jordi
"""

#import matplotlib.pyplot as plt
#import autograd

#### notes: 
#### - there is only one decay pr. output, thus decays[lf] below is wrong
#### 

from autograd import grad, elementwise_grad
import autograd.numpy as np
from numpy.linalg import LinAlgError
#from scipy.linalg.misc import LinAlgError
#import autograd.scipy.stats.norm as norm
from autograd.scipy.stats import norm
from autograd.scipy.special import erf
from autograd.scipy.special import expit as sigmoid
from autograd.numpy.linalg import solve, cholesky
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten_func


class LFM:
    def __init__(self, p, init_logparams=None, nlf=1, jitter=1e-10):
        """
        p: number of inputs and outputs
        init_logparams: optional, initial log parameters
        nlf: number of latent forces
        jitter: when computing Cholesky, for numerical stability
        """
        self.p = p
        self.nlf = nlf
        self.jitter = jitter
        self.clamplengthscales = None
        if init_logparams:
            self.logparams = init_logparams
        else:
            self.logparams = {'lengthscales': [np.log(1)] * nlf,  # for lf in range(nlf)],
                              'decays': [[np.log(1)] * p] * nlf,  #  [[np.log(1) for i in range(p)] for lf in range(nlf)],
                              'couplings': [[np.log(1)] * p] * nlf,  # [[np.log(1) for i in range(p)] for lf in range(nlf)],
                              'sigmas': [np.log(1)] * p}  # [np.log(1) for i in range(p)]}
        # Boolean dict to fix parameters for optimization
        self.fixparams = {k: False for k in self.logparams.keys()}


    def get_params(self):
        return {k: np.exp(v) for k, v in self.logparams.items()}


    def set_params(self, params):
        for k, v in params.items():
            self.logparams[k] = np.log(v)


    def set_logparams(self, params):
        for k, v in params.items():
            self.logparams[k] = v


    def hqp(self, T1, T2, Dq, Dp, l):
        """
        TERM1: [ exp(\nu_q^2) / (Dp+Dq) ] * exp(-D_q * T1)
        TERM2: exp(Dq*T2)*[ erf((T1-T2)/l - \nu_q) + erf(T1/l + \nu_q) ]
        TERM3: exp(-Dp*T2)*[ erf(T1/l - \nu_q) + erf(\nu_q) ]
        K = TERM1*(TERM2 - TERM3)
        """
        n1, n2 = len(T1), len(T2)
        T1cols = np.hstack([T1.reshape(n1,1) for i in range(n2)])
        T2rows = np.vstack([T2.reshape(1,n2) for i in range(n1)])

        nuq = l * Dq * 0.5
        # TERM1 = np.exp(nuq**2)/(Dq+Dp) * np.exp(-Dq * T1cols)
        # TERM2 = np.exp(Dq * T2rows) * (erf((T1cols-T2rows)/l - nuq) + erf(T2rows/l + nuq))
        # TERM3 = np.exp(-Dp * T2rows) * (erf(T1cols/l - nuq) + erf(nuq))
        TERM1 = np.exp(nuq**2)/(Dq+Dp)
        TERM2 = np.exp(-Dq * T1cols + Dq * T2rows) * (erf((T1cols-T2rows)/l - nuq) + erf(T2rows/l + nuq))
        TERM3 = np.exp(-Dq * T1cols - Dp * T2rows) * (erf(T1cols/l - nuq) + erf(nuq))
        return TERM1 * (TERM2 - TERM3)


    def simKpq(self, T1, T2, Dp, Dq, l):
        return 0.5 * np.sqrt(np.pi) * l * (self.hqp(T1,T2,Dp,Dq,l) + self.hqp(T2,T1,Dq,Dp,l).T)

    def multiK(self, Ttr, decays, lengthscale, couplings):
        ndict= {p:len(Ttr[p]) for p in range(len(Ttr))}
        blockdiag = {i:[] for i in range(self.p)}
        blockoffdiag = {i:[] for i in range(self.p)}
        diagrows, offdiagrows = [], []
        lf = 0
        for i in range(self.p):
            for j in range(self.p):
                blockdiag[i].append(np.zeros([ndict[i],ndict[j]]))
                blockoffdiag[i].append(np.zeros([ndict[i],ndict[j]]))               
                if i == j:
                    blockdiag[i][i] = self.simKpq(Ttr[i], Ttr[i], decays[i], decays[i], lengthscale)*couplings[i]*couplings[j]
                if j > i:
                    blockoffdiag[i][j] = self.simKpq(Ttr[i], Ttr[j], decays[i], decays[j], lengthscale)*couplings[i]*couplings[j]
            diagrows.append( np.hstack( blockdiag[i] ) )
            offdiagrows.append( np.hstack( blockoffdiag[i] ) )
    
        diagK = np.vstack(diagrows)
        offdiagK = np.vstack(offdiagrows)
        return diagK + offdiagK + offdiagK.T
    
    def multiK_jordi(self, Ttr, decays, lengthscale, couplings):
        ndict = {p:len(Ttr[p]) for p in range(len(Ttr))}
        blockdiag = {i:[] for i in range(self.p)}
        blockoffdiag = {i:[] for i in range(self.p)}
        diagrows, offdiagrows = [], []
        # lf = 0
        for i in range(self.p):
            for j in range(self.p):
                blockdiag[i].append(np.zeros([ndict[i], ndict[j]]))
                blockoffdiag[i].append(np.zeros([ndict[i], ndict[j]]))
                # if i == j:
                #     blockdiag[i][i] = self.simKpq(Ttr[i], Ttr[i], decays[i], decays[i], lengthscale)*couplings[i]*couplings[j]
                # elif j > i:
                #     blockoffdiag[i][j] = self.simKpq(Ttr[i], Ttr[j], decays[i], decays[j], lengthscale)*couplings[i]*couplings[j]
                if i >= j:
                    blockdiag[i][j] = self.simKpq(Ttr[i], Ttr[j], decays[i], decays[j], lengthscale) * \
                                        couplings[i] * couplings[j]
            diagrows.append(np.hstack(blockdiag[i]))
            offdiagrows.append(np.hstack(blockoffdiag[i]))

        diagK = np.vstack(diagrows)
        offdiagK = np.vstack(offdiagrows)
        return diagK + offdiagK + offdiagK.T


    def sigmablocks(self, Ttr, sigmas):
        ndict= {p:len(Ttr[p]) for p in range(len(Ttr))}
        blockdiag = {i:[] for i in range(self.p)}
        blockoffdiag = {i:[] for i in range(self.p)}
        diagrows, offdiagrows = [], []
        # lf = 0
        for i in range(self.p):
            for j in range(self.p):
                blockdiag[i].append(np.zeros([ndict[i], ndict[j]]))
                blockoffdiag[i].append(np.zeros([ndict[i], ndict[j]]))
                if i == j:
                    blockdiag[i][i] = np.eye(ndict[i]) * sigmas[i]
            diagrows.append(np.hstack(blockdiag[i]))
            offdiagrows.append(np.hstack( blockoffdiag[i]))

        diagK = np.vstack(diagrows)
        offdiagK = np.vstack(offdiagrows)
        return diagK + offdiagK + offdiagK.T


    def predmultiK(self, Tnew, Ttr, decays, lengthscale, couplings, whichp, latent=False):
        # ndict = {p:len(Ttr[p]) for p in range(len(Ttr))}
        # lf = 0
        blocks = []
        if latent:
            for i in range(len(Ttr)):
                blocks.append(couplings[i] * self.cross(Ttr[i], Tnew, decays[i], lengthscale).T)
        else:
            for i in range(len(Ttr)):
                blocks.append(couplings[whichp] * couplings[i] * \
                    self.simKpq(Tnew, Ttr[i], decays[whichp], decays[i], lengthscale))
        return np.hstack(blocks)


    def get_logparam(self, param, logparams, fixedlogparams):
        return np.exp(logparams[param]) if param in logparams.keys() else np.exp(fixedlogparams[param])


    def neg_log_marg_like(self, Ttr, Ytr, fixedlogparams, logparams):
        """
        Compute negative log marginal likelihood for hyperparameter optimization
        """
        ns = [len(t) for t in Ttr]
        n = sum(ns)
        lengthscales = self.get_logparam('lengthscales', logparams, fixedlogparams)
        if self.clamplengthscales:
            lengthscales = np.minimum(np.array([self.clamplengthscales]), lengthscales)
        decays = self.get_logparam('decays', logparams, fixedlogparams)#[0]  # np.exp(logparams['decays'])
        sigmas = self.get_logparam('sigmas', logparams, fixedlogparams)  # np.exp(logparams['sigmas']) # np.array([0.01,0.2])
        couplings = self.get_logparam('couplings', logparams, fixedlogparams)  # np.exp(logparams['couplings'])
        #couplings = np.minimum([100*np.var(y) for y in Ytr], couplings) # avoid overfitting
        #print(+ np.diag( np.repeat(sigmas,ns) ))

        # Lets forget about multiple latent functions for a while
        #K = self.multiK(Ttr, decays[0], lengthscales[0], couplings[0])
        K = np.zeros([n,n])
        for lf in range(self.nlf):
            K += self.multiK(Ttr, decays, lengthscales[lf], couplings[lf])
        # Multiple latent functions
        #K = np.sum([self.multiK(Ttr, decays[lf], lengthscales[lf], couplings[lf]) for lf in range(self.nlf)], 0)
        # Jitter and sigma blocks
        K += self.jitter*np.eye(n) + self.sigmablocks(Ttr, sigmas)

        # T = np.vstack(Ttr)
        Y = np.vstack(Ytr)
        self.Y = Y

        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L,Y))
            logmarglike = \
                 - 0.5*np.dot(Y.T, alpha)[0,0]  \
                 - np.sum(np.log(np.diag(L)))   \
                 - 0.5*n*np.log(2*np.pi)
        except LinAlgError:
            print('Warning, Cholesky failed!')
            alpha = solve(K,Y)
            logmarglike = \
                 - 0.5*np.dot(Y.T, alpha)[0,0]  \
                 - np.sum(np.log(np.linalg.det(K)))   \
                 - 0.5*n*np.log(2*np.pi)

        # MSE
        #logmarglike = -np.mean((np.dot(K,alpha) - Y)**2)

        return -logmarglike


    def predict(self, Tnew, Ttr, Ytr, whichp=0, logparams=None, latent=False, retPV=False):#(self, Tnew, Ttr, Ytr, whichp=0, logparams=None, latent=False):
        if not logparams:
            logparams = self.logparams
            print([(key,np.exp(val)) for key,val in logparams.items()])
        ns = [len(t) for t in Ttr]
        n = sum(ns)
        lengthscales = np.exp(logparams['lengthscales'])
        if self.clamplengthscales:
            lengthscales = np.minimum( np.array([self.clamplengthscales]), lengthscales)
        decays = np.exp(logparams['decays'])
        sigmas = np.exp(logparams['sigmas']) # np.array([0.01,0.2])
        couplings = np.exp(logparams['couplings'])
        #couplings = np.minimum([100*np.var(y) for y in Ytr], couplings) # avoid overfitting

        # T = np.vstack(Ttr)
        Y = np.vstack(Ytr)
        K = np.sum([self.multiK(Ttr, decays, lengthscales[lf], couplings[lf]) for lf in range(self.nlf)], 0)
        K += self.jitter*np.eye(n) + self.sigmablocks(Ttr, sigmas)
        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L,Y))
        except LinAlgError:
            print('Warning, Cholesky failed!')
            alpha = solve(K,Y)

        # Returned values
        yp, vp, k, y_lf, v_lf, k_lf = None, None, None, None, None, None

        # Kstar
        k = np.sum([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp)
                for lf in range(self.nlf)], 0)
        # Prediction
        yp = np.dot(k, alpha)
        # Kstar,star if predictive variance is asked for
        if retPV:
            v = solve(L, k.T)
            # Daniel is right, for the outputs we have to compute the simKpq kernel
            kss = 0
            for lf in range(self.nlf):
                kss += np.diag(
                    self.simKpq(Tnew, Tnew, decays[whichp], decays[whichp], lengthscales[lf])
                ) * couplings[lf][whichp] * couplings[lf][whichp]
            # vp = (kss - np.sum(v*v,0))[:,None]
            # Adding the extra sigmas[whichp] here (Daniel knows why)
            vp = (sigmas[whichp] + kss - np.sum(v*v,0))[:,None]

        if latent:
            # Previously we sum all LFs, which make little sense
            # k_lf = np.sum([self.predmultiK(Tnew, Ttr, decays[lf], lengthscales[lf], couplings[lf], whichp=whichp, latent=True)
            #             for lf in range(self.nlf)], 0)
            # Right way: stack cross-cov. kernels, one per LF
            # I think I fixed and error in Daniel's code: he passed 'decays' as paramters,
            # but we should pass 'decays[lf]'.
            k_lf = np.vstack([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp, latent=True)
                        for lf in range(self.nlf)])
            y_lf = np.dot(k_lf, alpha)
            if retPV:
                v = solve(L, k_lf.T)
                v_lf = (1 - np.sum(v*v, 0))[:,None]

        return (yp, vp, k, y_lf, v_lf, k_lf)
            
#        k = np.sum([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp) for lf in range(self.nlf)], 0)
#
#        if latent:
#            #k_lf = np.sum([self.predmultiK(Tnew, Ttr, decays[lf], lengthscales[lf], couplings[lf], whichp=whichp, latent=True) for lf in range(self.nlf)], 0)
#            #return np.dot(k, alpha), np.dot(k_lf, alpha), k, k_lf
#            k_lf = np.vstack([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp, latent=True) for lf in range(self.nlf)])
#            return np.dot(k, alpha),np.dot(k_lf, alpha), k, k_lf
#
#        else:
#            return np.dot(k, alpha), k

    def predict_var(self, Tnew, Ttr, Ytr, whichp=0, logparams=None, latent=False):
        if not logparams:
            logparams = self.logparams
            print([(key,np.exp(val)) for key,val in logparams.items()])
        ns = [len(t) for t in Ttr]
        n = sum(ns)
        lengthscales = np.exp(logparams['lengthscales'])
        if self.clamplengthscales:
            lengthscales = np.minimum( np.array([self.clamplengthscales]), lengthscales)
        decays = np.exp(logparams['decays'])
        sigmas = np.exp(logparams['sigmas']) # np.array([0.01,0.2])
        couplings = np.exp(logparams['couplings'])
        #couplings = np.minimum([100*np.var(y) for y in Ytr], couplings) # avoid overfitting

        # T = np.vstack(Ttr)
        Y = np.vstack(Ytr)
        K = np.sum([self.multiK(Ttr, decays, lengthscales[lf], couplings[lf]) for lf in range(self.nlf)], 0)
        K += self.jitter*np.eye(n) + self.sigmablocks(Ttr, sigmas)
        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L,Y))
        except LinAlgError:
            print('Warning, Cholesky failed!')
            alpha = solve(K,Y)

        k = np.sum([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp) for lf in range(self.nlf)], 0)

            
        if latent:
            k_lf = np.vstack([self.predmultiK(Tnew, Ttr, decays, lengthscales[lf], couplings[lf], whichp=whichp, latent=True) for lf in range(self.nlf)])             
            predmean_lf = np.dot(k_lf, alpha)
            # The first term of predictive variance, cheating here as we only need diagonal and simple rbf diag = 1
            Kpredpred = np.eye(self.nlf*len(Tnew))
            v = solve(L,k_lf.T)
            predvar_lf = np.diag(Kpredpred) - np.diag( np.dot(v.T, v) )
            return predmean_lf, predvar_lf

        else:
            predmean = np.dot(k, alpha)
            # The first term of predictive variance, only need diagonal
            Kpredpred = np.zeros([len(Tnew),len(Tnew)])
            for lf in range(self.nlf):
                Kpredpred += self.simKpq(Tnew, Tnew, decays[whichp], decays[whichp], lengthscales[lf])*couplings[lf][whichp]*couplings[lf][whichp]
            v = solve(L,k.T)
            predvar = sigmas[whichp] + np.diag(Kpredpred) - np.diag( np.dot(v.T, v) )
            return predmean, predvar


    def fit(self, Ttr, Ytr, n_iter=1000, n_update=100, lr=0.01, jitter=1e-10):
        # print([100*np.var(y) for y in Ytr])
        self.jitter = jitter

        # Copy non-fixed self.logparams to logparams and fixed to fixedlogparams
        # logparams = {par:val in self.logparams.items() if not self.fixparams[par]}
        # fixedlogparams = {par:val in self.logparams.items() if self.fixparams[par]}
        logparams = self.logparams.copy()
        fixedlogparams = {}
        for param, fixed in self.fixparams.items():
            if fixed:
                parval = logparams.pop(param)
                fixedlogparams[param] = parval

        obj = lambda logparams, itr: self.neg_log_marg_like(Ttr, Ytr, fixedlogparams, logparams)  # second param is the iteration
        flat_obj, unflatten, self.flat_logparams = flatten_func(obj, logparams)  # use logparams, no self.logparams

        # return obj, flat_obj, unflatten, logparams, fixedlogparams

        for i in range( np.int(n_iter/n_update) ):
            #print(grad(flat_obj)(self.flat_logparams,0))
            self.flat_logparams = adam(grad(flat_obj), self.flat_logparams, step_size=lr, num_iters=n_update)

            nll = flat_obj(self.flat_logparams, 0)
            if np.isnan(nll):
                print('Error: nll is NaN, stopping optimization')
                break

            self.logparams = unflatten(self.flat_logparams)
            self.params = unflatten(np.exp(self.flat_logparams))

            print('Iteration ' + str((i+1)*n_update) + '. Current params:')
            for key in self.params.keys():
                if key == 'lengthscales' and self.clamplengthscales:
                    print('  ', key, ':', np.array([self.clamplengthscales]) )
                    #print(np.array([self.clamplengthscales]), 'vs', self.params[key])
                elif key == 'couplings':
                    print('  ', key, ':', self.params[key])
                    #print(key,':', np.minimum([100*np.var(y) for y in Ytr], self.params[key]))
                    #print([100*np.var(y) for y in Ytr], 'vs', self.params[key])
                else:
                    print('  ', key, ':', self.params[key])
            #print('  nll: ', flat_obj(self.flat_logparams, 0),'\n')
            print('  nll: ', nll, '\n')

        # Re-add fixed parameters
        for k, v in fixedlogparams.items():
            self.logparams[k] = v


    def cross(self, T1, T2, Dq, l):
        """
        (sqrt(pi)*l/2)exp(\nu_q^2)exp(Dq*(T1-T2))*[ erf((T2-T1)/l - \nu_q) + erf(T2/l + \nu_q) ]
        """
        n1, n2 = len(T1), len(T2)
        T1cols = np.hstack([T1.reshape(n1,1) for i in range(n2)])
        T2rows = np.vstack([T2.reshape(1,n2) for i in range(n1)])
        nuq = l * Dq * 0.5
        return (np.sqrt(np.pi)*l/2) * np.exp(nuq*nuq - Dq * (T1cols - T2rows)) * \
               (erf((T1cols-T2rows)/l - nuq) + erf(T2rows/l + nuq))
