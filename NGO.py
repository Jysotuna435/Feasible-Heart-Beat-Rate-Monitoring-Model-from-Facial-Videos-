import time

import numpy as np


def NGO(X, fit, Lowerbound, Upperbound, Max_iterations):
    Search_Agents, dimensions = X.shape[0], X.shape[1]
    X = []
    X_new = []
    fit_new = []
    average = np.zeros((1, Max_iterations))
    best_so_far = np.zeros((1, Max_iterations))
    NGO_curve = np.zeros((1, Max_iterations))
    ct = time.time()
    for t in np.arange(1, Max_iterations + 1).reshape(-1):
        ##  update: BEST proposed solution
        best, blocation = np.amin(fit)
        if t == 1:
            xbest = X[blocation, :]
            fbest = best
        else:
            if best < fbest:
                fbest = best
                xbest = X[blocation, :]
        ## UPDATE Northern goshawks based on PHASE1 and PHASE2
        for i in np.arange(1, Search_Agents + 1).reshape(-1):
            ## Phase 1: Exploration
            I = np.round(1 + np.random.rand())
            k = np.random.permutation(Search_Agents, 1)
            P = X[k, :]
            F_P = fit[k]
            if fit(i) > F_P:
                X_new[i, :] = X[i, :] + np.multiply(np.random.rand(1, dimensions), (P - np.multiply(I, X[i, :])))
            else:
                X_new[i, :] = X[i, :] + np.multiply(np.random.rand(1, dimensions), (X[i, :] - P))
            X_new[i, :] = np.amax(X_new[i, :], Lowerbound)
            X_new[i, :] = np.amin(X_new[i, :], Upperbound)
            # update position based on Eq (5)
            L = X_new[i, :]
            fit_new[i] = fit(L)
            if (fit_new[i] < fit[i]):
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]
            ## END PHASE 1
            ## PHASE 2 Exploitation
            R = 0.02 * (1 - t / Max_iterations)
            X_new[i, :] = X[i, :] + np.multiply((- R + 2 * R * np.random.rand(1, dimensions)), X[i, :])
            X_new[i, :] = np.amax(X_new[i, :], Lowerbound)
            X_new[i, :] = np.amin(X_new[i, :], Upperbound)
            # update position based on Eq (8)
            L = X_new[i, :]
            fit_new[i] = fit[L]
            if (fit_new[i] < fit(i)):
                X[i, :] = X_new[i, :]
                fit[i] = fit_new[i]
        best_so_far[t] = fbest
        average[t] = np.mean(fit)
        Score = fbest
        Best_pos = xbest
        NGO_curve[t] = Score

    ct = time.time() - ct
    return Score, Best_pos, NGO_curve, ct