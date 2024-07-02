import numpy as np

def Objfun_Cls(Soln):
    images = Glob_Vars.Images
    Targ = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(images .shape[0] * 0.75)
        train_data = images[learnper:, :]
        train_target = Targ[learnper:, :]
        test_data = images[:learnper, :]
        test_target = Targ[:learnper, :]
        Eval = Model_AMDRAN(train_data, train_target, test_data, test_target,sol.astype('int'))
        Fitn[i] = (1 / Eval[4]) + Eval[9]
    return Fitn


