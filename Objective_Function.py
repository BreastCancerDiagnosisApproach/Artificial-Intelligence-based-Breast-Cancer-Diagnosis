import numpy as np
from Global_Vars import Global_Vars
from relief import relief


def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
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
        weight = sol[7:]
        Selected_Feature = Feat[:, np.round(sol[0:7]).astype('int')]
        Weighted_Feature = Selected_Feature * weight
        chi_squared_stat = (((Weighted_Feature - Target) ** 2) / Weighted_Feature).sum().sum()
        scores = relief(Weighted_Feature, Target)
        Fitn[i] = (1 / (chi_squared_stat + scores))
    return Fitn
