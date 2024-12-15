import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math, sys, time
import time
import matplotlib.cm as cm
import time
from itertools import product


def propagate(g, b, infectedByb, TransmissionProb):
    addons = []
    for node in g[b]:
        if np.random.random() <= TransmissionProb:
            infectedByb.append(node)
            new_g = g.copy()
            new_g.remove_node(b)
            propagate(new_g, node, infectedByb, TransmissionProb)
        else:
            return infectedByb
    return infectedByb


def DeterministicRedBlue(Reds, Blues, Sets, Weights=None, LIMIT=None, output=False, testing=False, mipgap=None,
                         env=None):
    """
    Direct implementation of the deterministic red blue set covering problem
    """
    m = gp.Model(env=env)
    x = m.addVars(Sets.keys(), vtype=GRB.BINARY)
    y = m.addVars(Reds.keys(), vtype=GRB.CONTINUOUS, lb=0)
    if mipgap != None:
        m.Params.MIPGap = mipgap
    if LIMIT != None:
        m.Params.TimeLimit = LIMIT
    if not output:
        m.Params.OutputFlag = 0
    if Weights == None:
        m.setObjective(y.sum(), GRB.MINIMIZE)
    else:
        m.setObjective(gp.quicksum(Weights[r] * y[r] for r in Reds.keys()), GRB.MINIMIZE)

    # for r in Reds.keys():
    #   m.addConstrs((y[r] >= x[S] for S in Sets.keys() if r in Sets[S]))

    for S in Sets.keys():
        m.addConstrs((y[r] >= x[S] for r in set(Sets[S]).intersection(set(Reds.keys()))))
    for b in Blues.keys():
        m.addConstr(gp.quicksum(x[S] for S in \
                                {Set for Set in Sets.keys() if b in Sets[Set]}) \
                    >= 1)
    m.optimize()
    # gety
    SelectedReds = [r for r in y.keys() if y[r].x > 0.5]
    # getx
    SelectedSets = [s for s in x.keys() if x[s].x > 0.5]

    SolnEdgestoBlue = [(b, S) for S in SelectedSets for b in set(Sets[S]).intersection(set(Blues.keys()))]
    SolnEdgestoRed = [(r, S) for S in SelectedSets for r in set(Sets[S]).intersection(set(Reds.keys()))]
    SolnEdges = SolnEdgestoBlue + SolnEdgestoRed
    if testing:
        return SelectedReds, SelectedSets, SolnEdges, m.ObjVal, m.ObjBound, m.Runtime
    else:
        return SelectedReds, SelectedSets, SolnEdges


def FormAugmentedProblem(Reds, Blues, Sets):
    """
    Perform the set transformation in Carr et. al.
    """
    AugmentedSets = {}
    k = 0
    for S in Sets.keys():
        if len(set(Blues.keys()).intersection(Sets[S])) > 1:
            RedsInS = set(Reds.keys()).intersection(Sets[S])
            for b in set(Blues.keys()).intersection(Sets[S]):
                AugmentedSets['Set' + str(k)] = list(RedsInS) + [b]
                k = k + 1
        else:
            AugmentedSets['Set' + str(k)] = Sets[S]
        k = k + 1
    return AugmentedSets


def DeterministicRedBlueAugmented(Reds, Blues, AugmentedSets, Relax=True, Weights=None, output=False, q=None, env=None):
    ### Solving the augmented model
    m = gp.Model(env=env)

    if not output:
        m.Params.OutputFlag = 0

    if Relax:
        x = m.addVars(AugmentedSets.keys(), vtype=GRB.CONTINUOUS, lb=0, ub=1)
    else:
        x = m.addVars(AugmentedSets.keys(), vtype=GRB.BINARY)
    y = m.addVars(Reds.keys(), vtype=GRB.CONTINUOUS, lb=0)

    if Weights == None:
        m.setObjective(y.sum(), GRB.MINIMIZE)
    else:
        m.setObjective(gp.quicksum(Weights[r] * y[r] for r in Reds.keys()), GRB.MINIMIZE)

    for b in Blues.keys():
        # reds that are in sets that contain b
        RedsinSContainsb = [set(AugmentedSets[S]).intersection(set(Reds.keys())) for S in AugmentedSets.keys() if
                            b in AugmentedSets[S]]
        mu_b = min([len(Sb) for Sb in RedsinSContainsb])
        m.addConstr(y.sum() >= mu_b)

    # this constraint sucks
    for (r, b) in product(Reds.keys(), Blues.keys()):
      #sets containing r and b
      SetsContainingRandB = {S for S in AugmentedSets.keys() if (r in AugmentedSets[S] and b in AugmentedSets[S])}
      if SetsContainingRandB != set():
        m.addConstr(y[r] >= gp.quicksum(x[S] for S in SetsContainingRandB))

    for S in AugmentedSets.keys():
        m.addConstrs((y[r] >= x[S] for r in set(AugmentedSets[S]).intersection(set(Reds.keys()))))

    for b in Blues.keys():
        m.addConstr(gp.quicksum(x[S] for S in \
                                {Set for Set in AugmentedSets.keys() if b in AugmentedSets[Set]}) >= 1)
    m.optimize()

    # proceed = False
    # while not proceed:
    #     test = True
    #     for (r, b) in product(Reds.keys(), Blues.keys()):
    #       #sets containing r and b
    #       SetsContainingRandB = {S for S in AugmentedSets.keys() if (r in AugmentedSets[S] and b in AugmentedSets[S])}
    #       if SetsContainingRandB != set():
    #         if y[r].x < sum(x[S].x for S in SetsContainingRandB)-0.0001:
    #             test = False
    #
    #             #print("adding constraint violation")
    #             m.addConstr(y[r] >= gp.quicksum(x[S] for S in SetsContainingRandB))
    #     if test is False:
    #         #print("re-optimizing")
    #         m.optimize()
    #     if test is True:
    #         proceed = True
    # gety
    SelectedReds = [r for r in y.keys() if y[r].x > 0.5]
    # getx
    SelectedAugSets = [s for s in x.keys() if x[s].x > 0.5]

    SolnEdgestoBlue = [(b, S) for S in SelectedAugSets for b in set(AugmentedSets[S]).intersection(set(Blues.keys()))]
    SolnEdgestoRed = [(r, S) for S in SelectedAugSets for r in set(AugmentedSets[S]).intersection(set(Reds.keys()))]
    SolnEdges = SolnEdgestoBlue + SolnEdgestoRed
    vals_y = {r: y[r].x for r in y.keys()}
    vals_x = {S: x[S].x for S in x.keys()}
    if q != None:
        q.put([SelectedReds, SelectedAugSets, SolnEdges, vals_y, vals_x])
    return SelectedReds, SelectedAugSets, SolnEdges, vals_y, vals_x


def CarrApproximationAlgorithm(Reds, Blues, AugmentedSets, vals_y):
    GoodBlues = set()
    BadBlues = set()
    n = len(AugmentedSets.keys())
    rho = len(Reds.keys())
    for b in Blues.keys():
        SetsContainingb = [S for S in AugmentedSets.keys() if b in AugmentedSets[S]]
        if len(SetsContainingb) > n ** (0.5) + 0.0001:
            BadBlues.add(b)
        else:
            GoodBlues.add(b)
    yAug = {}
    for r in vals_y.keys():
        yAug[r] = math.floor(vals_y[r] * (n ** 0.5))
        if yAug[r] > 0.5:
            yAug[r] = 1
    SelectedRedsPhase1 = {r for r in yAug.keys() if yAug[r] > 0.5}
    SelectedAugSetsPhase1 = set()
    for S in AugmentedSets.keys():
        # if the reds in a set are a subset of the phase1 reds
        RedsinS = set(AugmentedSets[S]).intersection(set(Reds.keys()))
        if RedsinS.issubset(SelectedRedsPhase1):
            SelectedAugSetsPhase1.add(S)

    SelectedRedsPhase2 = set()
    SelectedAugSetsPhase2 = set()
    for b in BadBlues:
        best = None
        best_size = rho
        for S in [Set for Set in AugmentedSets.keys() if (b in AugmentedSets[S])]:
            RedsinS = set(AugmentedSets[S]).intersection(set(Reds.keys()))
            if len(RedsinS) < best_size:
                best = S
                best_size = len(RedsinS)

        if best != None:
            SelectedAugSetsPhase2.add(S)

    for S in SelectedAugSetsPhase2:
        RedsinS = set(AugmentedSets[S]).intersection(set(Reds.keys()))
        SelectedRedsPhase2 = SelectedRedsPhase2.union(RedsinS)

    SelectedAugSets = [s for s in SelectedAugSetsPhase2.union(SelectedAugSetsPhase1)]

    ###Remove redundant sets
    # for b in Blues.keys():
    #   for S1 in [S for S in SelectedAugSets if b in AugmentedSets[S]]:
    #     RedsinS1 = set(AugmentedSets[S1]).intersection(set(Reds.keys()))
    #     RedsinS2 = set()
    #     for S2 in [S for S in SelectedAugSets if (S != S1 and b in AugmentedSets[S])]:
    #       RedsinS2 = RedsinS2.union(set(AugmentedSets[S2]).intersection(set(Reds.keys())))

    #     # print("b ", b)
    #     # print(RedsinS1)
    #     # print(RedsinS2)
    #     if RedsinS1.issubset(RedsinS2):
    #         print("S1 Can be removed")

    # gety
    SelectedReds = [r for r in SelectedRedsPhase2.union(SelectedRedsPhase1)]
    # getx

    SolnEdgestoBlue = [(b, S) for S in SelectedAugSets for b in set(AugmentedSets[S]).intersection(set(Blues.keys()))]
    SolnEdgestoRed = [(r, S) for S in SelectedAugSets for r in set(AugmentedSets[S]).intersection(set(Reds.keys()))]
    SolnEdges = SolnEdgestoBlue + SolnEdgestoRed

    print("Number of reds covered: ", len(SelectedReds))
    return SelectedReds, SelectedAugSets, SolnEdges


def deg(r, Sets):
    '''
    Number of sets containing r
    '''
    counter = 0
    for S in Sets.keys():
        if r in Sets[S]:
            counter += 1
    return counter


def Delta(Reds, Sets):
    '''
    Maximum degree
    '''
    return max([deg(r, Sets) for r in Reds.keys()])


def ElementsinFamily(Sets, Elements):
    '''
    Elements in the collection
    '''
    ElementsinFamily = set()
    for S in Sets.keys():
        ElementsinFamily = ElementsinFamily.union(set(Sets[S]).intersection(set(Elements)))
    return ElementsinFamily

def get_average_number_of_reds(red_scenarios, sets):
    count = 0
    for scenario_idx in red_scenarios.keys():
        for set_idx in sets.keys():
            count += NumRedsinS(sets[set_idx], red_scenarios[scenario_idx], Weights=None)
    print("average # of reds", count / (len(sets) * len(red_scenarios.keys())))
    return count / (len(sets) * len(red_scenarios.keys()))

def NumRedsinS(Set, Reds, Weights=None):
    try:  # if reds is a dictionary where the keys are the elements
        RedsinS = set(Set).intersection(set(Reds.keys()))
        if Weights == None:
            return len(RedsinS)
        else:
            return sum(Weights[r] for r in RedsinS)
    except:  # else its just an iterable/set of reds
        RedsinS = set(Set).intersection(set(Reds))
        if Weights == None:
            return len(RedsinS)
        else:
            return sum(Weights[r] for r in RedsinS)


def Phi(Reds, Sets, S=None):
    if S == None:  # calculate phi for a collection
        try:  # if Reds  is a  dictionary
            return {S: set(Sets[S]) - set(Reds.keys()) for S in Sets.keys()}
        except:  # otherwise reds is an iterable/list
            return {S: set(Sets[S]) - set(Reds) for S in Sets.keys()}
    else:  # if S is supplied then Sets is a dictionary
        try:  # if Reds  is a  dictionary
            return set(Sets[S]) - set(Reds.keys())
        except:  # otherwise reds is an iterable/list
            return set(Sets[S]) - set(Reds)


def findMin(Sets, R, w):
    minCost = 99999.0
    minElement = -1
    for S in Sets.keys():
        try:
            cost = w[S] / (len(set(Sets[S]).intersection(R)))
            if cost < minCost:
                minCost = cost
                minElement = S
        except:
            # Division by zero, ignore
            pass
    return minElement, set(Sets[minElement]), w[minElement]


# while len(R) != 0:
#     S_i, cost = findMin(S, R)
#     C.append(S_i)
#     R = R.difference(S_i)
#     costs.append(cost)

def GreedyWeightedSetCover(Sets, weights, Elements, printt=False):
    '''
    Greedy weighted set cover algorithm
    Add sets to the cover that have the smallest weight/number of elements covered
    '''
    costs = []
    Uncovered = Elements.copy()
    Cover = []

    # check that all elements are covered by Sets:
    # SetsUnion = set()
    # for S in Sets.keys():
    #   SetsUnion = SetsUnion.union(set(Sets[S]))
    # if not set(Elements).issubset(SetsUnion):
    #   print("Not all elements covered")
    #   return None
    R = set(Elements)
    while len(R) != 0:
        Set_idx, S_i, cost = findMin(Sets, R, weights)
        Cover.append(Set_idx)
        R = R.difference(S_i)
        # costs.append(cost)

    return Cover


### Greedy Red Blue

def GreedyRB(Reds, Blues, Sets, Weights=None):
    '''
    Augment collection of sets to just the blue elements
    set the costs of each set equal to the number of reds in that set
    do a greedy weighted set cover on this new instance
    '''
    T = Phi(Reds, Sets)
    GreedyWeights = {S: NumRedsinS(Sets[S], Reds, Weights) for S in Sets.keys()}
    coverT = GreedyWeightedSetCover(T, GreedyWeights, set(Blues.keys()))
    return {S: Sets[S] for S in coverT}


def LowDeg(X, Reds, Blues, Sets, NumRedsinSLookup, deg_reds, Weights=None):
    # discard sets with more than X red elements
    S_X = {S: Sets[S] for S in Sets.keys() if NumRedsinSLookup[S] <= X}
    # check feasibility
    BluesinS_X = ElementsinFamily(S_X, Blues.keys())
    if BluesinS_X != set(Blues.keys()):
        # print("Not Feasible")
        return set(Sets.keys())

    Y = math.sqrt(len(Sets) / math.log(len(Blues)))
    # high degree reds
    RedsH = {r for r in Reds.keys() if deg_reds[r] > Y}
    RedsL = set(Reds.keys()) - RedsH
    S_XY = Phi(RedsH, S_X)
    GreedyCover = GreedyRB(RedsL, Blues, S_XY)
    return set(GreedyCover.keys())


def LowDeg2(Reds, Blues, Sets, Weights=None, q=None):
    BestCover = Sets
    MinReds = NumRedsinS(Reds, Reds, Weights)
    degrees = {}
    if Weights == None:
        X_pts = []
        S_X_Last = set()
        NumRedsinSLookup = {S: NumRedsinS(Sets[S], Reds, Weights) for S in Sets.keys()}

        for X in range(1, len(Reds) + 1):
            S_X = {S: Sets[S] for S in Sets.keys() if NumRedsinSLookup[S] <= X}
            if S_X_Last != S_X:
                X_pts.append(X)
                S_X_Last = S_X
        deg_reds = {r: 0 for r in Reds.keys()}
        X_Last = 0
        for X in X_pts:
            NewSets = {S: Sets[S] for S in Sets.keys() \
                       if (NumRedsinSLookup[S] <= X and NumRedsinSLookup[S] > X_Last)}
            NewReds = ElementsinFamily(NewSets, Reds.keys())
            for r in NewReds:
                deg_reds[r] += 1
            Candidates = LowDeg(X, Reds, Blues, Sets, NumRedsinSLookup, deg_reds, Weights)
            CandidateSets = {Candidate: Sets[Candidate] for Candidate in Candidates}
            RedsinCandidates = ElementsinFamily(CandidateSets, Reds)
            NumReds = NumRedsinS(RedsinCandidates, Reds, Weights)
            if NumReds < MinReds:
                BestCover = CandidateSets
                MinReds = NumReds
            X_Last = X
    else:
        X_pts = []
        S_X_Last = set()
        NumRedsinSLookup = {S: NumRedsinS(Sets[S], Reds, Weights) for S in Sets.keys()}
        # search over all possible X
        sumwgts = np.sum(list(Weights.values()))
        min_wgt = np.amin(list(Weights.values()))
        step_size = np.amin(np.diff(np.unique(list(Weights.values()))))
        T = 2 * math.ceil((sumwgts - min_wgt) / step_size)
        for X in np.linspace(min_wgt, sumwgts, num=T):
            # for X in np.cumsum(np.sort(list(Weights.values()))):
            S_X = {S: Sets[S] for S in Sets.keys() if NumRedsinSLookup[S] <= X}
            if S_X_Last != S_X:
                X_pts.append(X)
                S_X_Last = S_X

        deg_reds = {r: 0 for r in Reds.keys()}
        X_Last = 0
        for X in X_pts:
            NewSets = {S: Sets[S] for S in Sets.keys() \
                       if (NumRedsinSLookup[S] <= X and NumRedsinSLookup[S] > X_Last)}
            NewReds = ElementsinFamily(NewSets, Reds.keys())
            for r in NewReds:
                deg_reds[r] += 1
            Candidates = LowDeg(X, Reds, Blues, Sets, NumRedsinSLookup, deg_reds, Weights)
            CandidateSets = {Candidate: Sets[Candidate] for Candidate in Candidates}
            RedsinCandidates = ElementsinFamily(CandidateSets, Reds)
            NumReds = NumRedsinS(RedsinCandidates, Reds, Weights)
            if NumReds < MinReds:
                BestCover = CandidateSets
                MinReds = NumReds
            X_Last = X
    if q != None:
        q.put([BestCover, MinReds])
    return BestCover, MinReds
