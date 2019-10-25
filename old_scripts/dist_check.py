#!/usr/bin/env python
#title           :distribution_checkX.py
#description     :Checks a sample against 80 distributions by applying the Kolmogorov-Smirnov test.
#author          :Andre Dietrich
#email           :dietrich@ivs.cs.uni-magdeburg.de
#date            :07.10.2014
#version         :0.1
#usage           :python distribution_check.py -f filename -v
#python_version  :2.* and 3.*
#########################################################################################
from __future__ import print_function

import scipy.stats
import warnings
import random
import math
import numpy

# just for surpressing warnings
warnings.simplefilter('ignore')

from joblib import Parallel, delayed

########################################################################################

# list of all available distributions
cdfs = {
    "alpha": {"p":[], "D": [], "KL": []},           #Alpha
    "anglit": {"p":[], "D": [], "KL": []},          #Anglit
    "arcsine": {"p":[], "D": [], "KL": []},         #Arcsine
    "beta": {"p":[], "D": [], "KL": []},            #Beta
    "betaprime": {"p":[], "D": [], "KL": []},       #Beta Prime
    "bradford": {"p":[], "D": [], "KL": []},        #Bradford
    "burr": {"p":[], "D": [], "KL": []},            #Burr
    "cauchy": {"p":[], "D": [], "KL": []},          #Cauchy
    "chi": {"p":[], "D": [], "KL": []},             #Chi
    "chi2": {"p":[], "D": [], "KL": []},            #Chi-squared
    "cosine": {"p":[], "D": [], "KL": []},          #Cosine
    "dgamma": {"p":[], "D": [], "KL": []},          #Double Gamma
    "dweibull": {"p":[], "D": [], "KL": []},        #Double Weibull
    "erlang": {"p":[], "D": [], "KL": []},          #Erlang
    "expon": {"p":[], "D": [], "KL": []},           #Exponential
    "exponweib": {"p":[], "D": [], "KL": []},       #Exponentiated Weibull
    "exponpow": {"p":[], "D": [], "KL": []},        #Exponential Power
    "f": {"p":[], "D": [], "KL": []},               #F (Snecdor F)
    "fatiguelife": {"p":[], "D": [], "KL": []},     #Fatigue Life (Birnbaum-Sanders)
    "fisk": {"p":[], "D": [], "KL": []},            #Fisk
    "foldcauchy": {"p":[], "D": [], "KL": []},      #Folded Cauchy
    "foldnorm": {"p":[], "D": [], "KL": []},        #Folded Normal
    "frechet_r": {"p":[], "D": [], "KL": []},       #Frechet Right Sided, Extreme Value Type II
    "frechet_l": {"p":[], "D": [], "KL": []},       #Frechet Left Sided, Weibull_max
    "gamma": {"p":[], "D": [], "KL": []},           #Gamma
    "gausshyper": {"p":[], "D": [], "KL": []},      #Gauss Hypergeometric
    "genexpon": {"p":[], "D": [], "KL": []},        #Generalized Exponential
    "genextreme": {"p":[], "D": [], "KL": []},      #Generalized Extreme Value
    "gengamma": {"p":[], "D": [], "KL": []},        #Generalized gamma
    "genhalflogistic": {"p":[], "D": [], "KL": []}, #Generalized Half Logistic
    "genlogistic": {"p":[], "D": [], "KL": []},     #Generalized Logistic
    "genpareto": {"p":[], "D": [], "KL": []},       #Generalized Pareto
    "gilbrat": {"p":[], "D": [], "KL": []},         #Gilbrat
    "gompertz": {"p":[], "D": [], "KL": []},        #Gompertz (Truncated Gumbel)
    "gumbel_l": {"p":[], "D": [], "KL": []},        #Left Sided Gumbel, etc.
    "gumbel_r": {"p":[], "D": [], "KL": []},        #Right Sided Gumbel
    "halfcauchy": {"p":[], "D": [], "KL": []},      #Half Cauchy
    "halflogistic": {"p":[], "D": [], "KL": []},    #Half Logistic
    "halfnorm": {"p":[], "D": [], "KL": []},        #Half Normal
    "hypsecant": {"p":[], "D": [], "KL": []},       #Hyperbolic Secant
    "invgamma": {"p":[], "D": [], "KL": []},        #Inverse Gamma
    "invgauss": {"p":[], "D": [], "KL": []},        #Inverse Normal
    "invweibull": {"p":[], "D": [], "KL": []},      #Inverse Weibull
    "johnsonsb": {"p":[], "D": [], "KL": []},       #Johnson SB
    "johnsonsu": {"p":[], "D": [], "KL": []},       #Johnson SU
    "laplace": {"p":[], "D": [], "KL": []},         #Laplace
    "logistic": {"p":[], "D": [], "KL": []},        #Logistic
    "loggamma": {"p":[], "D": [], "KL": []},        #Log-Gamma
    "loglaplace": {"p":[], "D": [], "KL": []},      #Log-Laplace (Log Double Exponential)
    "lognorm": {"p":[], "D": [], "KL": []},         #Log-Normal
    "lomax": {"p":[], "D": [], "KL": []},           #Lomax (Pareto of the second kind)
    "maxwell": {"p":[], "D": [], "KL": []},         #Maxwell
    "mielke": {"p":[], "D": [], "KL": []},          #Mielke's Beta-Kappa
    "nakagami": {"p":[], "D": [], "KL": []},        #Nakagami
    "ncx2": {"p":[], "D": [], "KL": []},            #Non-central chi-squared
    "ncf": {"p":[], "D": [], "KL": []},             #Non-central F
    "nct": {"p":[], "D": [], "KL": []},             #Non-central Student's T
    "norm": {"p":[], "D": [], "KL": []},            #Normal (Gaussian)
    "pareto": {"p":[], "D": [], "KL": []},          #Pareto
    "pearson3": {"p":[], "D": [], "KL": []},        #Pearson type III
    "powerlaw": {"p":[], "D": [], "KL": []},        #Power-function
    "powerlognorm": {"p":[], "D": [], "KL": []},    #Power log normal
    "powernorm": {"p":[], "D": [], "KL": []},       #Power normal
    "rdist": {"p":[], "D": [], "KL": []},           #R distribution
    "reciprocal": {"p":[], "D": [], "KL": []},      #Reciprocal
    "rayleigh": {"p":[], "D": [], "KL": []},        #Rayleigh
    "rice": {"p":[], "D": [], "KL": []},            #Rice
    "recipinvgauss": {"p":[], "D": [], "KL": []},   #Reciprocal Inverse Gaussian
    "semicircular": {"p":[], "D": [], "KL": []},    #Semicircular
    "t": {"p":[], "D": [], "KL": []},               #Student's T
    "triang": {"p":[], "D": [], "KL": []},          #Triangular
    "truncexpon": {"p":[], "D": [], "KL": []},      #Truncated Exponential
    "truncnorm": {"p":[], "D": [], "KL": []},       #Truncated Normal
    "tukeylambda": {"p":[], "D": [], "KL": []},     #Tukey-Lambda
    "uniform": {"p":[], "D": [], "KL": []},         #Uniform
    "vonmises": {"p":[], "D": [], "KL": []},        #Von-Mises (Circular)
    "wald": {"p":[], "D": [], "KL": []},            #Wald
    "weibull_min": {"p":[], "D": [], "KL": []},     #Minimum Weibull (see Frechet)
    "weibull_max": {"p":[], "D": [], "KL": []},     #Maximum Weibull (see Frechet)
    "wrapcauchy": {"p":[], "D": [], "KL": []},      #Wrapped Cauchy
    "ksone": {"p":[], "D": [], "KL": []},           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign": {"p":[], "D": [], "KL": []}}       #Kolmogorov-Smirnov two-sided test for Large N

########################################################################################

def check(data, fct, verbose=False):
    #fit our data set against every probability distribution
    parameters = eval("scipy.stats."+fct+".fit(data)")
    #Applying the Kolmogorov-Smirnof two sided test
    D, p = scipy.stats.kstest(data, fct, args=parameters)

    f = eval("scipy.stats." + fct + ".freeze" + str(parameters))
    x = numpy.linspace(f.ppf(0.001), f.ppf(0.999), len(data))
    KL = scipy.stats.entropy(data, qk=f.pdf(x))

    if math.isnan(p): p=0
    if math.isnan(D): D=0

    if verbose:
        print(fct.ljust(16) + "p: " + str(p).ljust(25) + "D: " +str(D))

    return (fct, p, D, KL)

########################################################################################

def dist_check_main(data, iterative, processes=1, exclude=10.0):

    #########################################################################################
    # parser = OptionParser()
    # parser.add_option("-f", "--file",      dest="filename",  default="",    type="string",       help="file with measurement data", metavar="FILE")
    # parser.add_option("-v", "--verbose",   dest="verbose",   default=False, action="store_true", help="print all results immediately (default=False)" )
    # parser.add_option("-t", "--top",       dest="top",       default=10,    type="int",          help="define amount of printed results (default=10)")
    # parser.add_option("-p", "--plot",      dest="plot",      default=False, action="store_true", help="plot the best result with matplotlib (default=False)")
    # parser.add_option("-i", "--iterative", dest="iterative", default=1,     type="int",          help="define number of iterative checks (default=1)")
    # parser.add_option("-e", "--exclude",   dest="exclude",   default=10.0,  type="float",        help="amount (in per cent) of exluded samples for each iteration (default=10.0%)" )
    # parser.add_option("-n", "--processes", dest="processes", default=-1,    type="int",          help="number of process used in parallel (default=-1...all)")
    # parser.add_option("-d", "--densities", dest="densities", default=False, action="store_true", help="")
    # parser.add_option("-g", "--generate",  dest="generate",  default=False, action="store_true", help="generate an example file")

    # read data from file or generate
    DATA = data

    best_dist = ""

    for i in range(0, iterative):

        if iterative == 1:
            data = DATA
        else:
            data = [value for value in DATA if random.random()>= exclude/100]

        results = Parallel(n_jobs=processes)(delayed(check)(data, fct) for fct in cdfs.keys())

        for res in results:
            key, p, D, KL = res
            cdfs[key]["p"].append(p)
            cdfs[key]["D"].append(D)
            cdfs[key]["KL"].append(KL)

        # print( "-------------------------------------------------------------------" )
        # print( "Top %d after %d iteration(s)" % (10, i+1, ) )
        # print( "-------------------------------------------------------------------" )
        best = sorted(cdfs.items(), key=lambda elem : scipy.median(elem[1]["KL"]), reverse=False)
        best_dist = best[0][0]

        # for t in range(10):
        #     fct, values = best[t]
        #     print( str(t+1).ljust(4), fct.ljust(16),
        #            "\tp: ", scipy.median(values["p"]),
        #            "\tD: ", scipy.median(values["D"]),
        #            "\tKL: ", scipy.median(values["KL"]),
        #            end="")
        #     if len(values["p"]) > 1:
        #         print("\tvar(p): ", scipy.var(values["p"]),
        #               "\tvar(D): ", scipy.var(values["D"]),
        #               "\tvar(KL): ", scipy.var(values["KL"]), end="")
        #     print()

    return best_dist