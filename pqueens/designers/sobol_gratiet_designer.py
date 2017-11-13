import numpy as np
import math
from itertools import combinations
from .abstract_designer import AbstractDesigner

class SobolGratietDesigner(AbstractDesigner):
    """ Pseudo Saltelli designer for experiments

        The purpose of this class is to generate the design neccessary to compute
        the sensitivity indices according to Le Gratiet method presented in [1],
        with the two samples X and X_tilde.

    References:

    [1] Le Gratiet, L., Cannamela, C., & Iooss, B. (2014).
        "A Bayesian Approach for Global Sensitivity Analysis
        of (Multifidelity) Computer Codes." SIAM/ASA Uncertainty
        Quantification Vol. 2. pp. 336-363,
        doi:10.1137/13926869

    Attributes:
        self.num_samples (int):   number of design points
        self.ps (np.array):       array with all combinations from our
                                  samples/design points
    """
    def __init__(self, params, seed, num_samples):
        """ Initialize SobolGratietDesigner object

        Args:
            params (dict):      Dictionnary with the definition of the problem.
                                Usuaally contains   the name of each parameter,
                                and for each parameter its type, size,
                                minimum and maximum values, its distribution and
                                its distribution    parameters.
            seed (int):         Seed for random number generation
            num_samples (int):  Number of desired (random) samples
        """
        numparams = len(params)
        # Number of sensitivity indices
        nb_indices = 2**numparams - 1
        # fix seed of random number generator
        np.random.seed(seed)
        self.num_samples = num_samples
        # arrays to store our two samples X and X_tilde
        X = np.ones((self.num_samples, numparams))
        X_tilde = np.ones((self.num_samples, numparams))
        # array with samples
        self.ps = np.ones((num_samples, nb_indices+1, numparams))

        i = 0
        for _, value in params.items():
            # get appropriate random number generator
            random_number_generator = getattr(np.random, value['distribution'])
            # make a copy
            my_args = list(value['distribution_parameter'])
            my_args.extend([num_samples])
            X[:, i] = random_number_generator(*my_args)
            X_tilde[:, i] = random_number_generator(*my_args)
            i += 1

        # making all possible combinations between the factors of X and X_tilde
        # loop to store all combinations with only one factor from X
        # First generate all indices of the combinations
        p = dict()
        ind = 0
        for k in range(numparams):
            for subset in combinations(range(numparams), k):
                p[ind] = np.asarray(subset)
                ind = ind+1
        del p[0]
        # Generate now the tensor of size (nb_indices+1, )
        self.ps[:, 0, :] = X
        self.ps[:, nb_indices, :] = X_tilde
        ps_temp = np.ones((self.num_samples, numparams))

        k = 1
        for i in range(nb_indices-1):
            ps_temp[:, p[i+1]] = X[:, p[i+1]]
            ps_temp[:, p[nb_indices-1-i]] = X_tilde[:, p[nb_indices-1-i]]
            self.ps[:, k, :] = ps_temp
            k = k +1

    def get_all_samples(self):
        """ Return all samples

        Returns:
            np.array:    array with all combinations for all samples. The array
                         is of size (num_samples,nb_indices+1,numparams),
                         it stores vertically the different possible combinations
                         to compute sensitivity indices.
        """
        return self.ps
