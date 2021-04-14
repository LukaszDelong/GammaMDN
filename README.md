# GammaMDN
Gamma Mixture Density Networks

**Description:**

This repository contains scripts which can be used to train a Gamma Mixture Density Network - a mixture of Gamma distributions with mixing probabilities, shape and rate parameters all depending on explanatory variables and modelled with deep neural networks.

The architecture of the Gamma MDN and the calibration algorithms are based on the paper:

Ł. Delong, M. Lindholm, M.V. W&uuml;thrich, 2021, Gamma Mixture Density Networks and their application to modelling insurance claim amounts.

The paper is available at www.lukaszdelong.pl and https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346. We refer to this paper for details.

**Instructions:**

1. Open the file gamma_MDN.R which includes all information on how to run the scripts and fit a Gamma MDN.

2. The file data_set.rda includes the data set used in the synthethic examples in the paper. The file has 12 columns:

    a) x_1,x_2,x_3 give the values of the three regressors;

    b) p_1,p_2,p_3,alpha_1,alpha_2,alpha_3,beta give the true values of the parameters for individual cases - one can use them for a comparison of the calibration results;

    c) y_gamma gives the responses for individual cases genereted from a mixture of three Gamma distributions;

    d) y_logn gives the responses for individual cases genereted from a mixture of three Lognormal distributions.
    
3. The file frempl_data_set.R includes the transformations applied on the data set from CASdatasets used in the actuarial example in the paper.
