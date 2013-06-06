%LLE Log Likelihood Estimator
%
% Parent class of Log Likelihood Estimator for RBM models
% 
% $ L(theta) = 1/N * Sigma_n ln P(x^{(n)};theta) $
%
% Examples 1
% objRBM = BinaryRBM(W, b, c);
% objLLE = LLEwithBruteMethodforRBM(objRBM);
% Z = objLLE.estimatePartitionFn(options);
% 
% See also RBM, LLEwithBruteMethodforRBM, LLEwithAISforRBM
%
% Copyright 2013- Kim, Kwonill
% kwonill.kim@gmail.com or kikim@bi.snu.ac.kr
% $Revision: 1.0 $  $Date: 2013/06/04 17:35:00 $
classdef LLE
    properties
        model = NaN; % Model object to estimate
    end
    
    methods(Abstract)
        % Estimate log of the partition function Z of the model
        % logZ: estimated log(Z) (scalar)
        % logZ_up: upper bound of estimated log(Z) (scalar)
        % logZ_down: lower bound of estimated log(Z) (scalar)
        [logZ, logZ_up, logZ_down] = estimateLogPartitionFn(obj, options)
        
        % Estimate log likelihood of the model given dataset X
        % X: dataset (DxN)
        % logL: log likelihood (scalar)
        logL = estimateLogLikelihood(obj, X, options)
    end
end