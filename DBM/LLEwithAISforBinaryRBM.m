classdef LLEwithAISforBinaryRBM < LLE
    properties
    end
    
    methods
        % Generator function
        function obj = LLEwithAISforBinaryRBM(model)
            if ~isa(model, 'RBM')
                error('model should be an object of RBM, but %s', class(model)); 
            end
            obj.model = model;
        end
        
        % estimate log Z with AIS method
        % 
        % ет_0 = 0 < ет_k < ет_{k+1} < ет_K = 1
        % P_K б╒ P(v,h;W,b,c)
        % P_0 б╒ P(v,h;W=0,b=0,c=0)
        % P_k б╒ P(v,h;W_k=ет_k*W,b_k=ет_k*b,c=ет_k*c)
        %    Energy_k б╒ ет_k * Energy_0 + (1 - ет_k) * Energy_K
        % 
        % ratio of partition fns: $ Z_K / Z_0 = 1/N е╥е°^(n) $
        % importance weight: $ е°^(n) = 1/P^*_0(v_1) *
        %                                  P^*_1(v_1)/P^*_1(v_2) * 
        %                                  P^*_2(v_2)/P^*_2(v_3) * ... *
        %                                  P^*_k(v_k)/P^*_k(v_{k+1}) * ... *
        %                                  P^*_{K-1}(v_{K-1})/P^*_{K-1}(v_K) *
        %                                  P^*_K(v_K)
        %                    , where v_{k+1} ~ T_k(v,v_k)
        %                              ==> h_k     ~ P_k(h|v_k)
        %                                  v_{k+1} ~ P_k(v|h_k) $
        % intermediate importance weight: $ е°^(n)_0 = 1/P^*_0(v_1)
        %                                   е°^(n)_k = P^*_k(v_k)/P^*_k(v_{k+1})
        %                                   е°^(n)_K = P^*_K(v_K) $
        %
        % options.numSamples: # of sample points which will be used for AIS (scalar, default=100)
        %         betas: [ет_k], k=[0...K],  
        %                0 = ет_0 < ет_k < ет_{k+1} < ет_K = 1
        %               (1xK, default=[0:1e-3:1])
        %         numStep: num of Gibbs sampling (scalar, default=1)
        %         ratioStd: Z_up = Z + ratioStd*std(е°), Z_down = Z - ratioStd*std(е°)
        %                        (scalar, default=3)
        %
        function [logZ, logZ_up, logZ_down] = estimateLogPartitionFn(obj, options)
            if nargin < 2, options = {}; end
            if ~isfield(options,'numSamples') || isnan(options.numSamples), options.numSamples = 100; end
            if ~isfield(options,'betas') || length(options.betas) < 2, options.betas = [0:1e-3:1]; end
            if ~isfield(options,'numStep') || isnan(options.numStep), options.numStep = 1; end
            if ~isfield(options,'ratioStd') || isnan(options.ratioStd), options.ratioStd = 3; end
            
            if size(options.betas, 1) > 1, error('options.betas should be a row vector.'); end
            if ~(options.betas(1) == 0 && options.betas(end) == 1 && issorted(options.betas))
                error('options.betas should satisfiy the condition, 0 = ет_0 < ет_k < ет_{k+1} < ет_K = 1.');
            end
            
            W = obj.model.vhWeight;
            b = obj.model.visBias;
            c = obj.model.hidBias;
            
            K = size(options.betas,2) - 1;
            D = size(W, 1);
            N = options.numSamples;
            
            logZ_0 = obj.computeBaseRateLogZ();
            
            interModel = BinaryRBM(0*W, 0*b, 0*c);
            V_1 = obj.sampleFromBaseRateModel(N);
            logImportanceWeights = - interModel.computeLogUnnormalizedMarginalProb(V_1);
            
            V_k = V_1;
            for k = 1:K-1
                beta = options.betas(k+1);
%                 fprintf('beta=%g', beta);
                interModel = BinaryRBM(beta*W, beta*b, beta*c);
                V_k1 = interModel.sampleNextV(V_k, options.numStep);
                
                logIntermediateImportanceWeights = ...
                    interModel.computeLogUnnormalizedMarginalProb(V_k) ...
                    - interModel.computeLogUnnormalizedMarginalProb(V_k1);
                
                logImportanceWeights = logImportanceWeights + logIntermediateImportanceWeights;
                
                V_k = V_k1;
            end
            
            V_K = V_k;
            logImportanceWeights = logImportanceWeights + ...
                obj.model.computeLogUnnormalizedMarginalProb(V_K);
            
            logRatioOfZ = MLUtil.logSumExp(logImportanceWeights(:)) - log(N);
            meanLogIW = mean(logImportanceWeights(:));
            logStdRatioOfZ = log(std(exp(logImportanceWeights(:) - meanLogIW))) + meanLogIW - log(N)/2;
            
            logZ = logRatioOfZ + logZ_0;
            logZ_up   = MLUtil.logSumExp([log(options.ratioStd)+logStdRatioOfZ, logRatioOfZ]) + logZ_0;
            logZ_down = MLUtil.logDiffExp([log(options.ratioStd)+logStdRatioOfZ, logRatioOfZ]) + logZ_0;
            
            if ~isreal(logZ_down)
                logZ_down = 0;
            end
        end
        
        
        
        % Estimate log likelihood of the model given visible states
        % logL: estimated log likelihood (1xN)
        % V: visible states (DxN)
        % options.logZ: log partition fn (necessary)
        %
        % Example 1:
        % >> objRBM = BinaryRBM(W,b,c);
        % >> objLLE = LLEwithAISforBinaryRBM(objRBM);
        % >> options.logZ = objLLE.estimateLogPartitionFn();
        % >> logL = objLLE.estimateLogLikelihood(V, options);
        %
        function logL = estimateLogLikelihood(obj, V, options)
            if nargin < 2, options = {}; end
            if ~isfield(options,'logZ') || isnan(options.logZ)
                error('options.logZ is necessary!!');
            end
            
            logL = obj.model.computeLogUnnormalizedMarginalProb(V) - options.logZ;
        end
        
        % get logZ from the base-rate model(every parameter is zero)
        %   Z_0 = 2^(D+M) in binary RBM models
        % logZ_0 = (D+M) ln 2
        function logZ_0 = computeBaseRateLogZ(obj)
            [D,M] = size(obj.model.vhWeight);
            logZ_0 = (D+M) * log(2);
        end
        
        % Get samples from the base-rate model, v_0 ~ P(v;W=0,b=0,c=0)
        % In this case, all variables are independent and P(v_d = 1) = 1/2
        % N: # of sample points which will be used for AIS (scalar, default=100)
        % V_0: samples from the base-rate model (DxN)
        function V_0 = sampleFromBaseRateModel(obj, N)
            D = size(obj.model.vhWeight, 1);
            V_0 = rand(D, N);
        end
    end
    
    methods(Static)
        
        % get next samples of intermediate model started from initV
        % V_k: k th sample points (DxN)
        % V_k1: k+1 th sample points (DxN)
        % interModel: an object of RBM which exists between the base-rate model and the target model
        %             $P(v,h;ет_k)$
        % options.numStep: num of Gibbs sampling (default=1)
        function V_k1 = sampleFromIntermediateModel(V_k, interModel, options)
            if nargin < 3, options = {}; end
            if ~isfield(options,'numStep') || isnan(options.numStep), options.numStep = 1; end
            [D,N] = size(V_k);
            if size(interModel.vhWeight,1) ~= D
                error('num of row of initV should be %d, but %d', size(interModel.vhWeight,1), D);
            end
            
            V_k1 = interModel.sampleNextV(V_k, options.numStep);
        end
        
        % compute intermediate importance weight
        %
        % ratio of partition fns: $ Z_K / Z_0 = 1/N sum_n е°^(n) $
        % importance weight: $ е°^(n) = 1/P^*_0(v_1) *
        %                                  P^*_1(v_1)/P^*_1(v_2) * 
        %                                  P^*_2(v_2)/P^*_2(v_3) * ... *
        %                                  P^*_k(v_k)/P^*_k(v_{k+1}) * ... *
        %                                  P^*_{K-1}(v_{K-1})/P^*_{K-1}(v_K) *
        %                                  P^*_K(v_K)
        %                    , where v_{k+1} ~ T_k(v,v_k)
        %                              ==> h_k     ~ P_k(h|v_k)
        %                                  v_{k+1} ~ P_k(v|h_k)  $
        % intermediate importance weight: $ е°^(n)_k = P^*_k(v_k)/P^*_k(v_{k+1}) $
        %
        % logInterIW_k: $ln е°^(n)_k$ (k=1...K-1)
        % V_k1: k+1 th sample points (DxN)
        % V_k: k th sample points (DxN)
        % interModel: an object of RBM which exists 
        %             between the base-rate model and the target model
        %             $P(v,h;ет_k)$
        % options.numStep: num of Gibbs sampling (default=1)
        function [logInterImportanceWeight_k, V_k1] = ...
                computeLogIntermediateImportanceWeight(V_k, interModel, options)
            if nargin < 3, options = {}; end
            if ~isfield(options,'numStep') || isnan(options.numStep), options.numStep = 1; end
            
            V_k1 = LLEwithAISforBinaryRBM.computeLogIntermediateImportanceWeight(V_k, interModel, options);
            logUnnormalP_k_V_k = interModel.computeLogUnnormalizedMarginalProb(V_k);
            logUnnormalP_k_V_k1 = interModel.computeLogUnnormalizedMarginalProb(V_k1);
            
            logInterImportanceWeight_k = logUnnormalP_k_V_k - logUnnormalP_k_V_k1;
        end
    end
end