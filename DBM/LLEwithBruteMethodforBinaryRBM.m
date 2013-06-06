classdef LLEwithBruteMethodforBinaryRBM < LLE
    properties
    end
    
    methods
        % Generator function
        function obj = LLEwithBruteMethodforBinaryRBM(model)
            if ~isa(model, 'BinaryRBM')
                error('model should be an object of BinaryRBM, but %s', class(model)); 
            end
            obj.model = model;
        end
        
        % Estimate log of the partition function Z of the model
        % logZ: estimated log(Z) (scalar)
        % logZ_up: upper bound of estimated log(Z) (scalar)
        % logZ_down: lower bound of estimated log(Z) (scalar)
        % options.running='batch': batch
        %                ='online': online
        %                ='miniBatch': mini-batch with ontions.batchSize
        %                               (default)
        % options.batchSize: max num of states to run in each mini-batch iteration
        %                    (default=1e5)
        function [logZ, logZ_up, logZ_down] = estimateLogPartitionFn(obj, options)
            [D,M] = size(obj.model.vhWeight);
            if nargin < 2, options = {}; end
            if ~isfield(options,'running'), options.running = 'minibatch'; end
            
            if strcmpi(options.running,'batch')
                totalVisibleStates = MLUtil.enumerateEveryPossibleStates(D);
                logZ = MLUtil.logSumExp(obj.model.computeLogUnnormalizedMarginalProb(totalVisibleStates),2);
            elseif strcmpi(options.running,'online')
                logZ = 0;
                visState = NaN;
                possVals = num2cell(repmat([0 1], D, 1), 2)';
                while(true)
                    visState = MLUtil.enumerateNextState(visState, possVals);
                    if isnan(visState), return; end
                    logUnnormalMarginalProb = obj.model.computeLogUnnormalizedMarginalProb(visState);
                    logZ = MLUtil.logSumExp([logZ, logUnnormalMarginalProb], 2);
                end
            elseif strcmpi(options.running,'miniBatch')
                if ~isfield(options,'batchSize') || isnan(options.batchSize)
                    options.batchSize = 1e5;
                end
                logZ = 0;
                visStates = zeros(D, options.batchSize);
                possVals = num2cell(repmat([0 1], D, 1), 2)';
                visState = NaN;
                isEnd = false;
                while(true)
                    for i = 1:options.batchSize     % build mini-batch states
                        visState = MLUtil.enumerateNextState(visState, possVals);
                        if isnan(visState)
                            isEnd = true;
                            visStates = visStates(:,1:i-1);
                            break;
                        end
                        visStates(:,i) = visState;
                    end
                    fprintf('size(miniBatch) = %s \n', mat2str(size(visStates)));
                    if ~isempty(visStates)
                        logUnnormalMarginalProb = obj.model.computeLogUnnormalizedMarginalProb(visStates);
                        logZ = MLUtil.logSumExp([logZ, logUnnormalMarginalProb], 2);
                    end
                    if isEnd, break; end
                end
            else
                error('invalid options.running=%s', options.running);
            end
            logZ_up = logZ;
            logZ_down = logZ;
        end
        
        
        % Estimate log likelihood of the model given visible states
        % logL: estimated log likelihood (1xN)
        % V: visible states (DxN)
        % options.logZ: log partition fn (necessary)
        %
        % Example 1:
        % >> objRBM = BinaryRBM(W,b,c);
        % >> objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
        % >> logZ = objLLE.estimateLogPartitionFn();
        % >> options.logZ = logZ;
        % >> logL = objLLE.estimateLogLikelihood(V, options);
        %
        function logL = estimateLogLikelihood(obj, V, options)
            if nargin < 2, options = {}; end
            if ~isfield(options,'logZ') || isnan(options.logZ)
                error('options.logZ is necessary!!');
            end
            
            logL = obj.model.computeLogUnnormalizedMarginalProb(V) - options.logZ;
        end
        
    end
end