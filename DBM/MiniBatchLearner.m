% Mini-batch learning procedure
%
classdef MiniBatchLearner < Learnable
    properties
        maxBatchSize;
        maxIter;
    end
    
    methods
        function obj = MiniBatchLearner(objToLearn, maxBatchSize, maxIter)
            if nargin < 3, maxIter = 1; end
            if ~isa(objToLearn, 'Learnable')
                error('objToLearn should be a subclass of Learnable, but %s', class(objToLearn));
            end
            
            obj.objToLearn = objToLearn;
            obj.maxBatchSize = maxBatchSize;
            obj.maxIter = maxIter;
        end
        
        function [obj, results] = learn(obj, trainData, options)
            
            miniBatchSet = obj.buildMiniBatchDataset(trainData);
            numBatch = length(miniBatchSet);
            
            for iter = 1:obj.maxIter
                fprintf('iter=%d ', iter);
                cumNumData = 0;
                for i = 1: numBatch
                    fprintf('mini-batch=%d ', i);
                    cumNumData = cumNumData + size(miniBatchSet{i},2);
                    [obj.objToLearn, results] = obj.objToLearn.learn(miniBatchSet{i}, options);
                end
            end
            
        end
        
        function [miniBatchSet, idxBatch] = buildMiniBatchDataset(obj, trainData)
            N = size(trainData,2);
            numBatch = ceil(N / obj.maxBatchSize);
            idxBatch = randi(numBatch, 1, N);
            miniBatchSet = cell(1, numBatch);
            for i = 1:numBatch
                miniBatchSet{i} = trainData(:,idxBatch == i);
            end
        end
    end
end