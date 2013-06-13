% Cross-validation learning procedure
%
classdef CrossValidationLearner < Learnable
    properties
        K;
    end
    
    methods
        function obj = CrossValidationLearner(objToLearn, K)
            if nargin < 2, K = 10; end
            if ~isa(objToLearn, 'Learnable')
                error('objToLearn should be a subclass of Learnable, but %s', class(objToLearn));
            end
            
            obj.objToLearn = objToLearn;
            obj.K = K;
        end
        
        function [obj, results] = learn(obj, data, options)
            
            idxCV = obj.buildIndexOfCrossValidation(data);
            
            objToLearn = obj.objToLearn;
            
            cumNumTestData = 0;
            cvResult = cell(1,obj.K);
            for k = 1: obj.K
                trainData = data(:, idxCV ~= k);
                testData = data(:, idxCV == k);
                
                options.testData = testData;
                cumNumTestData = cumNumTestData + size(testData,2);
                
                [obj.objToLearn, cvResult{k}] = objToLearn.learn(trainData, options);
            end
            
            results = cvResult{obj.K};
        end
        
        function idxCV = buildIndexOfCrossValidation(obj, trainData)
            N = size(trainData,2);
            idxCV = randi(obj.K, 1, N);
        end
    end
end