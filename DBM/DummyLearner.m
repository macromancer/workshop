% Dummy class for testing Learnable subclasses
%
classdef DummyLearner < Learnable
    
    methods
        function obj = DummyLearner(D)
            obj.objToLearn = {};
            obj.objToLearn.mean = zeros(D,1);
            obj.objToLearn.cov = eye(D);
            obj.objToLearn.numObservedSample = 0;
        end
        
        function [obj, results] = learn(obj, trainData, options)
            N = size(trainData,2);
            meanData = mean(trainData, 2);
            covData = cov(trainData');
            
            objModel = obj.objToLearn;
            objModel.mean = (objModel.mean * objModel.numObservedSample + meanData * N) /...
                            (objModel.numObservedSample + N);
            objModel.cov = (objModel.cov * (objModel.numObservedSample - 1) + covData * (N-1)) / ...
                                (objModel.numObservedSample + N - 1);
            objModel.numObservedSample = objModel.numObservedSample + N;
            
            obj.objToLearn = objModel;
            
            results = {};
            results.Error_mean = mean((objModel.mean(:) - options.trueMean(:)).^2);
            results.Error_cov = mean((objModel.cov(:) - options.trueCov(:)).^2);
            
            fprintf('N=%d, ¥ì_D=%s, ¥Ò_D=%s, #Observed=%d, ¥ì_M=%s, ¥Ò_M=%s, MSE=[%g,%g]\n', ...
                N, mat2str(meanData,3), mat2str(covData,3), objModel.numObservedSample, ...
                mat2str(objModel.mean,3), mat2str(objModel.cov,3), results.Error_mean, results.Error_cov);
        end
    end
end