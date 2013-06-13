% Abstract class for Learnable classes
% To be compatible for various learning procedures(cross-validation, 
% mini-batch, boosting etc.), it is designed to recursively call the learn() 
% method of the objToLearn, which is usually an object of subclasses of 
% Learnable.
% 
% Example 1:
% >> [trainData, testData] = MLUtil.splitDataSet(data, .3);
% >> objRBM = BinaryRBM(W,b,c);
% >> objRBMLearner = RBMLearnerWithPCD(objRBM, learningParams);
% >> objMBLearner  = MiniBatchLearner(objRBMLearner, maxBatchSize);
% >> objCVLearner  = CrossValidationLearner(objMBLearner, K);
% >> options.testData = testData;
% >> results = objCVLearner.learn(trainData, options);

classdef Learnable
    properties
        % objToLearn: object to be learned. It should be an object of
        %                 one of subclasses of Learnable or an object of a model.
        objToLearn;
    end
    
    
    methods(Abstract)
        
        % learn objToLearn with trainData
        % Abstract function
        % results: structure which contains various results of learning
        % trainData: training data set
        % options: various learning options
        [obj, results] = learn(obj, trainData, options)
    end
end