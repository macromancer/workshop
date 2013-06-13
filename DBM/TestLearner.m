classdef TestLearner < matlab.unittest.TestCase
    methods (Test)
        function testDummyLearner(testCase)
            D = 2; N = 1000;
            trueMean = randn(D,1);
            trueCov = cov(randn(3,D));
            
            trainData = chol(trueCov)' * randn(D, N) + repmat(trueMean, 1, N);
            options.trueMean = trueMean;
            options.trueCov = trueCov;
            objLearner = DummyLearner(D);
            
            fprintf('\n¥ì_true=%s, ¥Ò_true=%s\n', mat2str(trueMean,3), mat2str(trueCov,3));
            [objLearner, results] = objLearner.learn(trainData, options);
            
            testCase.verifyEqual(results.Error_mean, 0, 'AbsTol', 2e-2);
            testCase.verifyEqual(results.Error_cov, 0, 'AbsTol', 2e-2);
        end
        
        
        function testDummyLearner_Iter(testCase)
            D = 2; N = 100; maxIter=10;
            trueMean = randn(D,1);
            trueCov = cov(randn(3,D));
            fprintf('\n¥ì_true=%s, ¥Ò_true=%s\n', mat2str(trueMean,3), mat2str(trueCov,3));
            
            options.trueMean = trueMean;
            options.trueCov = trueCov;
            objLearner = DummyLearner(D);
            
            for i=1:maxIter
                trainData = chol(trueCov)' * randn(D, N) + repmat(trueMean, 1, N);

                [objLearner, results] = objLearner.learn(trainData, options);
            end
            testCase.verifyEqual(results.Error_mean, 0, 'AbsTol', 2e-2);
            testCase.verifyEqual(results.Error_cov, 0, 'AbsTol', 2e-2);
        end
        
        function testMiniBatchLearner_buildMiniBatchDataset(testCase)
            trainData = rand(3, 1234); maxBatchSize = 100;
            objMBLearner = MiniBatchLearner(DummyLearner(3), maxBatchSize);
            miniBatchSet = objMBLearner.buildMiniBatchDataset(trainData);
            testCase.verifyEqual(length(miniBatchSet), 13);
            testCase.verifyEqual(unique(cell2mat(miniBatchSet)','rows'), unique(trainData', 'rows'));
        end
        
        function testMiniBatchLearner(testCase)

            D = 2; N = 1000; maxBatchSize = 100; maxIter = 2;
            trueMean = randn(D,1);
            trueCov = cov(randn(3,D));
            fprintf('\n¥ì_true=%s, ¥Ò_true=%s\n', mat2str(trueMean,3), mat2str(trueCov,3));
            
            objLearner = DummyLearner(D);
            objMBLearner = MiniBatchLearner(objLearner, maxBatchSize, maxIter);
            
            trainData = chol(trueCov)' * randn(D, N) + repmat(trueMean, 1, N);
            options.trueMean = trueMean;
            options.trueCov = trueCov;
            [objMBLearner, results] = objMBLearner.learn(trainData, options);
            
            testCase.verifyEqual(results.Error_mean, 0, 'AbsTol', 2e-2);
            testCase.verifyEqual(results.Error_cov, 0, 'AbsTol', 2e-2);
            
        end
        
        
        function testCrossValidationLearner_buildCVSet(testCase)
            N = 1234; K=10;
            trainData = rand(3, N); 
            objCVLearner = CrossValidationLearner(DummyLearner(3), K);
            
            idxCV = objCVLearner.buildIndexOfCrossValidation(trainData);
            testCase.verifySize(idxCV, [1,N]);
            testCase.verifyGreaterThanOrEqual(idxCV, 1);
            testCase.verifyLessThanOrEqual(idxCV, K);
        end
                
        function testCrossValidationLearner(testCase)

            D = 2; N = 1000; K = 3;
            trueMean = randn(D,1);
            trueCov = cov(randn(3,D));
            fprintf('\n¥ì_true=%s, ¥Ò_true=%s\n', mat2str(trueMean,3), mat2str(trueCov,3));
            
            objLearner = DummyLearner(D);
            objCVLearner = CrossValidationLearner(objLearner, K);
            
            trainData = chol(trueCov)' * randn(D, N) + repmat(trueMean, 1, N);
            options.trueMean = trueMean;
            options.trueCov = trueCov;
            [objCVLearner, results] = objCVLearner.learn(trainData, options);
            
            testCase.verifyEqual(results.Error_mean, 0, 'AbsTol', 2e-2);
            testCase.verifyEqual(results.Error_cov, 0, 'AbsTol', 2e-2);
            
        end
        
    end
end
