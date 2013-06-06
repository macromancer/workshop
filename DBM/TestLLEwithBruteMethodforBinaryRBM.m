classdef TestLLEwithBruteMethodforBinaryRBM < matlab.unittest.TestCase

    methods (Test)
        function testGeneratorGivenSize(testCase)
            objRBM = BinaryRBM(10, 20);
            objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
            objModel = objLLE.model;
            testCase.verifySize(objModel.vhWeight,[10,20]);
            testCase.verifySize(objModel.visBias, [10,1]);
            testCase.verifySize(objModel.hidBias, [20,1]);
        end
        
        function testEstimateLogPartitionFn(testCase)
            D = 10; M = 15;
            W = zeros(D,M);
            b = zeros(D,1);
            c = zeros(M,1);
            expectedLogZ = (D+M) * log(2);
            objRBM = BinaryRBM(W,b,c);
            objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);

            options.running = 'batch';
            tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
            testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
            
            options.running = 'online';
            tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
            testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
            
            options.running = 'miniBatch';
            options.batchSize = ceil(2^D / 10);
            tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
            testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
        end
        
        function testEstimateLogLikelihood(testCase)
            D = 10; M = 10;
            W = rand(D,M);
            b = rand(D,1);
            c = rand(M,1);
            objRBM = BinaryRBM(W,b,c);
            objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
            logZ = objLLE.estimateLogPartitionFn();
            
            V = MLUtil.enumerateEveryPossibleStates(D);
            options.logZ = logZ;
            logL = objLLE.estimateLogLikelihood(V, options);
            
            % Sigma_x P(x) = 1
            % => log(Sigma_x exp(log P(x))) = log(1) = 0
            testCase.verifyEqual(MLUtil.logSumExp(logL), 0);
        end
    end

end