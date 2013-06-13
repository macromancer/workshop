classdef TestLLEwithAISforBinaryRBM < matlab.unittest.TestCase

    methods (Test)
        function testGenerator(testCase)
            objRBM = BinaryRBM(10, 20);
            objLLE = LLEwithAISforBinaryRBM(objRBM);
            objModel = objLLE.model;
            testCase.verifySize(objModel.vhWeight,[10,20]);
            testCase.verifySize(objModel.visBias, [10,1]);
            testCase.verifySize(objModel.hidBias, [20,1]);
        end
        
        function testComputeBaseRateLogZ(testCase)
            D = 8; M = 8;
            W = zeros(D,M);
            b = zeros(D,1);
            c = zeros(M,1);
            objRBM = BinaryRBM(W,b,c);
            objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
            expectedLogZ0 = objLLE.estimateLogPartitionFn();
            objLLEAIS = LLEwithAISforBinaryRBM(objRBM);
            logZ0 = objLLEAIS.computeBaseRateLogZ();
            testCase.verifyEqual(logZ0, expectedLogZ0, 'AbsTol', 1e-3);
        end
        
        function testEstimateLogPartitionFn(testCase)
            D = 10; M = 8;
            W = randn(D,M);
            b = randn(D,1);
            c = randn(M,1);
            objRBM = BinaryRBM(W,b,c);
            
            objLLE_Brute = LLEwithBruteMethodforBinaryRBM(objRBM);
            expectedLogZ = objLLE_Brute.estimateLogPartitionFn();
            
            objLLE_AIS = LLEwithAISforBinaryRBM(objRBM);
            [logZ, logZ_up, logZ_down] = objLLE_AIS.estimateLogPartitionFn();
            
            fprintf('[%g ~ %g ~ %g] vs. %g\n', logZ_down, logZ, logZ_up, expectedLogZ);
            
            testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-1);
        end
        
        function testEstimateLogLikelihood(testCase)
            D = 10; M = 8; N = 100;
            W = randn(D,M);
            b = randn(D,1);
            c = randn(M,1);
            V = randn(D,N);
            objRBM = BinaryRBM(W,b,c);
            
            objLLE_Brute = LLEwithBruteMethodforBinaryRBM(objRBM);
            options.logZ = objLLE_Brute.estimateLogPartitionFn();
            expectedLogL = objLLE_Brute.estimateLogLikelihood(V, options);
            
            objLLE_AIS = LLEwithAISforBinaryRBM(objRBM);
            options.logZ = objLLE_AIS.estimateLogPartitionFn();
            logL = objLLE_AIS.estimateLogLikelihood(V, options);
            
%             fprintf('%g vs. %g\n', [logL; expectedLogL]);
            testCase.verifyEqual(logL, expectedLogL, 'AbsTol', 2e-2);
        end
%         
%         function testEstimateLogPartitionFn(testCase)
%             D = 10; M = 15;
%             W = zeros(D,M);
%             b = zeros(D,1);
%             c = zeros(M,1);
%             expectedLogZ = (D+M) * log(2);
%             objRBM = BinaryRBM(W,b,c);
%             objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
% 
%             options.running = 'batch';
%             tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
%             testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
%             
%             options.running = 'online';
%             tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
%             testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
%             
%             options.running = 'miniBatch';
%             options.batchSize = ceil(2^D / 10);
%             tic;logZ = objLLE.estimateLogPartitionFn(options);toc;
%             testCase.verifyEqual(logZ, expectedLogZ, 'AbsTol', 1e-5);
%         end
%         
%         function testEstimateLogLikelihood(testCase)
%             D = 10; M = 10;
%             W = rand(D,M);
%             b = rand(D,1);
%             c = rand(M,1);
%             objRBM = BinaryRBM(W,b,c);
%             objLLE = LLEwithBruteMethodforBinaryRBM(objRBM);
%             logZ = objLLE.estimateLogPartitionFn();
%             
%             V = MLUtil.enumerateEveryPossibleStates(D);
%             options.logZ = logZ;
%             logL = objLLE.estimateLogLikelihood(V, options);
%             
%             % Sigma_x P(x) = 1
%             % => log(Sigma_x exp(log P(x))) = log(1) = 0
%             testCase.verifyEqual(MLUtil.logSumExp(logL), 0);
%         end
    end

end