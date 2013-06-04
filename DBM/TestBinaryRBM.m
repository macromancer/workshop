classdef TestBinaryRBM < matlab.unittest.TestCase

    methods (Test)
        function testGeneratorGivenSize(testCase)
            objRBM = BinaryRBM(10, 20);
            testCase.verifySize(objRBM.vhWeight,[10,20]);
            testCase.verifySize(objRBM.visBias, [10,1]);
            testCase.verifySize(objRBM.hidBias, [20,1]);
        end
        
        function testGeneratorGivenMatrix(testCase)
            W = rand(3, 5);
            b = rand(3, 1);
            c = rand(5, 1);
            objRBM = BinaryRBM(W, b, c);
            testCase.verifyEqual(objRBM.vhWeight, W);
            testCase.verifyEqual(objRBM.visBias, b);
            testCase.verifyEqual(objRBM.hidBias, c);
        end
        
        function testComputeEnergy(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            v = [1 0]';
            h = [1 0 1]';
            objRBM = BinaryRBM(W, b, c);
            expected = -8;
            testCase.verifyEqual(objRBM.computeEnergy(v,h), expected);
        end
        
        function testComputeEnergyGivenMultipleCases(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            v = [1 0; 0 1]';
            h = [1 0 1; 0 1 1]';
            objRBM = BinaryRBM(W, b, c);
            expected = [-8 -12];
            testCase.verifyEqual(objRBM.computeEnergy(v,h), expected);
        end
        
        function testComputeLogUnnormalizedMarginalProb(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            v = [1 0]';
            objRBM = BinaryRBM(W, b, c);
            logUnnormMarginP = objRBM.computeLogUnnormalizedMarginalProb(v);
            h = [0 0 0; 0 0 1; 0 1 0; 0 1 1;
                 1 0 0; 1 0 1; 1 1 0; 1 1 1]';
            expectedEnergy = objRBM.computeEnergy(repmat(v,1,8), h);
            expected = MLUtil.logSumExp(-expectedEnergy, 2);
            testCase.verifyEqual(logUnnormMarginP, expected, 'AbsTol', 1e-5);
            
            expected = [expected, expected];
            logUnnormMarginP = objRBM.computeLogUnnormalizedMarginalProb([v v]);
            testCase.verifyEqual(logUnnormMarginP, expected, 'AbsTol', 1e-5);
        end
        
        function testGetProbHGivenV(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            v = [1 0]';
            objRBM = BinaryRBM(W, b, c);
            PH_V = objRBM.getProbHGivenV(v);
            expected = logsig([2 1 5]');
            testCase.verifyEqual(PH_V, expected, 'AbsTol', 1e-5);
            
            PH_V = objRBM.getProbHGivenV([v v]);
            testCase.verifyEqual(PH_V, [expected expected], 'AbsTol', 1e-5);
        end
        
        function testSampleHGivenV(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            v = [1 0]';
            objRBM = BinaryRBM(W, b, c);
%             PH_V = objRBM.getProbHGivenV(v);
            PH_V = logsig([2 1 5]');
            H = objRBM.sampleHGivenV(repmat(v,1,1000));
            
            testCase.verifyEqual(mean(H,2), PH_V, 'AbsTol', 1e-1);
        end
        
        function testGetProbVGivenH(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            h = [1 0 1]';
            objRBM = BinaryRBM(W, b, c);
            PV_H = objRBM.getProbVGivenH(h);
            expected = logsig([5 10]');
            testCase.verifyEqual(PV_H, expected, 'AbsTol', 1e-5);
            
            PV_H = objRBM.getProbVGivenH([h h]);
            testCase.verifyEqual(PV_H, [expected expected], 'AbsTol', 1e-5);
        end
        
        function testSampleVGivenH(testCase)
            W = [1 2 3; 4 5 6];
            b = [1 0]';
            c = [1 -1 2]';
            h = [1 0 1]';
            objRBM = BinaryRBM(W, b, c);
%             PV_H = objRBM.getProbVGivenH(h);
            PV_H = logsig([5 10]');
            V = objRBM.sampleVGivenH(repmat(h,1,1000));
            
            testCase.verifyEqual(mean(V,2), PV_H, 'AbsTol', 1e-1);
        end
        
    end

end