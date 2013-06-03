classdef TestRBM < matlab.unittest.TestCase

    methods (Test)
        function testGeneratorGivenSize(testCase)
            objRBM = RBM(10, 20);
            testCase.verifySize(objRBM.vhWeight,[10,20]);
            testCase.verifySize(objRBM.visBias, [10,1]);
            testCase.verifySize(objRBM.hidBias, [20,1]);
        end
        
        function testGeneratorGivenMatrix(testCase)
            W = rand(3, 5);
            b = rand(3, 1);
            c = rand(5, 1);
            objRBM = RBM(W, b, c);
            testCase.verifyEqual(objRBM.vhWeight, W);
            testCase.verifyEqual(objRBM.visBias, b);
            testCase.verifyEqual(objRBM.hidBias, c);
        end
    end

end