classdef TestMLUtil < matlab.unittest.TestCase

    methods (Test)
        function testLogSumExp(testCase)
            X = [1000 1001 1000; -1000 -999 -1000];
            lse1 = MLUtil.logSumExp(X,1);
            testCase.verifyEqual(lse1, [1000 1001 1000]);
        end
        
    end

end