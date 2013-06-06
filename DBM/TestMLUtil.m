classdef TestMLUtil < matlab.unittest.TestCase

    methods (Test)
        function testLogSumExp(testCase)
            X = [1000 1001 1000; -1000 -999 -1000];
            lse1 = MLUtil.logSumExp(X,1);
            testCase.verifyEqual(lse1, [1000 1001 1000]);
        end
        
        function testEnumerateEveryPossibleStates(testCase)
            states = MLUtil.enumerateEveryPossibleStates(3);
            expected = [0 0 0 0 1 1 1 1;
                        0 0 1 1 0 0 1 1;
                        0 1 0 1 0 1 0 1];
            testCase.verifyEqual(states, expected);
            
            states = MLUtil.enumerateEveryPossibleStates(2, [1 2 3]);
            expected = [1 1 1 2 2 2 3 3 3;
                        1 2 3 1 2 3 1 2 3];
            testCase.verifyEqual(states, expected);
            
            states = MLUtil.enumerateEveryPossibleStates({[0 1], [1 2 3]});
            expected = [0 0 0 1 1 1;
                        1 2 3 1 2 3];
            testCase.verifyEqual(states, expected);
        end
        
        function testEnumerateNextState(testCase)
            possVals = {[0 1], [1 2 3], [-1 1], [3 2 1]};
            currState = [0 2 1 2]';
            expected = [0 2 1 1]';
            state = MLUtil.enumerateNextState(currState, possVals);
            testCase.verifyEqual(state, expected);
            
            currState = expected;
            expected = [0 3 -1 3]';
            state = MLUtil.enumerateNextState(currState, possVals);
            testCase.verifyEqual(state, expected);
            
            currState = [1 3 1 1]';
            expected = NaN;
            state = MLUtil.enumerateNextState(currState, possVals);
            testCase.verifyEqual(state, expected);
        end
    end

end