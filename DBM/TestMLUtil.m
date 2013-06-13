classdef TestMLUtil < matlab.unittest.TestCase

    methods (Test)
        function testLogSumExp(testCase)
            X = [1000 1001 1000; -1000 -999 -1000];
            lse = MLUtil.logSumExp(X,1);
            testCase.verifyEqual(lse, [1000 1001 1000]);
            
            X = [1000 1001 1000; -1000 -999 -1000] * 1e-2;
            lse = MLUtil.logSumExp(X,2);
            expected = log(sum(exp(X),2));
            testCase.verifyEqual(lse, expected, 'AbsTol', 1e-5);
        end
        
        function testLogDiffExp(testCase)
            X = [1000 1001 1000; -1000 -999 -1000];
            lde = MLUtil.logDiffExp(X,2);
            expected = [1000 + log(exp(1)-1) 1000 + log(1-exp(1));
                        -1000 + log(exp(1)-1) -1000 + log(1-exp(1))];
            testCase.verifyEqual(lde, expected, 'AbsTol', 1e-5);
            
            X = [1000 1001 1000; -1000 -999 -1000] * 1e-2;
            lde = MLUtil.logDiffExp(X,2);
            expected = log(diff(exp(X),[],2));
            testCase.verifyEqual(lde, expected, 'AbsTol', 1e-5);
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