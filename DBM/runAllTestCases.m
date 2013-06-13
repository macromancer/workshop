clear all;

testCase = TestBinaryRBM;
res = run(testCase);

testCase = TestMLUtil;
res = run(testCase);

testCase = TestLLEwithBruteMethodforBinaryRBM;
res = run(testCase);

testCase = TestLLEwithAISforBinaryRBM;
res = run(testCase);

testCase = TestLearner;
res = run(testCase);

