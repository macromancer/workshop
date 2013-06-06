%MLUtil Utility functions
%
%
% Copyright 2013- Kim, Kwonill
% kwonill.kim@gmail.com or kikim@bi.snu.ac.kr
% $Revision: 1.0 $  $Date: 2013/06/04 13:22:00 $
classdef MLUtil
    
    methods (Static)
        
        % log(sum(exp([x1 x2 x3 ...]), dim))
        % https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        %
        % lse: results
        % X: input matrix (NxM)
        % dim: dimension to sum (scalar)
        %
        % This program was originally written by Sam Roweis
        function lse = logSumExp(X, dim)
            if(length(X(:))==1) lse=X; return; end

            xdims=size(X);
            if(nargin<2) 
              dim=find(xdims>1);
            end

            alpha = max(X,[],dim)-log(realmax)/2;
            repdims=ones(size(xdims)); repdims(dim)=xdims(dim);
            lse = alpha+log(sum(exp(X-repmat(alpha,repdims)),dim));
        end
        
        
        % Enumerate all possible states
        % (to compute log likelihood through brute-force method)
        %
        % states = MLUtil.enumerateEveryPossibleStates(numVar)
        % states = MLUtil.enumerateEveryPossibleStates(numVar, possVal)
        % states = MLUtil.enumerateEveryPossibleStates({possVal})
        %
        % Example 1:
        % >> states = MLUtil.enumerateEveryPossibleStates(3)
        % states = 
        % 0 0 0 0 1 1 1 1
        % 0 0 1 1 0 0 1 1
        % 0 1 0 1 0 1 0 1
        %
        % Example 2:
        % >> states = MLUtil.enumerateEveryPossibleStates(2, [1 2 3])
        % states = 
        % 1 1 1 2 2 2 3 3 3
        % 1 2 3 1 2 3 1 2 3
        %
        % Example 3:
        % >> states = MLUtil.enumerateEveryPossibleStates({[0 1], [1 2 3]})
        % states = 
        % 0 0 0 1 1 1
        % 1 2 3 1 2 3
        function states = enumerateEveryPossibleStates(varargin)
            numVar = 0;
            possVals = {};
            
            if length(varargin) == 1 && isnumeric(varargin{1}) && length(varargin{1}) == 1
                numVar = varargin{1};
                commonPossVals = [0 1];
                possVals = num2cell(repmat(commonPossVals, numVar, 1), 2)';
            elseif length(varargin) == 2 && isnumeric(varargin{1}) && length(varargin{1}) == 1 ...
                    && isnumeric(varargin{2})
                numVar = varargin{1};
                commonPossVals = varargin{2};
                possVals = num2cell(repmat(commonPossVals, numVar, 1), 2)';
            elseif length(varargin) == 1 && iscell(varargin{1})
                possVals = varargin{1};
                numVar = length(possVals);
            else
                error('Invalid arguments: %s', varargin);
            end
            
            lenVals = cellfun(@length, possVals);
            numPossStates = prod(lenVals);
            
            if numPossStates > 1e7
                error('Too many states!! %d', numPossStates); 
            end
            
            states = zeros(numVar, numPossStates);
            state = NaN;
            for i = 1:numPossStates
                state = MLUtil.enumerateNextState(state, possVals);
%                 display(mat2str(state'));
                states(:,i) = state;
            end
            
        end
        
        % Enumerate next state
        % (to get all possible state to compute log likelihood through brute-force method)
        % state: (Dx1)
        % currState: (Dx1)
        % possVals: (cell(Dx1))
        %
        % Example 1:
        % >> possVals = {[0 1], [1 2 3], [-1 1], [3 2 1]};
        % >> currState = [0 2 1 1]';
        % >> state = MLUtil.enumerateNextState(currState, possVals)
        % state = 
        % [0 3 -1 3]'
        %
        % Example 2:
        % >> possVals = {[0 1], [1 2 3], [-1 1], [3 2 1]};
        % >> state = MLUtil.enumerateNextState(NaN, possVals);
        % state = 
        % [0 1 -1 3]' % return the 1st state
        %
        % Example 3:
        % >> possVals = {[0 1], [1 2 3], [-1 1], [3 2 1]};
        % >> state = MLUtil.enumerateNextState([1 3 1 1], possVals);
        % state = 
        % NaN
        function nextState = enumerateNextState(currState, possVals, position)
            numVar = length(possVals);
            if nargin < 3, position = 0; end
            if isnan(currState)
                nextState = zeros(numVar, 1);
                for i = 1:numVar
                    nextState(i) = possVals{i}(1);
                end
                return
            end
            
            vals = possVals{end-position};
            currVal = currState(end-position);
            idx = find(vals == currVal);
            nextIdx = idx + 1;
            if nextIdx > length(vals)
                nextIdx = 1; 
                isCarry = true;
                
                if position == numVar-1
                    nextState = NaN;
                    return
                end
            else
                isCarry = false;
            end
            
            candidateState = currState;
            candidateState(end-position) = vals(nextIdx);
            
            if isCarry
                candidateState = MLUtil.enumerateNextState(candidateState, possVals, position + 1);
            end
            
            nextState = candidateState;
        end
    end
end