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
        
    end
end