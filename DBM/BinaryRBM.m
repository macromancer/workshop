%BinaryRBM Restricted Boltzmann Machines
%
% RBM models dealing with binary variables
% $ P(v,h;W,b,c) = P^* (v,h;W,b,c) / Z $
% $ P^* (v,h;W,b,c) = exp(-E(v,h;W,b,c)) $
% $ Z = Sigma_{v,h} P^* (v,h;W,b,c) $
% $ E(v,h;W,b,c) = -b'v -c'h - v'Wh $
%
% Examples 1
% objRBM = BinaryRBM(10, 50);
%
% Examples 2
% objRBM = BinaryRBM(W, b, c);
%
% See also RBM, GaussianRBM, Hypernetworks, DBM, RBMLearner, RBMLLE
%
% Copyright 2013- Kim, Kwonill
% kwonill.kim@gmail.com or kikim@bi.snu.ac.kr
% $Revision: 1.0 $  $Date: 2013/06/04 12:22:00 $
classdef BinaryRBM < RBM
    
    properties
    end
    
    methods
        function obj = BinaryRBM(varargin)
        %BinaryRBM Generator function
        % obj = BinaryRBM(D, M) : construct BinaryRBM with D visible nodes & M hidden nodes.
        %                         all parameters are 0.
        % obj = BinaryRBM(W,b,c) : construct BinaryRBM with weights W, bias b& c.
            if length(varargin) == 2
                D = varargin{1};
                M = varargin{2};
                obj.vhWeight = zeros(D,M);
                obj.visBias = zeros(D,1);
                obj.hidBias = zeros(M,1);
            elseif length(varargin) == 3
                [D,M] = size(varargin{1});
                if size(varargin{2},1) ~= D || size(varargin{2},2) ~= 1
                    error('Invalid visible bias size: shold be (Dx1), but (%dx%d)!!', size(varargin{2}));
                end
                if size(varargin{3},1) ~= M || size(varargin{3},2) ~= 1
                    error('Invalid hidden bias size: shold be (Mx1), but (%dx%d)!!', size(varargin{3}));
                end
                obj.vhWeight = varargin{1};
                obj.visBias = varargin{2};
                obj.hidBias = varargin{3};
            else
                error('Invalid arguments!!');
            end
        end
        
        % Compute energy function of RBM given V and H. $E(v,h;W,b,c)$
        % V: values of visible (D x N)
        % H: values of hidden (M x N)
        % E: $E(v,h;W,b,c)$. value of energy function of RBM (1 x N)
        function E = computeEnergy(obj, V, H)
            E = - obj.visBias'*V - obj.hidBias'*H - dot(V, obj.vhWeight*H);
        end
        
        % Compute log unnormalized Marginal Probability of V. $ln P^* (v)$
        % V: values of visible (D x N)
        % logUnnormMarginP: $ln P^* (v)$
        function logUnnormMarginP = computeLogUnnormalizedMarginalProb(obj, V)
            N = size(V,2);
            logUnnormMarginP = obj.visBias'*V + ...
                sum(log(1 + exp(repmat(obj.hidBias,1,N) + obj.vhWeight'*V)),1);
        end
        
        
        % Compute conditional probability of hidden given visible
        % V: values of hidden (D x N)
        % probH_V: $P(H|V)$ (M x N)
        function probH_V = getProbHGivenV(obj, V)
            N = size(V,2);
            probH_V = logsig(repmat(obj.hidBias,1,N) + obj.vhWeight'*V);
        end
        
        % Sample hidden given visible
        % V: values of visible (D x N)
        % H: sampled hidden (M x N)
        % probH_V: $P(H|V)$ (M x N)
        function [H, probH_V] = sampleHGivenV(obj, V)
            N = size(V,2);
            M = size(obj.vhWeight,2);
            probH_V = obj.getProbHGivenV(V);
            H = probH_V >= rand(M,N);
        end
        
        
        % Compute conditional probability of visible given hidden
        % H: values of hidden (M x N)
        % probV_H: P(V|H) (1 x N)
        function probV_H = getProbVGivenH(obj, H)
            N = size(H,2);
            probV_H = logsig(repmat(obj.visBias,1,N) + obj.vhWeight*H);
        end

        % Sample visible given hidden
        % H: values of hidden (M x N)
        % V: sampled visible (D x N)
        % probV_H: P(V|H) (D x N)
        function [V, probV_H] = sampleVGivenH(obj, H)
            N = size(H,2);
            D = size(obj.vhWeight,1);
            probV_H = obj.getProbVGivenH(H);
            V = probV_H >= rand(D,N);
        end
        
        % Sample new visible after numStep Gibbs sampling
        % V: sampled visible (D x N)
        % initV: initial visible (D x N)
        % numStep: # step of Gibbs sampling (scalar, default=1)
        function V = sampleNextV(obj, initV, numStep)
            if nargin < 3, numStep = 1; end
            
            V = initV;
            for i = 1:numStep
                H = obj.sampleHGivenV(V);
                V = obj.sampleVGivenH(H);
            end
        end
        
    end
    
end