%RBM Restricted Boltzmann Machines
%
% Parent class for various RBM models
% $ P(v,h;W,b,c) = P^* (v,h;W,b,c) / Z $
% $ P^* (v,h;W,b,c) = exp(-E(v,h;W,b,c)) $
% $ Z = Sigma_{v,h} P^* (v,h;W,b,c) $
%
% Examples 1
% objRBM = BinaryRBM(10, 50);
%
% Examples 2
% objRBM = BinaryRBM(W, b, c);
%
% See also BinaryRBM, GaussianRBM, Hypernetworks, DBM, RBMLearner, RBMLLE
%
% Copyright 2013- Kim, Kwonill
% kwonill.kim@gmail.com or kikim@bi.snu.ac.kr
% $Revision: 1.0 $  $Date: 2013/06/03 21:22:00 $
classdef RBM

    properties
        % weights between visible and hidden nodes. 
        % visible x hidden. (DxM)
        vhWeight = zeros(0,1);
        
        % biases of visible nodes (Dx1)
        visBias = zeros(0,1);  
        
        % biases of hidden nodes (Mx1)
        hidBias = zeros(0,1);
    end
    
    methods
%         function obj = RBM(varargin)
%         %RBM Generator function
%         % obj = RBM(D, M) : construct RBM with D visible nodes & M hidden nodes.
%         %                     all parameters are 0.
%         % obj = RBM(W,b,c) : construct RBM with weights W, bias b& c.
%             if length(varargin) == 2
%                 D = varargin{1};
%                 M = varargin{2};
%                 obj.vhWeight = zeros(D,M);
%                 obj.visBias = zeros(D,1);
%                 obj.hidBias = zeros(M,1);
%             elseif length(varargin) == 3
%                 [D,M] = size(varargin{1});
%                 if size(varargin{2},1) ~= D || size(varargin{2},2) ~= 1
%                     error('Invalid visible bias size: shold be (Dx1), but (%dx%d)!!', size(varargin{2}));
%                 end
%                 if size(varargin{3},1) ~= M || size(varargin{3},2) ~= 1
%                     error('Invalid hidden bias size: shold be (Mx1), but (%dx%d)!!', size(varargin{3}));
%                 end
%                 obj.vhWeight = varargin{1};
%                 obj.visBias = varargin{2};
%                 obj.hidBias = varargin{3};
%             else
%                 error('Invalid arguments!!');
%             end
%         end
    end
    
    methods (Abstract)
        
        % Compute energy function of RBM given V and H. $E(v,h;W,b,c)$
        % V: values of visible (D x N)
        % H: values of hidden (M x N)
        % E: $E(v,h;W,b,c)$. value of energy function of RBM (1 x N)
        E = computeEnergy(obj, V, H)
        
        % Compute log unnormalized Marginal Probability of V. $ln P^* (v)$
        % V: values of visible (D x N)
        % logUnnormMarginP: $ln P^* (v)$
        logUnnormMarginP = computeLogUnnormalizedMarginalProb(obj, V)
        
        
        % Compute conditional probability of hidden given visible
        % V: values of hidden (D x N)
        % probH_V: $P(H|V)$ (M x N)
        probH_V = getProbHGivenV(obj, V)
        
        % Sample hidden given visible
        % V: values of visible (D x N)
        % H_V: sampled hidden (M x N)
        % probH_V: $P(H|V)$ (M x N)
        [H_V, probH_V] = sampleHGivenV(obj, V)
        
        
        % Compute conditional probability of visible given hidden
        % H: values of hidden (M x N)
        % probV_H: P(V|H) (1 x N)
        probV_H = getProbVGivenH(obj, H)
                
        % Sample visible given hidden
        % H: values of hidden (M x N)
        % V_H: sampled visible (D x N)
        % probV_H: P(V|H) (D x N)
        [V_H, probV_H] = sampleVGivenH(obj, H)
        
    end
end