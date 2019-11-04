clc;
close all;
clear all;
N=1000;
prompt = 'Enter Sample Size: ';
N = input(prompt);
x = generate_random_data(N)';
X = split_data(x);
lkl_scores = [];
folds = 10;
for k=1:6
    for i=1:folds
        [Xval,Xtrain] = split_Val(X,i);
         model1 = fitgmdist(Xtrain,k);
         alpha1=model1.ComponentProportion;
         mu1=model1.mu;
         mu1=mu1';
         Sigma1=model1.Sigma;
         logLikelihood(i) = sum(log(evalGMM(Xval,alpha1,mu1,Sigma1)));
    end
    lkl_scores(k) = sum(logLikelihood)/folds;
end
figure(2)
plot([1 2 3 4 5 6],lkl_scores)
axis equal, hold on;
ylabel('Log Likelihood score ');
xlabel('K value');
    
    
   
function x = generate_random_data(N)

m(:,1) = [0;-1]; Sigma(:,:,1) = [1 0;0 1]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [-1;0]; Sigma(:,:,2) = [1 -0.4;-0.4 0.5]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [1;0]; Sigma(:,:,3) = [0.5 0;0 0.2];% mean and covariance of data pdf conditioned on label 1
m(:,4) = [0;1]; Sigma(:,:,4) = [0.1 0;0 0.1];
classPriors = [0.5,0.35,0.1,0.05]; thr = [0,cumsum(classPriors)];
 u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf;
gtrue=zeros(1,4);
for l = 1:4 
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
     figure(1), plot(x(1,indices),x(2,indices),'.'); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{7} Plot of IID samples with Class label L1&L2']);
    legend('Class L1','Class L2','Class L3','Class L4');
    gtrue(1,l)=length(indices);
    disp('Number of samples generated for Class:');
    disp(l);
    disp(length(indices));
end
end
 function X = split_data(x)
       [m n] = size(x);
       X=zeros(m/10,2,1);
       for i=1:10
           X(:,:,i)=x((m/10)*(i-1)+1:(m/10)*i,:);
         
       end
 end
 function [Xval,Xtrain] = split_Val(X,i)
 Xtrain= [];
  for j=1:10
      if j == i
          Xval = X(:,:,j);
      else
          Xtrain = [Xtrain;X(:,:,j)];
      end
  end
 end
function gmm = evalGMM(x,alpha,mu,Sigma)
x=x';
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end

%%%
end
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
 
 
 
        
