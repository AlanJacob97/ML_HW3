clc;close all; clear all;
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; 
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 3;3,6]; 
classPriors = [0.3,0.7]; thr = [0,cumsum(classPriors)];
N = 999; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf;
gtrue=zeros(1,2);
for l = 1:2 
    indices = find(thr(l)<=u & u<thr(l+1)); 
    L(1,indices) = (l-1)*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.'); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{10} Plot of IID samples with Class labels - & +']);
    legend('Class -','Class +');
    gtrue(1,l)=length(indices);
    disp('Number of samples generated for Class:');
    disp(l);
    disp(length(indices));
end
X = x';
% Computing cost and Gradient
y = L';
[m, n] = size(X);
initial_Theta = zeros((n+1),1);
[J, grad] = calc_Cost(initial_Theta,X,y);
options = optimset('GradObj','on','MaxIter',400);
theta = fminunc(@(t)calc_Cost(t,X,y),initial_Theta,options);
predictions = predict(theta,X);
c=0;
for i=1:999
    if predictions(i,1)~=L(1,i)
        c=c+1;
    end
end
disp(' The number of Misclassification errors:');
disp(c);
disp('Probability of error:');
disp(c/999);
Linf = predictions';
for l=1:2
    indices = find(Linf(1,:)==(l-1));
    figure(2), plot(x(1,indices),x(2,indices),'.'); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{10}  Inferred Class labels - & +']);
    legend('Inferred Class - ','Inferred Class +');
    ginf(1,l)=length(indices);
    disp('Number of samples inferred as Class:');
    disp(l);
    disp(length(indices));
end



function [J, grad] = calc_Cost(Theta,X,y)
m = size(X,1);
X = [ones(m,1) X];
h = 1./(1+exp(-X*Theta));
J = -(1/m)*sum(y.*log(h) + (1-y).*log(1-h));
grad = zeros(size(Theta,1),1);
for i=1:size(grad)
    grad(i) = 1/m*sum((h-y)'*X(:,i));
end
end
function p = predict(theta,X)
m = size(X,1);
X = [ones(m,1) X];
p = round(1./(1+exp(-X*theta)));
end







