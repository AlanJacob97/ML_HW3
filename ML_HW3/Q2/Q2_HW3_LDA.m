clc;clear all;close all;
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; 
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 3;3,6]; 
classPriors = [0.3,0.7]; thr = [0,cumsum(classPriors)];
N = 999; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf;
gtrue=zeros(1,2);
for l = 1:2 
    indices = find(thr(l)<=u & u<thr(l+1)); 
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.'); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{10} Plot of IID samples with Class label - & +']);
    legend('Class -','Class +');
    gtrue(1,l)=length(indices);
    disp('Number of samples generated for Class:');
    disp(l);
    disp(length(indices));
end
X=cat(2,x',L');
Sb= (m(:,1)-m(:,2))*(m(:,1)-m(:,2))';
Sw= Sigma(:,:,1)+Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1));
yLDA = wLDA'*x;
X=cat(2,X,zeros(length(yLDA),1));
for i=1:length(yLDA)
    if yLDA(i)<0
        X(i,4)=1;
    else
        X(i,4)=2;
    end
end
Linf=X(:,4)';
m1=0;
for i=1:length(yLDA)
    if X(i,3)~=X(i,4)
        m1=m1+1;
    end
end
disp(' The number of Misclassification errors:');
disp(m1);
disp('Probability of error:');
disp(m1/999);
ginf = zeros(1,2);
for l=1:2
    indices = find(Linf(1,:)==l);
    figure(2), plot(x(1,indices),x(2,indices),'.'); axis equal, hold on,
    ylabel('Feature value x2');
    xlabel('Feature value x1');
    title(['\fontsize{10}  Inferred Class labels - & +']);
    legend('Inferred Class -','Inferred Class +');
    ginf(1,l)=length(indices);
    disp('Number of samples inferre l=d as Class:');
    disp(l);
    disp(length(indices));
end

for l=1:2
    indices = find(Linf(1,:)==l);
    figure(3),scatter(yLDA(indices),zeros(1,length(indices)),'*');
    hold on
    ylabel('');
    xlabel('Data vector x projected onto W');
    title(['\fontsize{10} LDA discriminant function plot']);
    xline(0,'-.b');
    legend(' Inferred Class -','Threshold','Inferred Class 1');
end
    
    
    




