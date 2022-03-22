%% function: testing for inference
% Revised at time 2021-11-25 by zongze 
%% 10000 images for testing

clc; clear; close;
tic;
%% Initialize Parameters
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes; 10 batch)
filterDim = 9;   % Filter size for conv layer
numFilters = 10;  % Number of filters for conv layer
poolDim = 2;     % Pooling dimension (should divide imageDim-filterDim+1)
index = 11;      % ͼƬ��Ӧ��λ����Ϣ

%test data
addpath ./common/;
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
%testLabels = testLabels(1:10);
testLabels(testLabels==0) = 10; % Remap 0 to 10

testImage = testImages(:,:,index); % ѡȡ����һ��ͼƬ��index��Ϊ��׼����Ϊtesting data
testLabel = testLabels(index);    % ��ѡ��ͼƬ��Ӧ��label 

for i=1:1:16
    colormap(gray);
    subplot(4,4,i);
    imshow(testImages(:,:,i)); 
end


% ��ʾѡ���ͼƬ��д����

figure;imshow(testImages(:,:,index)); 



%% load CNN training�Ľ��
load('CNN_training.mat','Wc','bc','Wd','bd'); % training weights and bias

% cnnConvolve��cnnPool�����о���������ʹ�ù���������
activations = cnnConvolve(filterDim, numFilters, testImage, Wc, bc); %sigmoid(wx+b)
activationsPooled = cnnPool(poolDim, activations); 


activationsPooled = reshape(activationsPooled,[],1);
h = exp(bsxfun(@plus,Wd*activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
[~,preds] = max(probs); % �������ֵ����Ӧ��λ�ã�inference �õ���label��
preds = preds';
if preds == 10
    s = 0;
else
    s = preds;
end
fprintf('The input number: %d\n',s);

%acc = sum(preds==testLabel)/length(preds);
%fprintf('Accuracy: %f\n',acc);
%plot(C,'LineWidth',2);
%xlabel('Number of Iterations');ylabel('Loss');
%grid on;


toc;



