%% function: testing for inference
% Revised at time 2021-11-25 by zongze 
%% 10000 images for testing

clc; clear; close;
%% Initialize Parameters
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes; 10 batch)
filterDim = 9;   % Filter size for conv layer
numFilters = 10;  % Number of filters for conv layer
poolDim = 1;     % Pooling dimension (should divide imageDim-filterDim+1)

tic;
%test data
addpath ./common/;
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%% 选取部分testing的图像
for i=1:1:16
    colormap(gray);
    subplot(4,4,i);
    imshow(testImages(:,:,i)); 
end

%% load CNN training 的结果
load('CNN_training.mat','Wc','bc','Wd','bd'); % training weights and bias

% cnnConvolve和cnnPool函数中卷积运算可以使用光计算来替代
activations = cnnConvolve(filterDim, numFilters, testImages, Wc, bc); %sigmoid(wx+b)
%activationsPooled = cnnPool(poolDim, activations); 
%activationsPooled = reshape(activationsPooled,[],length(testLabels));
activationsPooled = reshape(activations,[],length(testLabels));
h = exp(bsxfun(@plus,Wd * activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1)); % softmax function network 
[~,preds] = max(probs,[],1);
preds = preds';
acc = sum(preds==testLabels)/length(preds);
fprintf('Accuracy: %f\n',acc);
%plot(C,'LineWidth',2);
%xlabel('Number of Iterations');ylabel('Loss');
%grid on;


toc;



