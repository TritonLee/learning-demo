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
index = 11;      % 图片相应的位置信息

%test data
addpath ./common/;
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
%testLabels = testLabels(1:10);
testLabels(testLabels==0) = 10; % Remap 0 to 10

testImage = testImages(:,:,index); % 选取其中一幅图片（index作为标准）作为testing data
testLabel = testLabels(index);    % 所选择图片对应的label 

for i=1:1:16
    colormap(gray);
    subplot(4,4,i);
    imshow(testImages(:,:,i)); 
end


% 显示选择的图片手写数字

figure;imshow(testImages(:,:,index)); 



%% load CNN training的结果
load('CNN_training.mat','Wc','bc','Wd','bd'); % training weights and bias

% cnnConvolve和cnnPool函数中卷积运算可以使用光计算来替代
activations = cnnConvolve(filterDim, numFilters, testImage, Wc, bc); %sigmoid(wx+b)
activationsPooled = cnnPool(poolDim, activations); 


activationsPooled = reshape(activationsPooled,[],1);
h = exp(bsxfun(@plus,Wd*activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
[~,preds] = max(probs); % 返回最大值所对应的位置（inference 得到的label）
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



