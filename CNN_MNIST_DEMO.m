

%% Revised by ZZ at 2021-12-10
% main function: MNIST_data generate, save and inference 


function varargout = CNN_MNIST_DEMO(varargin)
% CNN_MNIST_DEMO MATLAB code for CNN_MNIST_DEMO.fig
%      CNN_MNIST_DEMO, by itself, creates a new CNN_MNIST_DEMO or raises the existing
%      singleton*.
%
%      H = CNN_MNIST_DEMO returns the handle to a new CNN_MNIST_DEMO or the handle to
%      the existing singleton*.
%
%      CNN_MNIST_DEMO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CNN_MNIST_DEMO.M with the given input arguments.
%
%      CNN_MNIST_DEMO('Property','Value',...) creates a new CNN_MNIST_DEMO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CNN_MNIST_DEMO_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CNN_MNIST_DEMO_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CNN_MNIST_DEMO
% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CNN_MNIST_DEMO_OpeningFcn, ...
                   'gui_OutputFcn',  @CNN_MNIST_DEMO_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CNN_MNIST_DEMO is made visible.
function CNN_MNIST_DEMO_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CNN_MNIST_DEMO (see VARARGIN)

% Choose default command line output for CNN_MNIST_DEMO
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CNN_MNIST_DEMO wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CNN_MNIST_DEMO_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



% --- Executes on mouse press over axes background.
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global draw_enable;
global x;
global y;
draw_enable = 1;
if draw_enable
    position = get(gca,'currentpoint');
    x(1) = position(1);
    y(1) = position(3);
end

% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles   structure with handles and user data (see GUIDATA)
global draw_enable;
global x;
global y;
if draw_enable
    position = get(gca,'currentpoint');
    x(2) = position(1);
    y(2) = position(3);
    h1 = line(x,y,'EraseMode','xor','LineWidth',5,'color','black');
    x(1) = x(2);
    y(1) = y(2);
end


% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonUpFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global draw_enable
draw_enable = 0;



% --- Executes on button press in pushbutton1.
%  delete and clear the figure and data in the UI 
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);cla;
set(handles.edit1,'string',' ');
set(handles.edit3,'string',' ');


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h = getframe(handles.axes1);
imwrite(h.cdata,'output.jpg','jpg');
img = imread('output.jpg');
img = imresize(img,[28,28]);   % match the dim of the images 
img = rgb2gray(img);   
img = 255 - img;
load('CNN_training.mat','Wc','bc','Wd','bd'); % load CNN network with training weights and bias
filterDim = 9;   % Filter size for conv layer
numFilters = 10;  % Number of filters for conv layer
poolDim = 1;     % Pooling dimension (should divide imageDim-filterDim+1)
% cnnConvolve和cnnPool函数中卷积运算可以使用光计算来替代
activations = cnnConvolve(filterDim, numFilters, img, Wc, bc); %sigmoid(wx+b)
activationsPooled = cnnPool(poolDim, activations); 
activationsPooled = reshape(activationsPooled,[],1);
h = exp(bsxfun(@plus,Wd*activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
[~,preds] = max(probs);  % 返回最大值所对应的位置（inference 得到的label）
% to handle the case of label "0" when training "10"
if preds == 10
    s = 0;
else
    s = preds;
end
set(handles.edit1,'string',string(s));  % show results of inference from optical device
set(handles.edit3,'string',string(s));  % show results of inference from electrical device




% --- Executes during object creation, after setting all properties.
function text2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
% control the editor button for output MNIST inference number 
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function uipanel1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to uipanel1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
