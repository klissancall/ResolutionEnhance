% Author: Kevin Huynh

function [] = resolution_enhancement() 

    % Load the faces in grayscale
    face_dir = 'FacesDatabase 2/faces/jpg/';
    high_res_images = nan(256,256,200);
    for i=1:100
        if i < 10
            str=strcat('MPIf00',sprintf('%01d_0r.jpg', i));
            str2=strcat('MPIm00',sprintf('%01d_0r.jpg', i));
        elseif i < 100
            str=strcat('MPIf0',sprintf('%01d_0r.jpg', i));
            str2=strcat('MPIm0',sprintf('%01d_0r.jpg', i));
        else
            str=strcat('MPIf',sprintf('%01d_0r.jpg', i));
            str2=strcat('MPIm',sprintf('%01d_0r.jpg', i));
        end
        high_res_images(:,:,(2*i)-1) = im2uint8(rgb2gray(imread(strcat(face_dir,str))));
%         imshow(high_res_images(:,:,(2*i)-1), []);
        high_res_images(:,:,(2*i)) = im2uint8(rgb2gray(imread(strcat(face_dir,str2))));
%         imshow(high_res_images(:,:,(2*i)), []);
    end
    
    % Obtain 32 x 32 low-resolution images using bicubic interpolation
    low_res_images = nan(32,32,200);
    for i=1:200
        low_res_images(:,:,i) = imresize(high_res_images(:,:,i), [32,32], 'method', 'bicubic');
%         imshow(low_res_images(:,:,i), []);
    end
    
    % Split data into 100 training and 100 testing
    rng(10); % set random seed
    idx = randperm(200);
    indexToGroup1 = (idx<=100);
    indexToGroup2 = (idx>100);
    train_high_res_images = high_res_images(:,:,indexToGroup1);
    test_high_res_images = high_res_images(:,:,indexToGroup2);
    train_low_res_images = low_res_images(:,:,indexToGroup1);
    test_low_res_images = low_res_images(:,:,indexToGroup2);
    
    % Obtain shape of high-resolution training images using optical flow
    opticFlow = opticalFlowFarneback;   
    train_high_res_ref = train_high_res_images(:,:,1);
    train_high_res_flow = nan(131072,99);
    for i=2:100
        estimateFlow(opticFlow,train_high_res_ref);
        flow = estimateFlow(opticFlow,train_high_res_images(:,:,i));
        train_high_res_flow(:, i-1) = flow_zip(flow, 65536);
    end
    
    % Obtain shape of low-resolution images using optical flow
    opticFlow = opticalFlowFarneback;   
    train_low_res_ref = train_low_res_images(:,:,1);
    train_low_res_flow = nan(2048,99);
    test_low_res_flow = nan(2048,99);
    for i=2:100
        estimateFlow(opticFlow,train_low_res_ref);
        flow = estimateFlow(opticFlow,train_low_res_images(:,:,i));
        train_low_res_flow(:, i-1) = flow_zip(flow, 1024);
        estimateFlow(opticFlow,train_low_res_ref);
        flow = estimateFlow(opticFlow,test_low_res_images(:,:,i));
        test_low_res_flow(:, i-1) = flow_zip(flow, 1024);
    end
    
    % Obtain textures of high-resolution images using backwards warping
    
    % Obtain S+, by performing PCA on high and low-resolution shape
    train_shape = transpose(vertcat(train_low_res_flow,train_high_res_flow));
    coeff = pca(train_shape)
end

% Zip X and Y as a vector like how they want in S+ from two displacement matrices
function [vector] = flow_zip(flow, size)
    A = reshape(flow.Vx,[size,1]);
    B = reshape(flow.Vy,[size,1]);
    C = [A(:),B(:)].';
    vector = C(:); % zip x and y
end
    
    
    
    