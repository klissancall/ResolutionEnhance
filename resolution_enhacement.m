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
        low_res_images(:,:,i) = imresize(squeeze(high_res_images(:,:,i)), [32,32], 'method', 'bicubic');
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
    train_high_res_flow = opticalFlow.empty(99,0);
    for i=2:100
        estimateFlow(opticFlow,squeeze(train_high_res_ref));
        train_high_res_flow(i-1) = estimateFlow(opticFlow,train_high_res_images(:,:,i));
    end
    
    % Obtain shape of low-resolution images using optical flow
    opticFlow = opticalFlowFarneback;   
    train_low_res_ref = train_low_res_images(:,:,1);
    train_low_res_flow = opticalFlow.empty(99,0);
    test_low_res_flow = opticalFlow.empty(99,0);
    for i=2:100
        estimateFlow(opticFlow,squeeze(train_low_res_ref));
        train_low_res_flow(i-1) = estimateFlow(opticFlow,train_low_res_images(:,:,i));
        estimateFlow(opticFlow,squeeze(train_low_res_ref));
        test_low_res_flow(i-1) = estimateFlow(opticFlow,test_low_res_images(:,:,i));
    end
    
    
    
    
    