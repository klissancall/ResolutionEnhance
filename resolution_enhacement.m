% Author: Kevin Huynh

function [] = resolution_enhancement() 

    % Load the faces in grayscale
    face_dir = 'FacesDatabase/faces/jpg/';
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
    
    % Obtain textures of low-resolution images using backwards warping
    
    % Obtain S+, by performing PCA on high and low-resolution shape
    train_shape = transpose(vertcat(train_low_res_flow,train_high_res_flow));
    coeff = pca(train_shape); % Rows of X correspond to observations and columns correspond to variables
    
    % Obtain T+, by performing PCA on high and low-resolution texture
    
    % Estimate a high-resolution shape from the given low-resolution shape by using S+
    
    % Estimate a high-resolution texture from the given low-resolution texture by using S+

    % Synthesize a high-resolution facial image by forward warping the estimated texture with the estimated shape. 

end

% Zip X and Y as a vector like how they want in S+ from two displacement matrices
function [vector] = flow_zip(flow, size)
    A = reshape(flow.Vx,[size,1]);
    B = reshape(flow.Vy,[size,1]);
    C = [A(:),B(:)].';
    vector = C(:); % zip x and y
end
    
% Improve the high-resolution shape/texture by recursive error back-projection.
function [high_res_estimate] = recursive_error_back_projection(low_res_data, S)
    T1 = 1;
    T2 = 1;
    T = 10;
    t = 1;
    w = 1;
    prevdistance = 0;
%     high_res_estimate = 
    low_res_estimate = imresize(high_res_estimate, [32,32], 'method', 'bicubic');
    distance = imabsdiff(low_res_estimate, high_res_estimate);
    while (distance >= T1 || abs(prevdistance - distance) >= T2)
        prevdistance = distance;
        low_res_error = low_res_data - low_res_estimate;
%         high_res_estimate = high_res_estimate + w * 
        low_res_estimate = imresize(high_res_estimate, [32,32], 'method', 'bicubic');
        distance = imabsdiff(low_res_estimate, high_res_estimate);
        if (t >= T) 
            break
        end
        t = t + 1;
    end
end
    
    