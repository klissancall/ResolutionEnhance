% Author: Kevin Huynh, Xinyu Wang

function [] = resolution_enhacement() 

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
    
    % Obtain textures of high-resolution images
    % Obtain textures of low-resolution images
    train_low_text = zeros(32*32,99);
    train_high_text = zeros(256*256,99);
    test_low_text = zeros(32*32,99);
    for i = 2 : 100
        %index = (i-2)*1024+1;
        x_l = reshape(train_low_res_flow(1:2:2048,i-1),[32,32]);
        y_l = reshape(train_low_res_flow(2:2:2048,i-1),[32,32]);
        flow_low = cat(3,x_l,y_l);
        low_text = imwarp(train_low_res_images(:,:,i),flow_low);
        train_low_text(:,i-1) = reshape(low_text,[32*32,1]);
        x_h = reshape(train_high_res_flow(1:2:131072,i-1),[256,256]);
        y_h = reshape(train_high_res_flow(2:2:131072,i-1),[256,256]);
        flow_high = cat(3,x_h,y_h);
        high_text = imwarp(train_high_res_images(:,:,i),flow_high);
        train_high_text(:,i-1) = reshape(high_text,[256*256,1]);
        
        x_l = reshape(test_low_res_flow(1:2:2048,i-1),[32,32]);
        y_l = reshape(test_low_res_flow(2:2:2048,i-1),[32,32]);
        flow_low = cat(3,x_l,y_l);
        low_text = imwarp(test_low_res_images(:,:,i),flow_low);
        test_low_text(:,i-1) = reshape(low_text,[32*32,1]);
    end
    
    % Obtain S+, by performing PCA on high and low-resolution shape
    % Obtain T+, by performing PCA on high and low-resolution texture
    train_shape = transpose(vertcat(train_low_res_flow,train_high_res_flow));
    coeff_shape = pca(train_shape); % Rows of X correspond to observations and columns correspond to variables
    train_text = transpose(vertcat(train_low_text,train_high_text));
    coeff_text = pca(train_text);

    mean_shape = mean(train_shape);
    mean_text = mean(train_text);
    
    % Estimate a high-resolution shape from the given low-resolution shape by using S+
    % Estimate a high-resolution texture from the given low-resolution texture by using T+
    
    test_low_shape = test_low_res_flow - transpose(mean_shape(1:2048));
    test_low_texture = test_low_text - transpose(mean_text(1:1024));
    
    test_high_shape = nan(131072, 99);
    test_high_texture = nan(65536, 99);
    for i=1:99
        test_high_shape(:,i) = recursive_error_back_projection(test_low_shape(:,i), coeff_shape, coeff_text);
        test_high_texture(:,i) = recursive_error_back_projection(test_low_texture(:,i), coeff_shape, coeff_text);
    end
    
%     train_low_eig_shape = coeff_shape(1:2048,:);
%     alpha_shape = inv(transpose(train_low_eig_shape) * train_low_eig_shape) * transpose(train_low_eig_shape) * test_low_shape;
%     train_low_eig_text = coeff_text(1:1024,:);
%     alpha_text = inv(transpose(train_low_eig_text) * train_low_eig_text) * transpose(train_low_eig_text) * test_low_texture;
%     
%     test_high_shape = nan(131072,99);
%     for i=1:99
%         answer = zeros(131072,1);
%         for k=1:98
%             answer = answer + alpha_shape(k,i) * coeff_shape(2049:133120,k);
%         end
%         test_high_shape(:,i) = answer;
%     end
%     test_high_texture = nan(65536,99);
%     for i=1:99
%         answer = zeros(65536,1);
%         for k=1:98
%             answer = answer + alpha_text(k,i) * coeff_text(1025:66560,k);
%         end
%         test_high_texture(:,i) = answer;
%     end

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
function [high_res_estimate] = recursive_error_back_projection(low_res_data, coeff_shape, coeff_text)
    T1 = 1;
    T2 = 1;
    T = 10;
    t = 1;
    w = 0.5;
    prevdistance = 0;
    high_res_estimate = estimate_shape_or_texture(low_res_data, coeff_shape, coeff_text);
    low_res_estimate = downsample(high_res_estimate,64, 32);
    distance = norm(low_res_estimate - low_res_data);
    while (distance >= T1 || abs(prevdistance - distance) >= T2)
        prevdistance = distance;
        low_res_error = low_res_data - low_res_estimate;
        high_res_estimate = high_res_estimate + w * estimate_shape_or_texture(low_res_error, coeff_shape, coeff_text);
        low_res_estimate = downsample(high_res_estimate,64, 32);
        distance = norm(low_res_estimate - low_res_data);
        if (t >= T) 
            break
        end
        t = t + 1;
    end
end
    
function [high_res_estimate] = estimate_shape_or_texture(low_res_data, coeff_shape, coeff_text)
    if size(low_res_data, 1) == 2048
        train_low_eig_shape = coeff_shape(1:2048,:);
        alpha_shape = inv(transpose(train_low_eig_shape) * train_low_eig_shape) * transpose(train_low_eig_shape) * low_res_data;
        
        high_res_estimate = nan(131072, 1);
        answer = zeros(131072, 1);
        for k=1:98
            answer = answer + alpha_shape(k) * coeff_shape(2049:133120,k);
        end
        high_res_estimate(:) = answer;
    else
        train_low_eig_text = coeff_text(1:1024,:);
        alpha_text = inv(transpose(train_low_eig_text) * train_low_eig_text) * transpose(train_low_eig_text) * low_res_data;

        high_res_estimate = nan(65536, 1);
        answer = zeros(65536, 1);
        for k=1:98
            answer = answer + alpha_text(k) * coeff_text(1025:66560,k);
        end
        high_res_estimate(:) = answer;
    end
end
