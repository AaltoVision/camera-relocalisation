%% script to filter pose estimates from NN

clear all
addpath('utils')

%% Paths and estimations
file_id = fopen('cnn_part/data/NN_test_pairs_trained_nw.txt');
% read predictions from bin-file quat_trans_3
pred_file_id = fopen('cnn_part/results/results.bin', 'r');

data_cells = textscan(file_id, '%s %s %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
translation_gt_q = [data_cells{1,4+2} data_cells{1,5+2} data_cells{1,6+2}];
orientation_gt_q = [data_cells{1,7+2} data_cells{1,8+2} data_cells{1,9+2} ...
                  data_cells{1,10+2}];
translation_gt_db = [data_cells{1,11+2} data_cells{1,12+2} data_cells{1,13+2}];
orientation_gt_db = [data_cells{1,14+2} data_cells{1,15+2} data_cells{1,16+2} ...
                  data_cells{1,17+2}];

number_of_pairs = size(translation_gt_q, 1);

estimations = fread(pred_file_id, [7 Inf], 'float')';
fclose(file_id);
fclose(pred_file_id);

orientation_est = estimations(:, 1:4);
translation_est = estimations(:, 5:end);

%% main filtering stage

% intialize variables
orientation_err_deg = zeros(1, number_of_pairs);
translation_err_deg = zeros(1, number_of_pairs);

NN_count = 0; % counter over the NN from 1 to |NN|(=5)
allPairs = 0; % index to store the triangulated 3D camera locations
queryNum = 0;

% errors
err_trans = zeros(1, number_of_pairs/5);
err_quat = zeros(1, number_of_pairs/5);

% triangulations from mid-point algo
P = cell(1,5);
matches = zeros(1,4);
trans_tmp = zeros(10,3); % store all the possible camera locations from pairwise combinations of NN db images

% NN 
R_q_NN = zeros(5,4);
R_db_NN = zeros(5,4);
t_db_NN = zeros(5,3);
t_q_NN = zeros(5,3);

% estimated direction vectors from db to query
centers_rel_network = zeros(5,3);

falseC = 0;

for k=1:number_of_pairs
%     k
    %% ground truths
    %------- rotation
    R_q  = quat2rotm(orientation_gt_q(k,:)  ./ norm(orientation_gt_q(k,:)));
    R_db  = quat2rotm(orientation_gt_db(k,:)  ./ norm(orientation_gt_db(k,:)));
    %------- translation
    t_q = translation_gt_q(k,:)  ;
    t_db = translation_gt_db(k,:) ;
    
    %% estimations
    %------- rotation
    delR_est = quat2rotm(orientation_est(k,:) ./ norm(orientation_est(k,:)));
    R_q_est = R_db*delR_est;
    %------- translation
    t_est_center = translation_est(k,:)./norm(translation_est(k,:)); % (C_i - C_j)
    t_est = (R_db'*t_est_center'); %R_j'(C_i - C_j)
    
    %% -------------------------------------------------------------------------
    % store the estimations and db pose estimations for each NN related to
    % a query
    NN_count = NN_count + 1;
    R_q_NN(NN_count,:) = rotm2quat(R_q_est);
    t_q_NN(NN_count,:) = t_est;
    
    R_db_NN(NN_count,:) = rotm2quat(R_db);
    t_db_NN(NN_count,:) = t_db;
    
    P{NN_count} = [R_db' -R_db'*t_db'];
    
    centers_rel_network(NN_count,:) = t_est_center;
    
    % iterate over pairwise combinations {(1,2),(1,3),(2,3),(1,4),(2,4).....(4,5)}
    for i = 1:NN_count-1
        
        
        allPairs = allPairs + 1;
      
        % for triangulating a 3D camera position, we need the camera
        % matrices of the two db cameras: P1, P2 and the translation
        % directions from db to q: t1, t2 such that the z-cordinate is 1
        P1 = P{i};
        P2 = P{NN_count};      
        t1 = t_q_NN(i,:)./t_q_NN(i,3);
        t2 = t_q_NN(NN_count,:)./t_q_NN(NN_count,3);
        matches(1,1:2) = t1(1:2);
        matches(1,3:4) = t2(1:2);
        X = triangmidpoints(matches, P1, P2);
        
        trans_tmp(allPairs,:) = X;
    end
    
    
    %% Filtering stage
    
    % if all the NN for a query are processed 
    if NN_count == 5
        queryNum = queryNum + 1;

        % re-initialize the variables
        NN_count = 0;
        allPairs = 0;

        %% inlier process for trans
        
        % NaN can arise when the translation direction of two NN db
        % image-pairs used to triangulate the camera location have the same
        % direction. In the event all the pairwise combinations output same
        % translation directions, assign the translation vector of the NN
        % to the query
        if numel(find(isnan(trans_tmp))) == 10
            X_pred = t_db_NN(1,:);
            [err_trans(queryNum)] = norm(X_pred - t_q);
        else
            % remove the nan estimates
            nan_rows = any(isnan(trans_tmp),2) ;
            trans_tmp(nan_rows,:) = [];
           
            thresh_trans = 20; %10 degrees
            inlier_cnt_T = zeros(1,10); % store the inlier count estimates of the triangulated camera locs
            inlier_sum_T = zeros(1,10); % store the sum of residuals of distances for the inliers
            % estimate inliers for orientation
            % iterate over the triangulated 3D camera locs
            for h = 1:size(trans_tmp,1)
                
                % obtain the direction vectors from the database to query 
                centers_rel_triang = bsxfun(@minus, trans_tmp(h,:), t_db_NN);
                
                % make unit length
                centers_rel_triang = bsxfun(@rdivide,centers_rel_triang,sqrt(sum(abs(centers_rel_triang).^2,2)));
                
                % compute angular distance between the translation
                % directions predicted by the network: centers_rel_network and that
                % obtained from triangulation followed by backprojection:
                % centers_rel_triang
                
                angular_dist_T = 2*acos(abs(sum(centers_rel_triang.*centers_rel_network,2)))*180/pi;
              
                inlier_thresh_T = find(angular_dist_T<thresh_trans);
                inlier_cnt_T(h) = numel(inlier_thresh_T); %bcz the sample itself is an inlier to itself
                inlier_sum_T(h) = sum(angular_dist_T(inlier_thresh_T));
                
            end
            
            % select the best inlier
            [init_estimate_inlier_T, init_ID_T] = max(inlier_cnt_T);
          
            sim_inlier_cnt_T = find(inlier_cnt_T == init_estimate_inlier_T); % find other estimates with similar inlier counts
            
            if numel(sim_inlier_cnt_T) > 1 %if exists such other estimate
                
                  % OPtion 1: average the candidates
                  X_best = mean(trans_tmp(sim_inlier_cnt_T,:));
                  
% %                   
%                 % OPtion 2: select the inlier estimate with least residual sum
%                 [all_estimates_T, all_ID_T] = min(inlier_sum_T(sim_inlier_cnt_T));
%                 %             all_ID = randi([1 numel(sim_inlier_cnt)],1,1); % if randomly chosen
%                 X_best = trans_tmp(sim_inlier_cnt_T(all_ID_T),:);
                err_trans(queryNum) = norm(X_best - t_q);
                
                %             % take the inlier with the best estimate using GT
                %             inl_dist_GT = 2*acos(abs(sum(bsxfun(@times, R_qs(sim_inlier_cnt,:),rotm2quat(R_q)),2)))*180/pi;
                %             err_quat(queryNum) = min(inl_dist_GT);
            else
                X_best = trans_tmp(init_ID_T,:);
                err_trans(queryNum) = norm(X_best - t_q);
            end
%             
        end

        %% filtering process for rotation
        
        thresh_ort = 20; %10 degrees
        inlier_cnt = zeros(1,5); % store the inlier count estimates of the triangulated camera locs
        inlier_sum = zeros(1,5); % store the sum of residuals of distances for the inliers
        % iterate over the rotation estimates obtained from NN
        for h = 1:5
            
            % compute the angular distance between the current estimate of
            % query rotation R_q_NN(h,:) as indexed by h and the rest of
            % the estimations. 
            angular_dist = 2*acos(abs(sum(bsxfun(@times, R_q_NN(h,:),R_q_NN),2)))*180/pi;
            
            inlier_thresh = find(angular_dist<thresh_ort);
            inlier_cnt(h) = numel(inlier_thresh)-1; % bcz the sample itself is an inlier to itself
            inlier_sum(h) = sum(angular_dist(inlier_thresh));
            
        end
        
        % select the best inlier
        [init_estimate_inlier, init_ID] = max(inlier_cnt);

        sim_inlier_cnt = find(inlier_cnt == init_estimate_inlier); % find other estimates with similar inlier counts
        
        if numel(sim_inlier_cnt) > 1 %if exists such other estimate
            
            % OPtion 1: average the candidates
            for inl = 1:numel(sim_inlier_cnt)
                
               R_inl(:,:,inl)  = quat2rotm(R_q_NN(sim_inlier_cnt(inl),:));
                
            end
            
            R_avg = dqq_L1_mean_rotation_matrix(R_inl);
            q_best = rotm2quat(R_avg);
%             % OPtion 2: select the inlier estimate with least residual sum
%             [all_estimates, all_ID] = min(inlier_sum(sim_inlier_cnt));
%             q_best = R_qs(sim_inlier_cnt(all_ID),:);
            err_quat(queryNum) = 2*acos(abs(sum(q_best.*rotm2quat(R_q))))*180/pi;
            
        else
            q_best = R_q_NN(init_ID,:);
            err_quat(queryNum) = 2*acos(abs(sum(q_best.*rotm2quat(R_q))))*180/pi;
        end

    end
    
    
    
end

%% results
chess = median(err_quat(6001:8000));
fire = median(err_quat(1:2000));
heads = median(err_quat(8001:9000));
office = median(err_quat(2001:6000));
pumpkin = median(err_quat(15001:17000));
redkitchen = median(err_quat(10001:15000));
stairs = median(err_quat(9001:10000));
fprintf('Orientation error, deg:\n')
fprintf('chess: %.2f\n', chess)
fprintf('fire: %.2f\n', fire)
fprintf('heads: %.2f\n', heads)
fprintf('office: %.2f\n', office)
fprintf('pumpkin: %.2f\n', pumpkin)
fprintf('redkitchen: %.2f\n', redkitchen)
fprintf('stairs: %.2f\n', stairs)
fprintf('Mean averaged orientation: %.2f deg.\n', mean([chess fire heads office pumpkin redkitchen stairs]));
fprintf('--------------------------------------------------------\n');
chess = median(err_trans(6001:8000));
fire = median(err_trans(1:2000));
heads = median(err_trans(8001:9000));
office = median(err_trans(2001:6000));
pumpkin = median(err_trans(15001:17000));
redkitchen = median(err_trans(10001:15000));
stairs = median(err_trans(9001:10000));
fprintf('Translation error, m:\n')
fprintf('chess: %.2f\n', chess)
fprintf('fire: %.2f\n', fire)
fprintf('heads: %.2f\n', heads)
fprintf('office: %.2f\n', office)
fprintf('pumpkin: %.2f\n', pumpkin)
fprintf('redkitchen: %.2f\n', redkitchen)
fprintf('stairs: %.2f\n', stairs)
fprintf('Mean averaged translation: %.2f m.\n', mean([chess fire heads office pumpkin redkitchen stairs]));


