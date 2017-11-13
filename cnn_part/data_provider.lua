require 'nn'
require 'image'
require 'xlua'

local c = require 'trepl.colorize'

function init_data_provider()
    training_data_size_ = opt.training_dataset_size
    val_data_size_ = opt.validation_dataset_size
    

    if opt.dataset_name == '7-Scenes' then
        scenes_dicitionary_ = {[0] = 'chess', [1] = 'fire', [2] = 'heads', [3] = 'office', [4] = 'pumpkin', [5] = 'redkitchen', [6] = 'stairs'}
        test_data_size_ = 85000 --number of lines in NN_7Scenes.txt file
        eval_batch_size_ = 50   --should be a divisor of test_data_size_
    elseif opt.dataset_name == 'University' then
        scenes_dicitionary_ = {[0] = 'office', [1] = 'meeting', [2] = 'kitchen', [3] = 'conference'}
        test_data_size_ = 19915  --number of lines in NN_university.txt file
        eval_batch_size_ = 35    --should be a divisor of test_data_size_
    else
        print(c.red '==>' .. ' Dataset name is not correct. Check -dataset_name argument. Exit')
        do return end
    end

    -- loading precalculated mean and std of training dataset
    local mean_std_obj = { mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 },}
    mean_ = mean_std_obj.mean
    std_ = mean_std_obj.std

    if opt.do_evaluation then
        load_testing_data()
    else
        -- load training GT labels and filenames
        load_training_data()
        print(c.green '==>' .. ' Training GT labels have been loaded successfully')
        load_validation_data()
    end

    print(c.green '==>' .. ' Validation GT labels have been loaded successfully')
    
end


function load_training_data()
    train_filenames_ = {}
    train_scene_id_       = torch.IntTensor(training_data_size_, 1)
    train_translation_gt_ = torch.Tensor(training_data_size_, 3)
    train_quaternions_gt_ = torch.Tensor(training_data_size_, 4)

    local file = io.open(paths.concat(opt.precomputed_data_path, 'NN_test_pairs_trained_nw.txt'))
 
    if file then
        local id = 1
        for line in file:lines() do
            local line_info = string.split(line, " ")
            train_filenames_[id] = {line_info[1], line_info[2]}
            train_scene_id_[id] = tonumber(line_info[3])
            train_translation_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6]}) --/ torch.norm(torch.Tensor({line_info[4], line_info[5], line_info[6]}))
            --train_translation_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6]})
            train_quaternions_gt_[id] = torch.Tensor({line_info[7], line_info[8], line_info[9], line_info[10]})
            id = id + 1
        end
    end
end


function load_validation_data()
    val_filenames_      = {}
    val_scene_id_       = torch.IntTensor(val_data_size_, 1)
    val_translation_gt_ = torch.Tensor(val_data_size_, 3)
    val_quaternions_gt_ = torch.Tensor(val_data_size_, 4)

    local file = io.open(paths.concat(opt.precomputed_data_path, 'db_all_med_hard_wo_heads_valid.txt'))

    if file then
        local id = 1
        for line in file:lines() do
            local line_info = string.split(line, " ")
            val_filenames_[id] = {line_info[1], line_info[2]}
            val_scene_id_[id] = tonumber(line_info[3])
            val_translation_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6]}) --/ torch.norm(torch.Tensor({line_info[4], line_info[5], line_info[6]}))
            --val_translation_gt_[id] = torch.Tensor({line_info[4], line_info[5], line_info[6]})
            val_quaternions_gt_[id] = torch.Tensor({line_info[7], line_info[8], line_info[9], line_info[10]})
            id = id + 1
        end
    end
    --print(val_filenames_)
end


function load_testing_data()
    test_filenames_      = {}
    test_scene_id_       = torch.IntTensor(test_data_size_, 2)

    if opt.dataset_name == '7-Scenes' then
        local file = io.open(paths.concat(opt.precomputed_data_path, 'NN_7scenes.txt'))
    elseif opt.dataset_name == 'University' then
        local file = io.open(paths.concat(opt.precomputed_data_path, 'NN_university.txt'))
    else
        print(c.red '==>' .. ' Dataset name is not correct. Check -dataset_name argument. Exit')
        do return end
    end
    

    if file then
        local id = 1
        for line in file:lines() do
            local line_info = string.split(line, " ")
            test_filenames_[id] = {line_info[2], line_info[1]}
            test_scene_id_[id] = torch.IntTensor({tonumber(line_info[4]), tonumber(line_info[3])})
            id = id + 1
        end
    end
end
