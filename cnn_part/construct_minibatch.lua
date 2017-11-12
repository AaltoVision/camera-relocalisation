require 'nn'
require 'image'
require 'xlua'
t = require 'augment'


local transformer_train = t.Compose{
                              t.RandomCrop(224),
                              t.ColorJitter({
                                  brightness = 0.4,
                                  contrast = 0.4,
                                  saturation = 0.4,
                              })
                          }
local transformer_eval  = t.Compose{
                               t.CenterCrop(224)
                          }

function make_training_minibatch(rnd_idx_vec)

    local minibatch_size = rnd_idx_vec:size(1)
    local train_data = torch.Tensor(minibatch_size, 2, 3, opt.image_size, opt.image_size):zero()
    local train_quat = torch.Tensor(minibatch_size, 4):zero()
    local train_trans = torch.Tensor(minibatch_size, 3):zero()

    for k = 1,minibatch_size do
        local id = rnd_idx_vec[k]
        local label_id = train_scene_id_[id][1]

        local im1, im2

        local im1_tmp = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id] .. '/' .. train_filenames_[id][1], 3, 'float')
        local im2_tmp = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id] .. '/' .. train_filenames_[id][2], 3, 'float')
        
        local quat_gt_tmp = train_quaternions_gt_[id]
        local trans_gt_tmp = train_translation_gt_[id]

        --generate a random number [0,1]
        local a = torch.uniform()
        if (a > 0.5) then
            im1 = im2_tmp
            im2 = im1_tmp
            train_quat[k]  = torch.Tensor({quat_gt_tmp[1], -quat_gt_tmp[2], -quat_gt_tmp[3], -quat_gt_tmp[4]})
            train_trans[k] = -trans_gt_tmp
        else
            im1 = im1_tmp
            im2 = im2_tmp
            train_quat[k]  = quat_gt_tmp
            train_trans[k] = trans_gt_tmp
        end
        
        train_data[{k, 1, {}, {}, {}}] = transformer_train(im1)
        train_data[{k, 2, {}, {}, {}}] = transformer_train(im2)
    end

    -- normalization
    for n = 1,minibatch_size do
        for ch = 1,3 do
            train_data[{n, 1, ch, {}, {}}]:add(-mean_[ch])
            train_data[{n, 1, ch, {}, {}}]:div(std_[ch])
            
            train_data[{n, 2, ch, {}, {}}]:add(-mean_[ch])
            train_data[{n, 2, ch, {}, {}}]:div(std_[ch])
        end
    end
    --[[
    return {data = train_data, quaternion_labels  = train_quaternions_gt_:index(1, rnd_idx_vec), 
                               translation_labels = train_translation_gt_:index(1, rnd_idx_vec)}
                               --]]
    
    return {data = train_data, quaternion_labels  = train_quat, 
                               translation_labels = train_trans}
end


function make_validation_minibatch(rnd_idx_vec)

    local minibatch_size = rnd_idx_vec:size(1)
    local val_data = torch.Tensor(minibatch_size, 2, 3, opt.image_size, opt.image_size):zero()

    for k = 1,minibatch_size do
        local id = rnd_idx_vec[k]
        local label_id = val_scene_id_[id][1]

        local im1 = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id] .. '/' .. val_filenames_[id][1], 3, 'float')
        local im2 = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id] .. '/' .. val_filenames_[id][2], 3, 'float')

        im1 = transformer_eval(im1)
        im2 = transformer_eval(im2)

        val_data[{k, 1, {}, {}, {}}] = im1
        val_data[{k, 2, {}, {}, {}}] = im2

    end

    -- normalization
    for n = 1,minibatch_size do
        for ch = 1,3 do
            val_data[{n, 1, ch, {}, {}}]:add(-mean_[ch])
            val_data[{n, 1, ch, {}, {}}]:div(std_[ch])
            
            val_data[{n, 2, ch, {}, {}}]:add(-mean_[ch])
            val_data[{n, 2, ch, {}, {}}]:div(std_[ch])
        end
    end

    return {data = val_data, quaternion_labels  = val_quaternions_gt_:index(1, rnd_idx_vec), 
                             translation_labels = val_translation_gt_:index(1, rnd_idx_vec)}


end


function make_evaluation_minibatch(rnd_idx_vec)
    local minibatch_size = rnd_idx_vec:size(1)
    local test_data = torch.Tensor(minibatch_size, 2, 3, opt.image_size, opt.image_size):zero()

    for k = 1,minibatch_size do
        local id = rnd_idx_vec[k]
        local label_id = test_scene_id_[id]
        local im1 = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id[1]] .. test_filenames_[id][1], 3, 'float')
        local im2 = image.load(opt.dataset_src_path .. '/' .. scenes_dicitionary_[label_id[2]] .. test_filenames_[id][2], 3, 'float')
        
        test_data[{k, 1, {}, {}, {}}] = transformer_eval(im1)
        test_data[{k, 2, {}, {}, {}}] = transformer_eval(im2)

    end

    -- normalization
    for n = 1,minibatch_size do
        for ch = 1,3 do
            test_data[{n, 1, ch, {}, {}}]:add(-mean_[ch])
            test_data[{n, 1, ch, {}, {}}]:div(std_[ch])
            
            test_data[{n, 2, ch, {}, {}}]:add(-mean_[ch])
            test_data[{n, 2, ch, {}, {}}]:div(std_[ch])
        end
    end

    return {data = test_data}

end
