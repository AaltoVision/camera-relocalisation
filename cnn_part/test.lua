local c = require 'trepl.colorize'
local tnt = require 'torchnet'

averaged_loss_orient_val_error_ = torch.Tensor(opt.max_epoch):fill(0)
averaged_loss_trans_val_error_  = torch.Tensor(opt.max_epoch):fill(0)


function test()
    local time = sys.clock()
    print(c.green '==>' .. " start testing after " .. (epoch_) .. " epoch(s)")

    local ntest_batches = val_data_size_ / opt.val_batch_size

    local val_indices = torch.range(1, val_data_size_):long():split(opt.val_batch_size)
    val_indices[#val_indices] = nil

    cutorch.synchronize()
    model:evaluate()

    for t,v in ipairs(val_indices) do
        xlua.progress(t, #val_indices)
        local val_batch_info = make_validation_minibatch(v)

        local mini_batch_data = val_batch_info.data:cuda()
        local orientation_gt = val_batch_info.quaternion_labels:cuda()
        local translation_gt = val_batch_info.translation_labels:cuda()

        cutorch.synchronize()
        collectgarbage()

        local outputs = model:forward({mini_batch_data[{{}, 1, {}, {}, {}}], mini_batch_data[{{}, 2, {}, {}, {}}]})
        local err = criterion:forward(outputs, {translation_gt, orientation_gt})

        meter_test_t:add(criterion.weights[1] * criterion.criterions[1].output)
        meter_test_q:add(criterion.weights[2] * criterion.criterions[2].output)
        cutorch.synchronize()
    end
    cutorch.synchronize()
    collectgarbage()

    time = sys.clock() - time
    averaged_loss_orient_val_error_[epoch_] = meter_test_q:value()
    averaged_loss_trans_val_error_[epoch_]  = meter_test_t:value()
    
    print(c.green '==>' .. " time taken for test = " .. (time) .. " s")
    print(c.green '==>' .. " val: orientation loss error (averaged): " .. meter_test_q:value())
    print(c.green '==>' .. " val: translation loss error (averaged): " .. meter_test_t:value())
    print(c.green '==>' .. " criterion.weights[1]: " .. criterion.weights[1])
    print(c.green '==>' .. " criterion.weights[2]: " .. criterion.weights[2])
end


function evaluation()
    quaternion_estimations  = torch.Tensor(test_data_size_, 4):zero()
    translation_estimations = torch.Tensor(test_data_size_, 3):zero()

    local time = sys.clock()
    print(c.green '==>' .. " start evaluation after " .. (epoch_) .. " epoch(s)")

    local ntest_batches = test_data_size_ / eval_batch_size_

    local test_indices = torch.range(1, test_data_size_):long():split(eval_batch_size_)

    cutorch.synchronize()
    model:evaluate()

    for t,v in ipairs(test_indices) do
        xlua.progress(t, #test_indices)
        local test_batch_info = make_evaluation_minibatch(v)

        local mini_batch_data = test_batch_info.data:cuda()

        cutorch.synchronize()
        collectgarbage()

        local outputs = model:forward({mini_batch_data[{{}, 1, {}, {}, {}}], mini_batch_data[{{}, 2, {}, {}, {}}]})
        translation_estimations[{{(t-1) * eval_batch_size_ + 1, t * eval_batch_size_}, {}}] = outputs[1]:float()
        quaternion_estimations[{{ (t-1) * eval_batch_size_ + 1, t * eval_batch_size_}, {}}] = outputs[2]:float()
        cutorch.synchronize()
    end
    cutorch.synchronize()
    collectgarbage()

    time = sys.clock() - time

    print(c.green '==>' .. " time taken for test = " .. (time) .. " s")

    --save results
    if not paths.dirp(opt.path_results) then
        paths.mkdir(opt.path_results)
    end
    local results_file = torch.DiskFile(paths.concat(opt.path_results, 'results.bin'), 'w'):binary()
 
    local results = torch.cat(quaternion_estimations, translation_estimations, 2)
    results_file:writeFloat(results:storage())
    results_file:close()
end
