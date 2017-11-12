require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local c = require 'trepl.colorize'
local tnt = require 'torchnet'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

torch.manualSeed(opt.manualSeed)
epoch_ = opt.epoch_number

-- intialize visdom Torch client:
local visdom = require 'visdom'
local plot = visdom{server = 'http://localhost', port = 8097}

local val_orientation_handle, val_translation_handle

-- Getting the multi-gpu functions
paths.dofile('gpu_util.lua')
-- Initializing data provider
paths.dofile('data_provider.lua')
init_data_provider()

paths.dofile('construct_minibatch.lua')
-- Different initialization methods of CNN model
paths.dofile('weight_init.lua')
-- Loading CNN model
paths.dofile('model.lua')
collectgarbage()

-- Create Criterion
local mse_1 = nn.MSECriterion() -- orientation loss
local mse_2 = nn.MSECriterion() -- translation loss

local w_mse_1 = 1
local w_mse_2 = opt.beta
criterion = nn.ParallelCriterion():add(mse_1, w_mse_1):add(mse_2, w_mse_2):cuda()
collectgarbage()

-- Create Meters
meter_test_q  = tnt.AverageValueMeter()
meter_test_t  = tnt.AverageValueMeter()
meter_train_q = tnt.AverageValueMeter()
meter_train_t = tnt.AverageValueMeter()

-- Loading the functions for training
paths.dofile('train.lua')
-- Loading the functions for testing
paths.dofile('test.lua')

--print(model)
local model_parameters, _ = model:getParameters()
print(c.blue '==>' .. ' Number of parameters in the model: ' .. model_parameters:size(1))

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

preamble = 'resnet34_new_parametrization_'

if opt.do_evaluation then
    print(c.blue '==>' .. ' Start evaluation...')
    evaluation()
    print(c.blue '==>' .. ' Evaluation is done. Exit.')
    do return end
end

if not paths.dirp(opt.logs) then
    paths.mkdir(opt.logs)
end

logger = optim.Logger(paths.concat(opt.logs, preamble .. '_' .. '[' .. opt.epoch_number .. ',' .. opt.max_epoch .. '].log'))

for i = opt.epoch_number,opt.max_epoch do
    if epoch_ == 31 then
        optimState.learningRate = 0.01
    elseif epoch_ == 51 then
        optimState.learningRate = 0.001
    elseif epoch_ == 81 then
        optimState.learningRate = 0.0001
    elseif epoch_ == 111 then
        optimState.learningRate = 0.00001
    elseif epoch_ == 151 then
        optimState.learningRate = 0.000001
    elseif epoch_ == 201 then
        optimState.learningRate = 0.0000001
    end
    
    train()
    test()
    if (epoch_ % 5 == 0) then
        collectgarbage()
        model:clearState()
        -- Saving the model
        if not paths.dirp(opt.snapshot_dir) then
            paths.mkdir(opt.snapshot_dir)
        end
        saveDataParallel(paths.concat(opt.snapshot_dir, preamble .. '_ep_' .. (epoch_) .. '.t7'), model)
    end
    

    if epoch_ >= 2 then
        -- plot orientation and translation loss of test data
        val_orientation_handle = plot:line{
            Y = averaged_loss_orient_val_error_:narrow(1, 1, epoch_),
            X = torch.range(1, epoch_),
            win = val_orientation_handle,
            options = {
                markers = true,
                title = 'Averaged orientation loss on validation data',
                xlabel = 'Epoch',
                ylabel = 'Loss',
            },
        }
        val_translation_handle = plot:line{
            Y = averaged_loss_trans_val_error_:narrow(1, 1, epoch_),
            X = torch.range(1, epoch_),
            win = val_translation_handle,
            options = {
                markers = true,
                title = 'Averaged translation loss on validation data',
                xlabel = 'Epoch',
                ylabel = 'Loss',
            },
        }
    end
    logger:add{['epoch'] = epoch_, 
               ['train_orient_loss']      = meter_train_q:value(),
               ['train_translation_loss'] = meter_train_t:value(),
               ['val_orient_loss']        = meter_test_q:value(),
               ['val_translation_loss']   = meter_test_t:value(),
               ['lr'] = optimState.learningRate}


    epoch_ = epoch_ + 1

    meter_test_q:reset()
    meter_test_t:reset()
    meter_train_q:reset()
    meter_train_t:reset()

end
