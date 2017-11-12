require 'paths'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'
require 'inn'
require 'loadcaffe'
require 'optnet'

local c = require 'trepl.colorize'

-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
-- Kaiming He, 2015
local function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end

function siamese_resnet34()
    local model_weights = paths.concat(opt.model_zoo_path, 'resnet-34.t7')
    local model_base = torch.load(model_weights)
    --keep AVG-pooling, View and the last FC layers respectively
    model_base:remove(model_base:size())
    model_base:remove(model_base:size())
    
    model_base:add(nn.View(-1):setNumInputDims(3))
    
    --create siamese architecture
    local siamese = nn.ParallelTable()
    --local siamese = nn.Parallel(2,2)
    local branch2 = model_base:clone()
    model_base:share(branch2, 'weight', 'bias', 'gradWeight', 'gradBias')
    siamese:add(model_base)
    siamese:add(branch2)

    --regression part
    local translation = nn.Sequential()
    local orientation = nn.Sequential()

    local f_t = nn.Linear(1024, 3)
    local f_o = nn.Linear(1024, 4)

    translation:add(f_t)
    orientation:add(f_o)
    local estimation_part = nn.ConcatTable()
                    :add(translation)
                    :add(orientation)

    --build up the whole structure
    local model = nn.Sequential()
                :add(siamese)
                :add(nn.JoinTable(2))
                :add(nn.Linear(2*512, 1024))
                :add(nn.BatchNormalization(1024))
                :add(nn.ReLU())

    if (opt.use_dropout) then
        model:add(nn.Dropout(0.5))
    end

    model:add(estimation_part)

   return model
end


if opt.weights ~= "" then
    print(c.green '==>' .. " loading model from pretained weights from file: " .. opt.weights)
    model = loadDataParallel(opt.weights, opt.nGPU)
else
    --model = siamese_densenet169()
    --model = siamese_squeezenet()
    model = siamese_resnet34()
    model:cuda()
    local input = torch.rand(10, 2, 3, opt.image_size, opt.image_size):cuda()
    opts_t = {inplace=true, mode='training'}
    optnet = require 'optnet'
    optnet.optimizeMemory(model, input, opts_t)
    model = makeDataParallel(model, opt.nGPU)
end

model = model:cuda()
return model
