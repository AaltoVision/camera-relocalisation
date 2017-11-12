require 'image'

local home_path = "/ssd_storage/projects/Yar_colab"
local scenes_dicitionary_ = {[0] = 'office', [1] = 'meeting', [2] = 'kitchen', [3] = 'conference'}


local file = io.open(paths.concat(home_path, 'aalto_train.txt'))

local meanEstimate = {0,0,0}
local stdEstimate = {0,0,0}
local tm = torch.Timer()

local train_data_size = 0
if file then
    for line in file:lines() do
        local line_info = string.split(line, " ")
        local filename = line_info[1]
        local scene_id = tonumber(line_info[2])

        local img = image.load(home_path .. '/' .. scenes_dicitionary_[label_id] .. filename, 3, 'float')
        
        for j=1,3 do
            meanEstimate[j] = meanEstimate[j] + img[j]:mean()
            stdEstimate[j]  = stdEstimate[j]  + img[j]:std()
        end
        train_data_size = train_data_size + 1
    end
end
print(train_data_size)

for j=1,3 do
    meanEstimate[j] = meanEstimate[j] / train_data_size
    stdEstimate[j]  = stdEstimate[j] / train_data_size
end

local mean = meanEstimate
local std = stdEstimate

local cache = {}
cache.mean = mean
cache.std = std

print(mean, std)

local meanstdCache = paths.concat(home_path, 'meanstdCache_aalto.t7')
torch.save(meanstdCache, cache)

print('Time to estimate:', tm:time().real)
