

local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 relative pose estimation')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               2, 'Number of GPUs to use by default')
    cmd:option('-manualSeed',       333, 'Manually set RNG seed')
    ------------- Data options ------------------------
    cmd:option("-dataset_name", "7-Scenes", "either 7-Scenes or University")
    cmd:option("-dataset_src_path", "./data/images", "Image directory")
    cmd:option("-precomputed_data_path", "./data", "Path with meanstd info")
    cmd:option("-model_zoo_path", "./pretrained_models", "ImageNet pretrained (others) networks directory (in order to fine-tune)")
    cmd:option("-snapshot_dir", "./snapshots", "snapshot directory")
    cmd:option("-weights", "", "pretrained model to begin training from")
    cmd:option("-do_evaluation", false, "Evaluate trained network on test data")
    cmd:option("-use_dropout", false, "Add dropout to the network")
    cmd:option("-path_results", "./results", "Path to store evaluation results")
    cmd:option("-logs", "./logs", "Path to store evaluation results")
    cmd:option("-beta", 1, "beta parameter to balance loss function")
    cmd:option("-image_size", 224, "image size")
    cmd:option("-training_dataset_size", 39999, "size of the training dataset (number of pairs)")
    cmd:option("-validation_dataset_size", 10001, "size of the validation dataset (number of pairs)")
    --cmd:option("-test_dataset_size", 85000, "size of the test dataset (number of pairs)")
    ---------- Optimization options ----------------------
    cmd:option("-learning_rate", 0.1, "learning_rate")
    cmd:option("-momentum", 0.9, "momentum for sgd")
    cmd:option("-gamma", 0.001, "inv learning rate decay type: lr * (1 + gamma * epoch) ^ (-power)")
    cmd:option("-power", 0.5, "inv learning rate decay type: lr * (1 + gamma * epoch) ^ (-power)")
    cmd:option("-weight_decay", 0.00001, "weight decay")
    cmd:option("-beta1", 0.9, "first moment coefficient (for adam solver)")
    cmd:option("-beta2", 0.999, "second moment coefficient (for adam solver)")
     ------------- Training options --------------------
    cmd:option("-train_batch_size", 64, "training batch size")
    cmd:option("-val_batch_size", 40, "test batch size")
    --cmd:option("-test_batch_size", 50, "test batch size")
    cmd:option('-epoch_number', 1,     'Manual epoch number (useful on restarts)')
    cmd:option("-max_epoch",   250, "number of training epochs")
    cmd:text()

    local opt = cmd:parse(arg or {})

    return opt
end

return M
