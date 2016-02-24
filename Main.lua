require 'DataContainer'
require 'TripletNet'
require 'cutorch'
require 'eladtools'
require 'optim'
require 'xlua'
require 'trepl'
require 'DistanceRatioCriterion'
require 'cunn'
----------------------------------------------------------------------


cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a Triplet network on CIFAR 10/100')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
-- cmd:option('-network',            'Model.lua',            'embedding network file - must return valid network.')
cmd:option('-network',            'ModelCuDNN.lua',            'embedding network file - must return valid network.')
cmd:option('-LR',                 0.1,                    'learning rate')
cmd:option('-LRDecay',            1e-6,                   'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              22,                     'number of epochs to train, -1 for unbounded')
cmd:option('-earlyStop',	7,	'stop training if validation error is increasing M times in a row')
cmd:option('-splitTrainVal',	0.8,	'split test set into test (80%) and validation (20%) set')
cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10 or Cifar100')
cmd:option('-size',               640000,                 'size of training list' )
cmd:option('-normalize',          1,                      '1 - normalize using only 1 mean and std values')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          false,                  'display first level filters after each epoch')


opt = cmd:parse(arg or {})
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
if opt.augment then
    require 'image'
end

----------------------------------------------------------------------
-- Model + Loss:

local EmbeddingNet = require(opt.network)
local TripletNet = nn.TripletNet(EmbeddingNet)
local Loss = nn.DistanceRatioCriterion()
TripletNet:cuda()
Loss:cuda()


local Weights, Gradients = TripletNet:getParameters()

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('Loaded')
    Weights:copy(w)
end

--TripletNet:RebuildNet() --if using TripletNet instead of TripletNetBatch

local data = require 'Data'
local SizeTrain = opt.size or 640000
local SizeTest = SizeTrain*0.1
local SizeValidation = SizeTrain*0.1

function ReGenerateTrain()
    return GenerateList(data.TrainData.label,3, SizeTrain)
end
local TrainList = ReGenerateTrain()
local TestList = GenerateList(data.TestData.label,3, SizeTest)
local ValidationList = GenerateList(data.ValidationData.label,3, SizeTest)


------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local weights_filename = paths.concat(opt.save, 'Weights.t7')
local log_filename = paths.concat(opt.save,'ErrorProgress')
local Log = optim.Logger(log_filename)
----------------------------------------------------------------------

print '==> Embedding Network'
print(EmbeddingNet)
print '==> Triplet Network'
print(TripletNet)
print '==> Loss'
print(Loss)

----------------------------------------------------------------------
local TrainDataContainer = DataContainer{
    Data = data.TrainData.data,
    List = TrainList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize,
    Augment = opt.augment,
    ListGenFunc = ReGenerateTrain
}

local TestDataContainer = DataContainer{
    Data = data.TestData.data,
    List = TestList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize
}

local ValidationDataContainer = DataContainer{
    Data = data.ValidationData.data,
    List = ValidationList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize
}


local function ErrorCount(y)
    if torch.type(y) == 'table' then
      y = y[#y]
    end
    return (y[{{},2}]:ge(y[{{},1}]):sum())
end

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}


local optimizer = Optimizer{
    Model = TripletNet,
    Loss = Loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
}

function Train(DataC)
    DataC:Reset()
    DataC:GenerateList()
    TripletNet:training()
    local err = 0
    local num = 1
    local x = DataC:GetNextBatch()

    while x do
        local y = optimizer:optimize({x[1],x[2],x[3]}, 1)
        err = err + ErrorCount(y)
        xlua.progress(num*opt.batchSize, DataC:size())
        num = num + 1
        x = DataC:GetNextBatch()
    end
    return (err/DataC:size())
end

function Test(DataC)
    DataC:Reset()
    TripletNet:evaluate()
    local err = 0
    local x = DataC:GetNextBatch()
    local num = 1
    while x do
        local y = TripletNet:forward({x[1],x[2],x[3]})
        err = err + ErrorCount(y)
        xlua.progress(num*opt.batchSize, DataC:size())
        num = num + 1
        x = DataC:GetNextBatch()
    end
    return (err/DataC:size())
end


local epochCount = 1
local minValidationError = 1
local earlyStoppingCount = 0
print '\n==> Starting Training\n'
while epochCount <= opt.epoch do
    print('Epoch ' .. epochCount .. '/' .. opt.epoch)
    
	--train 
	local ErrTrain = Train(TrainDataContainer)
	print('Training Error = ' .. ErrTrain)
    
	--test on validation set
    local ErrValidation = Test(ValidationDataContainer)
    print('Validation Error = ' .. ErrValidation)
	
	--update validation error
	if minValidationError > ErrValidation then
		minValidationError = ErrValidation
		earlyStoppingCount = 0
		--save weights
		torch.save(weights_filename, Weights)
	else
		earlyStoppingCount = earlyStoppingCount + 1
	end
	
	--check if early stopping is satisfied
	if earlyStoppingCount >= opt.earlyStop then
		print("\nEarly stop!")
		break
	end
	  
    epochCount = epochCount + 1
end

--Load weights with min validation error
w = torch.load(weights_filename)
Weights:copy(w)

--Final results
print("\n______________________________________\n")
--local ErrTrain = Train(TrainDataContainer)
--print('Training Error = ' .. ErrTrain)
local ErrValidation = Test(ValidationDataContainer)
print('Validation Error = ' .. ErrValidation)
local ErrTest = Test(TestDataContainer)
print('Test Error = ' .. ErrTest)

--Save new features
--Save test features
data.TestData.data = data.TestData.data:cuda()
file_test = io.open("new_features_test.txt", "w")
for i=1,data.TestData.data:size(1) do
        e = EmbeddingNet:forward(data.TestData.data[i])
        for j=1,e:size()[1] do
                file_test:write(e[j])
                file_test:write(" ")
        end
        file_test:write(">")
        file_test:write(data.TestData.label[i])
        file_test:write("\n")
end

--Save train features
--EmbeddingNet:cuda()
--EmbeddingNet:float()
data.TrainData.data = data.TrainData.data:cuda()
file_train = io.open("new_features_train.txt", "w")
for i=1,data.TrainData.data:size(1) do
	e = EmbeddingNet:forward(data.TrainData.data[i])
	for j=1,e:size()[1] do
		file_train:write(e[j])
		file_train:write(" ")
	end
	file_train:write(">")
	file_train:write(data.TrainData.label[i])
	file_train:write("\n")
end

--Save validation features
data.ValidationData.data = data.ValidationData.data:cuda()
file_valid = io.open("new_features_valid.txt", "w")
for i=1,data.ValidationData.data:size(1) do
        e = EmbeddingNet:forward(data.ValidationData.data[i])
        for j=1,e:size()[1] do
                file_valid:write(e[j])
                file_valid:write(" ")
        end
        file_valid:write(">")
        file_valid:write(data.TrainData.label[i])
        file_valid:write("\n")
end

