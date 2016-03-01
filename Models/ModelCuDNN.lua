require 'nn'
require 'cudnn'
local model = nn.Sequential()
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true
-- Convolution Layers

model:add(cudnn.SpatialConvolution(3, 64, 5, 7))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(64, 128, 3, 6))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(128, 256, 3, 5))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(256, 256, 1, 3))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 2))

model:add(cudnn.SpatialConvolution(256, 128, 1, 3))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(2, 3))
model:add(nn.View(128))


return model

