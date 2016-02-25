require 'nn'
--require 'cudnn'
local model = nn.Sequential()
--cudnn.benchmark = true
--cudnn.fastest = true
--cudnn.verbose = true
-- Convolution Layers

model:add(nn.SpatialConvolution(3, 64, 5, 7))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(nn.SpatialConvolution(64, 128, 3, 6))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(nn.SpatialConvolution(128, 256, 3, 5))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(nn.SpatialConvolution(256, 256, 1, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2))

model:add(nn.SpatialConvolution(256, 128, 1, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 3))
model:add(nn.View(128))


return model

