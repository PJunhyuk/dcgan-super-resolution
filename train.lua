----------------------------------------------------------------------------
-- prepare require elements
require 'torch'
require 'nn'
require 'optim'
require 'image'

-- set default option
opt = {
    dataset = 'folder',       -- imagenet / lsun / folder
    batchSize = 100,
    loadSize = 96,
    fineSize = 64,
    ngf = 12,               -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    nThreads = 4,           -- #  of data loading threads to use
    niter = 1,             -- #  of iter at starting learning rate
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
    name = 'dcgan-sr-test-1',
}

-- check live opt settings
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- set threads
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------

local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m:noBias()
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end

local nc = 3
local ndf = opt.ndf
local ngf = opt.ngf

-- simplify library of nn
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

-- set network of Generator
-- local netG = nn.Sequential()
-- -- nc x 32 x 32
-- netG:add(SpatialFullConvolution(nc, ngf * 8, 9, 9, 1, 1, 0, 0))
-- netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- -- ngf*8 x 40 x 40
-- netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 9, 9, 1, 1, 0, 0))
-- netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- -- ngf*4 x 48 x 48
-- netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 9, 9, 1, 1, 0, 0))
-- netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- -- ngf*2 x 56 x 56
-- netG:add(SpatialFullConvolution(ngf * 2, nc, 9, 9, 1, 1, 0, 0))
-- netG:add(nn.Tanh())
-- -- nc x 64 x 64

-- set network of Generator
-- local netG = nn.Sequential()
-- -- nc x 32 x 32
-- netG:add(SpatialConvolution(nc, ngf, 5, 5))
-- -- ngf x 28 x 28
-- netG:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
-- -- ngf*2 x 14 x 14
-- netG:add(SpatialFullConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- -- ngf*4 x 28 x 28
-- netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 5, 5))
-- netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- -- ngf*2 x 32 x 32
-- netG:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
-- netG:add(nn.Tanh())
-- -- nc x 64 x 64

-- set network of Generator
local netG = nn.Sequential()
-- nc x 32 x 32
netG:add(SpatialFullConvolution(nc, ngf*8, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*8)):add(nn.ReLU(true))
-- ngf*8 x 64 x 64
netG:add(SpatialFullConvolution(ngf*8, ngf*4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
-- ngf*4 x 128 x 128
netG:add(SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
-- ngf*2 x 256 x 256
netG:add(SpatialConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2, true))
-- ngf x 128 x 128
netG:add(SpatialConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- nc x 64 x 64

---- 
netG:apply(weights_init)

-- set network of Discriminator
local netD = nn.Sequential()
---- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
---- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
---- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
---- state size: 1
----
netD:apply(weights_init)

-- set criterion
local criterion = nn.MSECriterion()
-- criterion.sizeAverage = false
---------------------------------------------------------------------------
optimStateG = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local inputG = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
local inputD = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local real_none = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local label = torch.Tensor(opt.batchSize)
local real_label = 1
local fake_label = 0
----------------------------------------------------------------------------
-- to use GPU
require 'cunn'
cutorch.setDevice(1) -- use GPU
input = input:cuda();
inputG = inputG:cuda(); inputD = inputD:cuda(); label = label:cuda()
real_none = real_none:cuda()

if pcall(require, 'cudnn') then
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.convert(netG, cudnn)
    cudnn.convert(netD, cudnn)
end
netD:cuda();           netG:cuda();           criterion:cuda()
----------------------------------------------------------------------------
-- calPSNR function
-- function calPSNR(img1, img2)
--     local MSE = (((img1[{ {1}, {}, {}, {} }] - img2[{ {1}, {}, {}, {} }]):pow(2)):sum()) / (img2:size(2)*img2:size(3)*img2:size(4))
--     if MSE > 0 then
--         PSNR = 10 * torch.log(1*1/MSE) / torch.log(10)
--     else
--         PSNR = 99
--     end
--     return PSNR
-- end

function calMSE(img1, img2)
    return (((img1[{ {1}, {}, {}, {} }] - img2[{ {1}, {}, {}, {} }]):pow(2)):sum()) / (4*img2:size(2)*img2:size(3)*img2:size(4))
end

function calMSENEW(img1, img2)
    return (((img1[{ {1}, {}, {}, {} }] - img2[{ {}, {}, {} }]):pow(2)):sum()) / (4*img1:size(2)*img1:size(3)*img1:size(4))
end
----------------------------------------------------------------------------

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

local errVal_PSNR = torch.Tensor(opt.batchSize)

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    gradParametersD:zero()

    -- get real_none from dataset
    data_tm:reset(); data_tm:resume()
    real_none = data:getBatch()
    data_tm:stop()

    -- train with original
    inputD:copy(real_none)
    local output = netD:forward(inputD)
    label:fill(0)
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(inputD, df_do)

    -- generate real_reduced
    local real_reduced = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
    for i = 1, opt.fineSize/2 do
        for j = 1, opt.fineSize/2 do
            real_reduced[{ {}, {}, {i}, {j} }] = (real_none[{ {}, {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {}, {2*i}, {2*j-1} }] + real_none[{ {}, {}, {2*i-1}, {2*j} }] + real_none[{ {}, {}, {2*i}, {2*j} }]) / 4
        end
    end

    -- generate fake_none
    inputG:copy(real_reduced)
    local fake_none = netG:forward(inputG)

    -- calculate PSNR
    for i = 1, opt.batchSize do
        errVal_PSNR[i] = calMSE(real_none[{ {i}, {}, {}, {} }], fake_none[{ {i}, {}, {}, {} }]:float())
    end

    -- train with fake
    inputD:copy(fake_none)
    output = netD:forward(inputD) -- output: output_fake

    label:copy(errVal_PSNR)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(inputD, df_do)

    errD = errD_real + errD_fake

    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    gradParametersG:zero()

    --[[ the three lines below were already executed in fDx, so save computation
    noise:uniform(-1, 1) -- regenerate random noise
    local fake = netG:forward(noise)
    input:copy(fake) ]]--

    label:fill(0)
    local output = netD.output -- output: output_fake
    errG = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    local df_dg = netD:updateGradInput(inputD, df_do) -- inputD: fake_none

    netG:backward(inputG, df_dg) -- inputG: real_reduced
    return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optim.adam(fDx, parametersD, optimStateD)

        -- (2) Update G network: maximize log(D(G(z)))
        optim.adam(fGx, parametersG, optimStateG)

        -- logging
        if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.16f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
        end
    end
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    -- paths.mkdir('checkpoints')
    -- torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
    -- torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

local real_none_sample = torch.Tensor(nc, opt.fineSize, opt.fineSize)
real_none_sample = data:getBatch()[1]
image.save('real_none_sample.png', image.toDisplayTensor(real_none_sample))

local real_reduced_sample = torch.Tensor(nc, opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_sample[{{}, {i}, {j} }] = (real_none_sample[{{}, {2*i-1}, {2*j-1} }] + real_none_sample[{{}, {2*i}, {2*j-1} }] + real_none_sample[{{}, {2*i-1}, {2*j} }] + real_none_sample[{{}, {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_sample.png', image.toDisplayTensor(real_reduced_sample))

local inputG_sample = torch.Tensor(1, nc, opt.fineSize/2, opt.fineSize/2)
inputG_sample[{{1}, {}, {}, {}}] = real_reduced_sample[{ {}, {}, {}}]
inputG_sample = inputG_sample:cuda()

local fake_none_sample = netG:forward(inputG_sample)

-- print('MSE: ' .. calMSENEW(inputG_sample:float(), fake_none_sample:float()))

image.save('fake_none_sample.png', image.toDisplayTensor(fake_none_sample))