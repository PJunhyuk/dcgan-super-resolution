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
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 1,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'dcgan-sr-test-1',
   noise = 'normal',       -- uniform / normal
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
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

-- simplify library of nn
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

-- set network of Generator
local netG = nn.Sequential()
netG:add(SpatialFullConvolution(nc, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
---- 
netG:apply(weights_init)

print('netG complete!!!')

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

print('netD complete!!!')

-- set criterion
local criterion = nn.BCECriterion()
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
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
-- convert elements to gpu version
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

print('checkpoint 2 complete!!!')

local function RGBMSE(x1, x2)
   x1 = iproc.float2byte(x1):float()
   x2 = iproc.float2byte(x2):float()
   return (x1 - x2):pow(2):mean()
end
local function CHMSE(x1, x2, ch)
   x1 = iproc.float2byte(x1):float()
   x2 = iproc.float2byte(x2):float()
   return (x1[ch] - x2[ch]):pow(2):mean()
end
local function YMSE(x1, x2)
   if opt.range_bug == 1 then
      local x1_2 = rgb2y_matlab(x1)
      local x2_2 = rgb2y_matlab(x2)
      return (x1_2 - x2_2):pow(2):mean()
   else
      local x1_2 = image.rgb2y(x1):mul(255.0)
      local x2_2 = image.rgb2y(x2):mul(255.0)
      return (x1_2 - x2_2):pow(2):mean()
   end
end
local function MSE(x1, x2, color)
   if color == "y" then
      return YMSE(x1, x2)
   elseif color == "r" then
      return CHMSE(x1, x2, 1)
   elseif color == "g" then
      return CHMSE(x1, x2, 2)
   elseif color == "b" then
      return CHMSE(x1, x2, 3)
   else
      return RGBMSE(x1, x2)
   end
end
local function PSNR(x1, x2, color)
   local mse = math.max(MSE(x1, x2, color), 1)
   return 10 * math.log10((255.0 * 255.0) / mse)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    print('fDx cp 1')
    gradParametersD:zero()

    -- train with real
    data_tm:reset(); data_tm:resume()

    local real_none = data:getBatch()

    -- generate real_reduced
    local real_reduced = torch.Tensor(opt.batchSize, 3, opt.fineSize/2, opt.fineSize/2)
    for i = 1, opt.fineSize/2 do
        for j = 1, opt.fineSize/2 do
            real_reduced[{ {}, {}, {i}, {j} }] = (real_none[{ {}, {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {}, {2*i}, {2*j-1} }] + real_none[{ {}, {}, {2*i-1}, {2*j} }] + real_none[{ {}, {}, {2*i}, {2*j} }]) / 4
        end
    end

    data_tm:stop()

    -- input:copy(real_none)
    -- label:fill(real_label)

    print('fDx cp 2')
    local fake_none = netG:forward(real_none)
    print('fDx cp 2.1')

    local errVal_fake = netD:forward(fake_none)
    print('fDx cp 2.2')

    local errVal_PSNR = torch.Tensor(opt.batchSize)
    print('fDx cp 2.3')

    for i = 1, opt.batchSize do
        errVal_PSNR[{ {i} }] = PSNR(real_none[{ {i}, {}, {}, {}}], fake_none[{ {i}, {}, {}, {}}], 'rgb')
    end
    print('fDx cp 5')

    local errD = criterion:forward(errVal_fake, errVal_PSNR)
    print('fDx cp 6')
    local df_do = criterion:backward(errVal_fake, errVal_PSNR)
    print('fDx cp 3')
    netD:backward(fake_none, df_do)

    print('fDx cp 4')
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

print('Lets train!!!')

-- train
for epoch = 1, opt.niter do
    print('cp#206!!!')
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optim.adam(fDx, parametersD, optimStateD)
        print('cp#212!!!')

        -- (2) Update G network: maximize log(D(G(z)))
        optim.adam(fGx, parametersG, optimStateG)

        -- logging
        if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
--    paths.mkdir('checkpoints')
--    torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
--    torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end


local images = netG:forward(noise)
print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. '.png')