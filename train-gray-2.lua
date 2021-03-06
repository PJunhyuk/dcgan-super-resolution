require 'torch'
require 'image'
require 'nn'
require 'optim'

-- set default option
opt = {
    batchSize = 100,
    fineSize = 64,
    ngf = 64,               -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    niter = 1,             -- #  of iter at starting learning rate
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    ntrain = 10000,     -- #  of examples per epoch. math.huge for full dataset
    name = 'dcgan-sr-test-1',
}

local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_num = 1
local file_set_num = 0

-- simplify library of nn
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling

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

local nc = 1
local ndf = opt.ndf
local ngf = opt.ngf

-- set network of Generator
local netG = nn.Sequential()
-- -- nc x 32 x 32
-- netG:add(nn.SpatialUpSamplingNearest(2))
-- netG:add(SpatialBatchNormalization(nc)):add(nn.ReLU(true))
-- -- nc x 64 x 64
-- netG:add(SpatialFullConvolution(nc, ngf*4, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
-- -- ngf*4 x 128 x 128
-- netG:add(SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
-- -- ngf*2 x 256 x 256
-- netG:add(SpatialConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf)):add(nn.LeakyReLU(0.2, true))
-- -- ngf x 128 x 128
-- netG:add(SpatialConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
-- netG:add(nn.Sigmoid())
-- -- nc x 64 x 64

-- nc x 32 x 32
netG:add(nn.SpatialUpSamplingNearest(2))
-- nc x 64 x 64
netG:add(nn.SpatialUpSamplingNearest(2))
-- nc x 128 x 128
netG:add(nn.SpatialUpSamplingNearest(2))
-- nc x 256 x 256
netG:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- ngf x 128 x 128
netG:add(SpatialConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Sigmoid())
-- nc x 64 x 64

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

netD:apply(weights_init)

----------------------------------------------------
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
local input = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)
local inputG = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
local inputD = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local real_none = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)
local real_color = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
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
function calPSNR(img1, img2)
    local MSE = (((img1 - img2):pow(2)):sum()) / (img1:size(1) * img1:size(2))
    print(('MSE: %.4f'):format(MSE))
    if MSE > 0 then
        PSNR = 10 * torch.log(1*1/MSE) / torch.log(10)
    else
        PSNR = 99
    end
    return PSNR
end

function calMSE(img1, img2)
    return (((img1 - img2):pow(2)):sum()) / (img2:size(3) * img2:size(4))
end

----------------------------------------------------------------------------

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

local errVal_MSE = torch.Tensor(opt.batchSize)

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    gradParametersD:zero()

    data_tm:reset(); data_tm:resume()
    data_tm:stop()

    for i = 1, opt.batchSize do
        file_num = file_set_num * opt.batchSize + i
        
        local file_name

        if file_num < 10 then
            file_name = file_name_route .. '00000' .. tostring(file_num) .. '.jpg'
        elseif file_num < 100 then
            file_name = file_name_route .. '0000' .. tostring(file_num) .. '.jpg'
        elseif file_num < 1000 then
            file_name = file_name_route .. '000' .. tostring(file_num) .. '.jpg'
        elseif file_num < 10000 then
            file_name = file_name_route .. '00' .. tostring(file_num) .. '.jpg'
        elseif file_num < 100000 then
            file_name = file_name_route .. '0' .. tostring(file_num) .. '.jpg'
        else
            file_name = file_name_route .. tostring(file_num) .. '.jpg'
        end

        local image_input_gray = image.load(file_name, 1, 'float')
        image_input_gray = image.scale(image_input_gray, opt.fineSize, opt.fineSize)

        real_none[{ {i}, {}, {} }] = image_input_gray[{ {}, {} }]

        inputD[{ {}, {1}, {}, {} }] = real_none[{ {}, {}, {} }]
    end

    file_set_num = file_set_num + 1

    -- train with real
    local outputD = netD:forward(inputD) -- inputD: real_none / outputD: output_real
    label:fill(0)
    local errD_real = 100000 * criterion:forward(outputD, label) -- output_real & 0
    local df_do = criterion:backward(outputD, label)
    netD:backward(inputD, df_do)

    -- generate real_reduced
    local real_reduced = torch.Tensor(opt.batchSize, opt.fineSize/2, opt.fineSize/2)
    real_reduced = real_reduced:cuda()
    for i = 1, opt.fineSize/2 do
        for j = 1, opt.fineSize/2 do
            real_reduced[{ {}, {i}, {j} }] = (real_none[{ {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {2*i}, {2*j-1} }] + real_none[{ {}, {2*i-1}, {2*j} }] + real_none[{ {}, {2*i}, {2*j} }]) / 4
        end
    end

    -- generate fake_none
    inputG[{ {}, {1}, {}, {} }] = real_reduced[{ {}, {}, {} }]

    local fake_none = netG:forward(inputG)

    -- calculate MSE
    for i = 1, opt.batchSize do
        errVal_MSE[i] = calMSE(real_none[{ {i}, {}, {} }]:float(), fake_none[{ {i}, {}, {} }]:float())
    end

    print(('errVal_MSE-max: %.8f  errVal_MSE-min: %.8f'):format(errVal_MSE:max(), errVal_MSE:min()))

    -- train with fake
    inputD[{ {}, {1}, {}, {} }] = fake_none[{ {}, {}, {} }]
    local outputD = netD:forward(inputD) -- inputD: fake_none / outputD: output_fake
    label:copy(errVal_MSE)
    local errD_fake = criterion:forward(outputD, label) -- output_fake & errVal_MSE
    local df_do = criterion:backward(outputD, label)
    netD:backward(inputD, df_do)

    print(('errD_real: %.8f  errD_fake: %.8f'):format(errD_real, errD_fake))

    -- conclusion
    errD = errD_real + errD_fake
    -- print('errD'); print(errD)
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    gradParametersG:zero()

    label:fill(0.001)
    local outputD = netD.output -- outputD: output_fake
    errG = criterion:forward(outputD, label) -- output_fake & 0
    local df_do = criterion:backward(outputD, label)
    local df_dg = netD:updateGradInput(inputD, df_do) -- inputD: fake_none
    netG:backward(inputG, df_dg) -- inputG: real_reduced

    return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, opt.ntrain, opt.batchSize do
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
                 math.floor(opt.ntrain / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
        end
    end
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

--------------------------------------------

local real_none_train = image.load('/CelebA/Img/img_align_celeba/Img/000001.jpg', 1, 'float')
real_none_train = image.scale(real_none_train, opt.fineSize, opt.fineSize)

image.save('real_none_train.jpg', image.toDisplayTensor(real_none_train))

print(('real_none_train-max: %.8f  real_none_train-min: %.8f'):format(real_none_train:max(), real_none_train:min()))
print(('real_none_train-sum: %.8f  real_none_train-std: %.8f'):format(real_none_train:sum(), real_none_train:std()))

local real_reduced_train = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_train[{ {i}, {j} }] = (real_none_train[{ {2*i-1}, {2*j-1} }] + real_none_train[{ {2*i}, {2*j-1} }] + real_none_train[{ {2*i-1}, {2*j} }] + real_none_train[{ {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_train.jpg', image.toDisplayTensor(real_reduced_train))

print(('real_reduced_train-max: %.8f  real_reduced_train-min: %.8f'):format(real_reduced_train:max(), real_reduced_train:min()))
print(('real_reduced_train-sum: %.8f  real_reduced_train-std: %.8f'):format(real_reduced_train:sum(), real_reduced_train:std()))

local real_bilinear_train = torch.Tensor(opt.fineSize, opt.fineSize)

real_bilinear_train = image.scale(real_reduced_train, opt.fineSize, opt.fineSize, bilinear)

real_bilinear_train = real_bilinear_train:float()
image.save('real_bilinear_train.jpg', image.toDisplayTensor(real_bilinear_train))

print(('real_bilinear_train-max: %.8f  real_bilinear_train-min: %.8f'):format(real_bilinear_train:max(), real_bilinear_train:min()))
print(('real_bilinear_train-sum: %.8f  real_bilinear_train-std: %.8f'):format(real_bilinear_train:sum(), real_bilinear_train:std()))

print(('PSNR btwn real_none_train & real_bilinear_train: %.4f'):format(calPSNR(real_none_train, real_bilinear_train)))

local inputG_train = torch.Tensor(1, 1, opt.fineSize/2, opt.fineSize/2)
inputG_train[{{1}, {1}, {}, {}}] = real_reduced_train[{ {}, {}}]
inputG_train = inputG_train:cuda()
local fake_none_train_temp = netG:forward(inputG_train)

local fake_none_train = torch.Tensor(opt.fineSize, opt.fineSize)
fake_none_train[{ {}, {} }] = fake_none_train_temp[{ {1}, {1}, {}, {} }]:float()

fake_none_train = fake_none_train:float()
image.save('fake_none_train.jpg', image.toDisplayTensor(fake_none_train))

print(('fake_none_train-max: %.8f  fake_none_train-min: %.8f'):format(fake_none_train:max(), fake_none_train:min()))
print(('fake_none_train-sum: %.8f  fake_none_train-std: %.8f'):format(fake_none_train:sum(), fake_none_train:std()))

print(('PSNR btwn real_none_train & fake_none_train: %.4f'):format(calPSNR(real_none_train, fake_none_train)))

-----------------------------------------------

local real_none_test = image.load('/CelebA/Img/img_align_celeba/Img/100001.jpg', 1, 'float')
real_none_test = image.scale(real_none_test, opt.fineSize, opt.fineSize)

image.save('real_none_test.jpg', image.toDisplayTensor(real_none_test))

print(('real_none_test-max: %.8f  real_none_test-min: %.8f'):format(real_none_test:max(), real_none_test:min()))
print(('real_none_test-sum: %.8f  real_none_test-std: %.8f'):format(real_none_test:sum(), real_none_test:std()))

local real_reduced_test = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_test[{ {i}, {j} }] = (real_none_test[{ {2*i-1}, {2*j-1} }] + real_none_test[{ {2*i}, {2*j-1} }] + real_none_test[{ {2*i-1}, {2*j} }] + real_none_test[{ {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_test.jpg', image.toDisplayTensor(real_reduced_test))

print(('real_reduced_test-max: %.8f  real_reduced_test-min: %.8f'):format(real_reduced_test:max(), real_reduced_test:min()))
print(('real_reduced_test-sum: %.8f  real_reduced_test-std: %.8f'):format(real_reduced_test:sum(), real_reduced_test:std()))

local real_bilinear_test = torch.Tensor(opt.fineSize, opt.fineSize)

real_bilinear_test = image.scale(real_reduced_test, opt.fineSize, opt.fineSize, bilinear)

real_bilinear_test = real_bilinear_test:float()
image.save('real_bilinear_test.jpg', image.toDisplayTensor(real_bilinear_test))

print(('real_bilinear_test-max: %.8f  real_bilinear_test-min: %.8f'):format(real_bilinear_test:max(), real_bilinear_test:min()))
print(('real_bilinear_test-sum: %.8f  real_bilinear_test-std: %.8f'):format(real_bilinear_test:sum(), real_bilinear_test:std()))

print(('PSNR btwn real_none_test & real_bilinear_test: %.4f'):format(calPSNR(real_none_test, real_bilinear_test)))

local inputG_test = torch.Tensor(1, 1, opt.fineSize/2, opt.fineSize/2)
inputG_test[{{1}, {1}, {}, {}}] = real_reduced_test[{ {}, {}}]
inputG_test = inputG_test:cuda()
local fake_none_test_temp = netG:forward(inputG_test)

local fake_none_test = torch.Tensor(opt.fineSize, opt.fineSize)
fake_none_test[{ {}, {} }] = fake_none_test_temp[{ {1}, {1}, {}, {} }]:float()

fake_none_test = fake_none_test:float()
image.save('fake_none_test.jpg', image.toDisplayTensor(fake_none_test))

print(('fake_none_test-max: %.8f  fake_none_test-min: %.8f'):format(fake_none_test:max(), fake_none_test:min()))
print(('fake_none_test-sum: %.8f  fake_none_test-std: %.8f'):format(fake_none_test:sum(), fake_none_test:std()))

print(('PSNR btwn real_none_test & fake_none_test: %.4f'):format(calPSNR(real_none_test, fake_none_test)))
