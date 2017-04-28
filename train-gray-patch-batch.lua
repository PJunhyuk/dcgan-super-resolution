require 'torch'
require 'image'
require 'nn'
require 'optim'

local total_tm = torch.Timer()

-- set default option
opt = {
    batchSize = 20,
    fineSize = 64,
    ngf = 16,               -- #  of gen filters in first conv layer
    ndf = 64,               -- #  of discrim filters in first conv layer
    niter = 1,             -- #  of iter at starting learning rate
    lr = 0.0002,            -- initial learning rate for adam
    beta1 = 0.5,            -- momentum term of adam
    ntrain = 10000,     -- #  of examples per epoch. math.huge for full dataset
    patchSize = 8,
}

local patchNumber = (opt.fineSize / opt.patchSize) * (opt.fineSize / opt.patchSize)

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_set_num = 0
local file_num = 1

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
-- nc x 4
netG:add(nn.SpatialUpSamplingNearest(2))
-- nc x 8
netG:add(SpatialFullConvolution(nc, ngf*4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
-- ngf*4 x 16
netG:add(SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
-- ngf*2 x 32
netG:add(SpatialFullConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- ngf x 64
netG:add(SpatialConvolution(ngf, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
-- ngf*2 x 32
netG:add(SpatialConvolution(ngf*2, ngf*4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
-- ngf*4 x 16
netG:add(SpatialConvolution(ngf*4, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Sigmoid())
-- nc x 8
netG:apply(weights_init)

-- set network of Discriminator
-- local netD = nn.Sequential()
-- ---- input is (nc) x 8 x 8
-- netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
-- netD:add(nn.LeakyReLU(0.2, true))
-- ---- state size: (ndf) x 4 x 4
-- netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
-- netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- ---- state size: (ndf*2) x 2 x 2
-- netD:add(SpatialConvolution(ndf * 2, 1, 4, 4, 2, 2, 1, 1))
-- netD:add(nn.Sigmoid())
-- ---- state size: 1 x 1 x 1
-- netD:add(nn.View(1):setNumInputDims(3))
-- ---- state size: 1
-- netD:apply(weights_init)

-- set network of Discriminator
local netD = nn.Sequential()
---- input is (nc) x 8 x 8
netD:add(SpatialConvolution(nc, ndf, 3, 3, 1, 1, 0, 0))
netD:add(nn.LeakyReLU(0.2, true))
---- state size: (ndf) x 6 x 6
netD:add(SpatialConvolution(ndf, ndf * 2, 3, 3, 1, 1, 0, 0))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*2) x 4 x 4
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 3, 3, 1, 1, 0, 0))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*4) x 2 x 2
netD:add(SpatialConvolution(ndf * 4, 1, 2, 2))
netD:add(nn.Sigmoid())
---- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
---- state size: 1
netD:apply(weights_init)

----------------------------------------------------
-- set criterion
local criterion = nn.BCECriterion()
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
local input = torch.Tensor(opt.batchSize * patchNumber, opt.patchSize, opt.patchSize)
local inputG = torch.Tensor(opt.batchSize * patchNumber, nc, opt.patchSize/2, opt.patchSize/2)
local inputD = torch.Tensor(opt.batchSize * patchNumber, nc, opt.patchSize, opt.patchSize)
local real_none = torch.Tensor(opt.batchSize * patchNumber, opt.patchSize, opt.patchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local label = torch.Tensor(opt.batchSize * patchNumber)
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
    if MSE > 0 then
        PSNR = 10 * torch.log(1*1/MSE) / torch.log(10)
    else
        PSNR = 99
    end
    return PSNR
end

----------------------------------------------------------------------------
-- Calculate SSIM
-- Reference: https://github.com/coupriec/VideoPredictionICLR2016
function calSSIM(img1, img2)
--[[
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error visibility to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 4, pp.600-612,
%Apr. 2004.
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output:     mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.]]

    img1_temp = torch.Tensor(opt.fineSize, opt.fineSize)
    img2_temp = torch.Tensor(opt.fineSize, opt.fineSize)

    img1_temp[{ {}, {} }] = img1[{ {}, {} }]
    img2_temp[{ {}, {} }] = img2[{ {}, {} }]

    img1_temp = img1_temp:float()
    img2_temp = img2_temp:float()

    -- place images between 0 and 255.
    img1_temp:add(1):div(2):mul(255)
    img2_temp:add(1):div(2):mul(255)

    local K1 = 0.01;
    local K2 = 0.03;
    local L = 255;

    local C1 = (K1*L)^2;
    local C2 = (K2*L)^2;
    local window = image.gaussian(11, 1.5/11,0.0708);

    local window = window:div(torch.sum(window));
    window = window:float()

    local mu1 = image.convolve(img1_temp, window, 'full')
    local mu2 = image.convolve(img2_temp, window, 'full')

    local mu1_sq = torch.cmul(mu1,mu1);
    local mu2_sq = torch.cmul(mu2,mu2);
    local mu1_mu2 = torch.cmul(mu1,mu2);

    local sigma1_sq = image.convolve(torch.cmul(img1_temp,img1_temp),window,'full')-mu1_sq
    local sigma2_sq = image.convolve(torch.cmul(img2_temp,img2_temp),window,'full')-mu2_sq
    local sigma12 =  image.convolve(torch.cmul(img1_temp,img2_temp),window,'full')-mu1_mu2

    local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)), torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
    local mssim = torch.mean(ssim_map);

    return mssim
end

----------------------------------------------------------------------------

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    gradParametersD:zero()

    data_tm:reset(); data_tm:resume()
    data_tm:stop()

    print('file_set_num: ' .. file_set_num)

    for k = 1, opt.batchSize do
        file_num = file_set_num * opt.batchSize + k

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

        for i = 1, patchNumber do
            for a = 1, opt.patchSize do
                for b = 1, opt.patchSize do
                    real_none[{ {(k-1) * patchNumber + i}, {a}, {b} }] = image_input_gray[{ { math.floor((i-1) / opt.patchSize) * opt.patchSize + a }, { (i-1 - math.floor((i-1) / opt.patchSize) * opt.patchSize) * opt.patchSize + b } }]
                end
            end
        end

        inputD[{ {}, {1}, {}, {} }] = real_none[{ {}, {}, {} }]
    end

    file_set_num = file_set_num + 1

    -- train with real
    local outputD = netD:forward(inputD) -- inputD: real_none / outputD: output_real
    label:fill(1)
    local errD_real = criterion:forward(outputD, label) -- output_real & 1
    local df_do = criterion:backward(outputD, label)
    netD:backward(inputD, df_do)

    -- generate real_reduced
    local real_reduced = torch.Tensor(opt.batchSize * patchNumber, opt.patchSize/2, opt.patchSize/2)
    real_reduced = real_reduced:cuda()
    for i = 1, opt.patchSize/2 do
        for j = 1, opt.patchSize/2 do
            real_reduced[{ {}, {i}, {j} }] = (real_none[{ {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {2*i}, {2*j-1} }] + real_none[{ {}, {2*i-1}, {2*j} }] + real_none[{ {}, {2*i}, {2*j} }]) / 4
        end
    end

    -- generate fake_none
    inputG[{ {}, {1}, {}, {} }] = real_reduced[{ {}, {}, {} }]

    local fake_none = netG:forward(inputG) -- inputG: real_reduced

    -- train with fake
    inputD[{ {}, {1}, {}, {} }] = fake_none[{ {}, {}, {} }]
    local outputD = netD:forward(inputD) -- inputD: fake_none / outputD: output_fake
    label:fill(0)
    local errD_fake = criterion:forward(outputD, label) -- output_fake & 0
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

    label:fill(1)
    local outputD = netD.output -- outputD: output_fake
    errG = criterion:forward(outputD, label) -- output_fake & 1
    local df_do = criterion:backward(outputD, label)
    local df_dg = netD:updateGradInput(inputD, df_do) -- inputD: fake_none
    netG:backward(inputG, df_dg) -- inputG: real_reduced

    return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
    epoch_tm:reset()
    file_set_num = 0
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
                 epoch, ((i-1) / opt.batchSize) + 1,
                 math.floor(opt.ntrain / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
        end
    end

   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   netG:clearState()
   netD:clearState()
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()

    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

------------------------------------------
-- Calculate Performance(Avrg. PSNR) of Train-set
local rn_rb_PSNR_average = 0
local rn_fn_PSNR_average = 0
local rn_rb_SSIM_average = 0
local rn_fn_SSIM_average = 0

local real_none_full = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)

for file_set_num = 0, 500/opt.batchSize - 1 do

    for k = 1, opt.batchSize do
        file_num = file_set_num * opt.batchSize + k

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

        for i = 1, patchNumber do
            for a = 1, opt.patchSize do
                for b = 1, opt.patchSize do
                    real_none[{ {(k-1) * patchNumber + i}, {a}, {b} }] = image_input_gray[{ { math.floor((i-1) / opt.patchSize) * opt.patchSize + a }, { (i-1 - math.floor((i-1) / opt.patchSize) * opt.patchSize) * opt.patchSize + b } }]
                end
            end
        end

        inputD[{ {}, {1}, {}, {} }] = real_none[{ {}, {}, {} }]
        
        real_none_full[{ {k}, {}, {} }] = image_input_gray[{ {}, {} }]
    end

    -- generate real_reduced
    local real_reduced = torch.Tensor(opt.batchSize, opt.fineSize/2, opt.fineSize/2)
    real_reduced = real_reduced:cuda()
    for i = 1, opt.fineSize/2 do
        for j = 1, opt.fineSize/2 do
            real_reduced[{ {}, {i}, {j} }] = (real_none_full[{ {}, {2*i-1}, {2*j-1} }] + real_none_full[{ {}, {2*i}, {2*j-1} }] + real_none_full[{ {}, {2*i-1}, {2*j} }] + real_none_full[{ {}, {2*i}, {2*j} }]) / 4
        end
    end

    -- generate real_bilinear
    local real_bilinear = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)
    local real_bilinear_temp = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
    for i = 1, opt.batchSize do
        real_bilinear_temp[{ {}, {} }] = (real_reduced:float())[i]
        real_bilinear[i] = image.scale(real_bilinear_temp, opt.fineSize, opt.fineSize, bilinear)
    end

    -- generate real_reduced_patch
    local real_reduced_patch = torch.Tensor(opt.batchSize * patchNumber, opt.patchSize/2, opt.patchSize/2)
    real_reduced_patch = real_reduced_patch:cuda()
    for i = 1, opt.patchSize/2 do
        for j = 1, opt.patchSize/2 do
            real_reduced_patch[{ {}, {i}, {j} }] = (real_none[{ {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {2*i}, {2*j-1} }] + real_none[{ {}, {2*i-1}, {2*j} }] + real_none[{ {}, {2*i}, {2*j} }]) / 4
        end
    end

    -- generate fake_none
    inputG[{ {}, {1}, {}, {} }] = real_reduced_patch[{ {}, {}, {} }]
    local fake_none = netG:forward(inputG) -- inputG: real_reduced

    -- generate fake_none_full
    local fake_none_full = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)
    fake_none = fake_none:float()
    fake_none_full = fake_none_full:float()
    for k = 1, opt.batchSize do
        for i = 1, patchNumber do
            for a = 1, opt.patchSize do
                for b = 1, opt.patchSize do
                    fake_none_full[{ {k}, { math.floor((i-1) / opt.patchSize) * opt.patchSize + a }, { (i-1 - math.floor((i-1) / opt.patchSize) * opt.patchSize) * opt.patchSize + b } }] = fake_none[{ {(k-1) * opt.batchSize + i}, {1}, {a}, {b} }]
                end
            end
        end
    end
    fake_none_train = fake_none_train:float()

    -- calculate PSNR
    local rn_rb_PSNR = torch.Tensor(opt.batchSize)
    for i = 1, opt.batchSize do
        rn_rb_PSNR[i] = calPSNR(real_none_full[i]:float(), real_bilinear[i]:float())
    end
    rn_rb_PSNR_average = rn_rb_PSNR_average + rn_rb_PSNR:sum()

    -- calculate SSIM
    local rn_rb_SSIM = torch.Tensor(opt.batchSize)
    for i = 1, opt.batchSize do
        rn_rb_SSIM[i] = calSSIM(real_none_full[i]:float(), real_bilinear[i]:float())
    end
    rn_rb_SSIM_average = rn_rb_SSIM_average + rn_rb_SSIM:sum()

    -- calculate PSNR
    local rn_fn_PSNR = torch.Tensor(opt.batchSize)
    for i = 1, opt.batchSize do
        rn_fn_PSNR[i] = calPSNR(real_none_full[i]:float(), fake_none_full[i]:float())
    end
    rn_fn_PSNR_average = rn_fn_PSNR_average + rn_fn_PSNR:sum()

    -- calculate SSIM
    local rn_fn_SSIM = torch.Tensor(opt.batchSize)
    for i = 1, opt.batchSize do
        rn_fn_SSIM[i] = calSSIM(real_none_full[i]:float(), fake_none_full[i]:float())
    end
    rn_fn_SSIM_average = rn_fn_SSIM_average + rn_fn_SSIM:sum()
end

rn_rb_PSNR_average = rn_rb_PSNR_average / opt.ntrain
rn_fn_PSNR_average = rn_fn_PSNR_average / opt.ntrain

rn_rb_SSIM_average = rn_rb_SSIM_average / opt.ntrain
rn_fn_SSIM_average = rn_fn_SSIM_average / opt.ntrain

print(('[Train-set] PSNR btwn real_none & real_bilinear: %.8f, train-Size: %d'):format(rn_rb_PSNR_average, opt.ntrain))
print(('[Train-set] PSNR btwn real_none & fake_none: %.8f, train-Size: %d'):format(rn_fn_PSNR_average, opt.ntrain))

print(('[Train-set] SSIM btwn real_none & real_bilinear: %.8f, train-Size: %d'):format(rn_rb_SSIM_average, opt.ntrain))
print(('[Train-set] SSIM btwn real_none & fake_none: %.8f, train-Size: %d'):format(rn_fn_SSIM_average, opt.ntrain))

-- --------------------------------------------
-- -- Calculate Performance(Avrg. PSNR) of Test-set
-- for file_set_num = 2000, 2020 do -- 200001 ~ 202100
--     for i = 1, opt.batchSize do
--         file_num = file_set_num * opt.batchSize + i
        
--         local file_name

--         if file_num < 10 then
--             file_name = file_name_route .. '00000' .. tostring(file_num) .. '.jpg'
--         elseif file_num < 100 then
--             file_name = file_name_route .. '0000' .. tostring(file_num) .. '.jpg'
--         elseif file_num < 1000 then
--             file_name = file_name_route .. '000' .. tostring(file_num) .. '.jpg'
--         elseif file_num < 10000 then
--             file_name = file_name_route .. '00' .. tostring(file_num) .. '.jpg'
--         elseif file_num < 100000 then
--             file_name = file_name_route .. '0' .. tostring(file_num) .. '.jpg'
--         else
--             file_name = file_name_route .. tostring(file_num) .. '.jpg'
--         end

--         local image_input_gray = image.load(file_name, 1, 'float')
--         image_input_gray = image.scale(image_input_gray, opt.fineSize, opt.fineSize)

--         real_none[{ {i}, {}, {} }] = image_input_gray[{ {}, {} }]
--     end

--     -- generate real_reduced
--     local real_reduced = torch.Tensor(opt.batchSize, opt.fineSize/2, opt.fineSize/2)
--     real_reduced = real_reduced:cuda()
--     for i = 1, opt.fineSize/2 do
--         for j = 1, opt.fineSize/2 do
--             real_reduced[{ {}, {i}, {j} }] = (real_none[{ {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {2*i}, {2*j-1} }] + real_none[{ {}, {2*i-1}, {2*j} }] + real_none[{ {}, {2*i}, {2*j} }]) / 4
--         end
--     end

--     -- generate real_bilinear
--     local real_bilinear = torch.Tensor(opt.batchSize, opt.fineSize, opt.fineSize)
--     local real_bilinear_temp = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
--     for i = 1, opt.batchSize do
--         real_bilinear_temp[{ {}, {} }] = (real_reduced:float())[i]
--         real_bilinear[i] = image.scale(real_bilinear_temp, opt.fineSize, opt.fineSize, bilinear)
--     end

--     -- generate fake_none
--     inputG[{ {}, {1}, {}, {} }] = real_reduced[{ {}, {}, {} }]
--     local fake_none = netG:forward(inputG) -- inputG: real_reduced

--     -- calculate PSNR
--     local rn_rb_PSNR = torch.Tensor(opt.batchSize)
--     for i = 1, opt.batchSize do
--         rn_rb_PSNR[i] = calPSNR(real_none[i]:float(), real_bilinear[i]:float())
--     end
--     rn_rb_PSNR_average = rn_rb_PSNR_average + rn_rb_PSNR:sum()

--     -- calculate SSIM
--     local rn_rb_SSIM = torch.Tensor(opt.batchSize)
--     for i = 1, opt.batchSize do
--         rn_rb_SSIM[i] = calSSIM(real_none[i]:float(), real_bilinear[i]:float())
--     end
--     rn_rb_SSIM_average = rn_rb_SSIM_average + rn_rb_SSIM:sum()

--     -- calculate PSNR
--     local rn_fn_PSNR = torch.Tensor(opt.batchSize)
--     for i = 1, opt.batchSize do
--         rn_fn_PSNR[i] = calPSNR(real_none[i]:float(), fake_none[i]:float())
--     end
--     rn_fn_PSNR_average = rn_fn_PSNR_average + rn_fn_PSNR:sum()

--     -- calculate SSIM
--     local rn_fn_SSIM = torch.Tensor(opt.batchSize)
--     for i = 1, opt.batchSize do
--         rn_fn_SSIM[i] = calSSIM(real_none[i]:float(), fake_none[i]:float())
--     end
--     rn_fn_SSIM_average = rn_fn_SSIM_average + rn_fn_SSIM:sum()
-- end

-- rn_rb_PSNR_average = rn_rb_PSNR_average / 2100
-- rn_fn_PSNR_average = rn_fn_PSNR_average / 2100

-- rn_rb_SSIM_average = rn_rb_SSIM_average / 2100
-- rn_fn_SSIM_average = rn_fn_SSIM_average / 2100

-- print(('[Test-set] PSNR btwn real_none & real_bilinear: %.8f, train-Size: %d'):format(rn_rb_PSNR_average, 2100))
-- print(('[Test-set] PSNR btwn real_none & fake_none: %.8f, train-Size: %d'):format(rn_fn_PSNR_average, 2100))

-- print(('[Test-set] SSIM btwn real_none & real_bilinear: %.8f, train-Size: %d'):format(rn_rb_SSIM_average, 2100))
-- print(('[Test-set] SSIM btwn real_none & fake_none: %.8f, train-Size: %d'):format(rn_fn_SSIM_average, 2100))
--------------------------------------------

local real_none_train = image.load('/CelebA/Img/img_align_celeba/Img/000001.jpg', 1, 'float')
real_none_train = image.scale(real_none_train, opt.fineSize, opt.fineSize)
real_none_patch_train = torch.Tensor(patchNumber, opt.fineSize, opt.fineSize)

image.save('real_none_train.jpg', image.toDisplayTensor(real_none_train))

print(('real_none_train-max: %.8f  real_none_train-min: %.8f'):format(real_none_train:max(), real_none_train:min()))
print(('real_none_train-sum: %.8f  real_none_train-std: %.8f'):format(real_none_train:sum(), real_none_train:std()))

for i = 1, patchNumber do
    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            real_none_patch_train[{ {i}, {a}, {b} }] = real_none_train[{ { math.floor((i-1) / opt.patchSize) * opt.patchSize + a }, { (i-1 - math.floor((i-1) / opt.patchSize) * opt.patchSize) * opt.patchSize + b } }]
        end
    end
end

local real_reduced_patch_train = torch.Tensor(patchNumber, opt.patchSize/2, opt.patchSize/2)
for i = 1, opt.patchSize/2 do
    for j = 1, opt.patchSize/2 do
        real_reduced_patch_train[{ {}, {i}, {j} }] = (real_none_patch_train[{ {}, {2*i-1}, {2*j-1} }] + real_none_patch_train[{ {}, {2*i}, {2*j-1} }] + real_none_patch_train[{ {}, {2*i-1}, {2*j} }] + real_none_patch_train[{ {}, {2*i}, {2*j} }]) / 4
    end
end

local real_reduced_train = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_train[{ {i}, {j} }] = (real_none_train[{ {2*i-1}, {2*j-1} }] + real_none_train[{ {2*i}, {2*j-1} }] + real_none_train[{ {2*i-1}, {2*j} }] + real_none_train[{ {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_train.jpg', image.toDisplayTensor(real_reduced_train))

local real_bilinear_train = torch.Tensor(opt.fineSize, opt.fineSize)
real_bilinear_train = image.scale(real_reduced_train, opt.fineSize, opt.fineSize, bilinear)
real_bilinear_train = real_bilinear_train:float()
image.save('real_bilinear_train.jpg', image.toDisplayTensor(real_bilinear_train))

print(('PSNR btwn real_none_train & real_bilinear_train: %.4f'):format(calPSNR(real_none_train, real_bilinear_train)))
print(('SSIM btwn real_none_train & real_bilinear_train: %.4f'):format(calSSIM(real_none_train, real_bilinear_train)))

local inputG_train = torch.Tensor(patchNumber, 1, opt.patchSize/2, opt.patchSize/2)
inputG_train[{{}, {1}, {}, {}}] = real_reduced_patch_train[{ {}, {}, {}}]
inputG_train = inputG_train:cuda()
local fake_none_patch_train = netG:forward(inputG_train)
fake_none_patch_train = fake_none_patch_train:float()

local fake_none_train = torch.Tensor(opt.fineSize, opt.fineSize)
for i = 1, patchNumber do
    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            fake_none_train[{ { math.floor((i-1) / opt.patchSize) * opt.patchSize + a }, { (i-1 - math.floor((i-1) / opt.patchSize) * opt.patchSize) * opt.patchSize + b } }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
        end
    end
end
fake_none_train = fake_none_train:float()

print(('fake_none_train-max: %.8f  fake_none_train-min: %.8f'):format(fake_none_train:max(), fake_none_train:min()))
print(('fake_none_train-sum: %.8f  fake_none_train-std: %.8f'):format(fake_none_train:sum(), fake_none_train:std()))

print(('PSNR btwn real_none_train & fake_none_train: %.4f'):format(calPSNR(real_none_train, fake_none_train)))
print(('SSIM btwn real_none_train & fake_none_train: %.4f'):format(calSSIM(real_none_train, fake_none_train)))

image.save('fake_none_train.jpg', image.toDisplayTensor(fake_none_train))

-- -----------------------------------------------

-- local real_none_test = image.load('/CelebA/Img/img_align_celeba/Img/100001.jpg', 1, 'float')
-- real_none_test = image.scale(real_none_test, opt.fineSize, opt.fineSize)

-- image.save('real_none_test.jpg', image.toDisplayTensor(real_none_test))

-- print(('real_none_test-max: %.8f  real_none_test-min: %.8f'):format(real_none_test:max(), real_none_test:min()))
-- print(('real_none_test-sum: %.8f  real_none_test-std: %.8f'):format(real_none_test:sum(), real_none_test:std()))

-- local real_reduced_test = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
-- for i = 1, opt.fineSize/2 do
--     for j = 1, opt.fineSize/2 do
--         real_reduced_test[{ {i}, {j} }] = (real_none_test[{ {2*i-1}, {2*j-1} }] + real_none_test[{ {2*i}, {2*j-1} }] + real_none_test[{ {2*i-1}, {2*j} }] + real_none_test[{ {2*i}, {2*j} }]) / 4
--     end
-- end
-- image.save('real_reduced_test.jpg', image.toDisplayTensor(real_reduced_test))

-- local real_bilinear_test = torch.Tensor(opt.fineSize, opt.fineSize)

-- real_bilinear_test = image.scale(real_reduced_test, opt.fineSize, opt.fineSize, bilinear)

-- real_bilinear_test = real_bilinear_test:float()
-- image.save('real_bilinear_test.jpg', image.toDisplayTensor(real_bilinear_test))

-- print(('PSNR btwn real_none_test & real_bilinear_test: %.4f'):format(calPSNR(real_none_test, real_bilinear_test)))
-- print(('SSIM btwn real_none_test & real_bilinear_test: %.4f'):format(calSSIM(real_none_test, real_bilinear_test)))

-- local inputG_test = torch.Tensor(1, 1, opt.fineSize/2, opt.fineSize/2)
-- inputG_test[{{1}, {1}, {}, {}}] = real_reduced_test[{ {}, {}}]
-- inputG_test = inputG_test:cuda()
-- local fake_none_test_temp = netG:forward(inputG_test)

-- local fake_none_test = torch.Tensor(opt.fineSize, opt.fineSize)
-- fake_none_test[{ {}, {} }] = fake_none_test_temp[{ {1}, {1}, {}, {} }]:float()

-- fake_none_test = fake_none_test:float()

-- print(('fake_none_test-max: %.8f  fake_none_test-min: %.8f'):format(fake_none_test:max(), fake_none_test:min()))
-- print(('fake_none_test-sum: %.8f  fake_none_test-std: %.8f'):format(fake_none_test:sum(), fake_none_test:std()))

-- print(('PSNR btwn real_none_test & fake_none_test: %.4f'):format(calPSNR(real_none_test, fake_none_test)))
-- print(('SSIM btwn real_none_test & fake_none_test: %.4f'):format(calSSIM(real_none_test, fake_none_test)))

-- image.save('fake_none_test.jpg', image.toDisplayTensor(fake_none_test))

print(('Total time: %.3f'):format(total_tm:time().real))