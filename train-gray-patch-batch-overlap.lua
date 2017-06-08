---- 0. settings ------------------------------------------------------------------
---- 0-1. basic settings ----------------------------------------------------------
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
    overlap = 4,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local patchNumber = (opt.fineSize / opt.patchSize) * (opt.fineSize / opt.patchSize)

local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_set_num = 0
local file_num = 1

local nc = 1
local ndf = opt.ndf
local ngf = opt.ngf

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

-- simplify library of nn
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local SpatialMaxPooling = nn.SpatialMaxPooling
local SpatialAveragePooling = nn.SpatialAveragePooling

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

---- 0-2. set network of D&G ------------------------------------------------------
-- set network of Generator
local netG = nn.Sequential()
-- nc x 4
netG:add(nn.SpatialUpSamplingNearest(2))
-- netG:add(SpatialFullConvolution(nc, ngf*8, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf*8)):add(nn.ReLU(true))
-- ngf*8 x 8
-- nc x 8
netG:add(SpatialFullConvolution(nc, ngf*4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
-- netG:add(SpatialFullConvolution(ngf*8, ngf*4, 4, 4, 2, 2, 1, 1))
-- netG:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU(true))
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

---- 0-3. settings for train ------------------------------------------------------
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

-- set criterion
local criterion = nn.BCECriterion()
-- criterion.sizeAverage = false

optimStateG = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}
optimStateD = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

---- 0-4. define functions --------------------------------------------------------
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

---- 1. set fDx and fGx ---------------------------------------------------------
---- 1-1. set fDx ---------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    ---- 1-1-1. settings --------------------------------------------------------
    gradParametersD:zero()

    data_tm:reset(); data_tm:resume()
    data_tm:stop()

    print('file_set_num: ' .. file_set_num)

    ---- 1-1-2. load images -----------------------------------------------------
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
    end

    file_set_num = file_set_num + 1

    inputD[{ {}, {1}, {}, {} }] = real_none[{ {}, {}, {} }]

    ---- 1-1-3. train with real ---------------------------------------------------
    local outputD = netD:forward(inputD) -- inputD: real_none / outputD: output_real
    label:fill(real_label) -- real_label = 1
    local errD_real = criterion:forward(outputD, label) -- output_real & 1
    local df_do = criterion:backward(outputD, label)
    netD:backward(inputD, df_do)

    ---- 1-1-4. generate real_reduced ---------------------------------------------
    local real_reduced = torch.Tensor(opt.batchSize * patchNumber, opt.patchSize/2, opt.patchSize/2)
    real_none = real_none:float()
    for i = 1, opt.patchSize/2 do
        for j = 1, opt.patchSize/2 do
            real_reduced[{ {}, {i}, {j} }] = (real_none[{ {}, {2*i-1}, {2*j-1} }] + real_none[{ {}, {2*i}, {2*j-1} }] + real_none[{ {}, {2*i-1}, {2*j} }] + real_none[{ {}, {2*i}, {2*j} }]) / 4
        end
    end
    real_reduced = real_reduced:cuda()

    ---- 1-1-5. generate fake_none ------------------------------------------------
    inputG[{ {}, {1}, {}, {} }] = real_reduced[{ {}, {}, {} }]
    local fake_none = netG:forward(inputG) -- inputG: real_reduced

    ---- 1-1-6. train with fake ---------------------------------------------------
    inputD[{ {}, {1}, {}, {} }] = fake_none[{ {}, {}, {} }]
    local outputD = netD:forward(inputD) -- inputD: fake_none / outputD: output_fake
    label:fill(fake_label) -- fake_label = 0
    local errD_fake = criterion:forward(outputD, label) -- output_fake & 0
    local df_do = criterion:backward(outputD, label)
    netD:backward(inputD, df_do)

    ---- 1-1-7. conclusion --------------------------------------------------------
    print(('errD_real: %.8f  errD_fake: %.8f'):format(errD_real, errD_fake))

    errD = errD_real + errD_fake
    return errD, gradParametersD
end

---- 1-2. set fGx -----------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    ---- 1-2-1. settings ----------------------------------------------------------
    gradParametersG:zero()

    ---- 1-2-2. train -------------------------------------------------------------
    label:fill(real_label) -- real_label = 0
    local outputD = netD.output -- outputD: output_fake
    errG = criterion:forward(outputD, label) -- output_fake & 1
    local df_do = criterion:backward(outputD, label)
    local df_dg = netD:updateGradInput(inputD, df_do) -- inputD: fake_none
    netG:backward(inputG, df_dg) -- inputG: real_reduced

    ---- 1-2-3. conclusion --------------------------------------------------------
    return errG, gradParametersG
end

---- 2. train ---------------------------------------------------------------------
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

---- 3. test ----------------------------------------------------------------------
-- @TODO

---- 4. make samples --------------------------------------------------------------
-- check real image: test
-- get real_none_test(size: 64x64)
local real_none_test = image.load('/CelebA/Img/img_align_celeba/Img/202001.jpg', 1, 'float')
real_none_test = image.scale(real_none_test, opt.fineSize, opt.fineSize)
image.save('real_none_test.jpg', image.toDisplayTensor(real_none_test))
print(('real_none_test-max: %.8f  real_none_test-min: %.8f'):format(real_none_test:max(), real_none_test:min()))
print(('real_none_test-sum: %.8f  real_none_test-std: %.8f'):format(real_none_test:sum(), real_none_test:std()))

-- image size: 64x64 / patchSize: 8x8 / overlap: 4
-- overlap = 4 / overlapPatchLine = 15 / overlapPatchSize = 255
local overlapPatchLine = (opt.fineSize - opt.overlap) / (opt.patchSize - opt.overlap)
local overlapPatchNumber = overlapPatchLine * overlapPatchLine

-- make real_none_patch_test
-- 1 -> (0,0) / 2 -> (0,4) / 3 -> (0,8) / 15 -> (0,56) / 16 -> (4,0) / 17 -> (4,4) / 255 -> (56,56)
local real_none_patch_test = torch.Tensor(overlapPatchNumber, opt.patchSize, opt.patchSize)
for i = 1, overlapPatchNumber do
    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            real_none_patch_test[{ {i}, {a}, {b} }] = real_none_test[{ { math.floor((i-1) / overlapPatchLine) * opt.overlap + a }, { ((i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine) * opt.overlap + b } }]
        end
    end
end

-- make real_reduced_patch_test
local real_reduced_patch_test = torch.Tensor(overlapPatchNumber, opt.patchSize/2, opt.patchSize/2)
for i = 1, opt.patchSize/2 do
    for j = 1, opt.patchSize/2 do
        real_reduced_patch_test[{ {}, {i}, {j} }] = (real_none_patch_test[{ {}, {2*i-1}, {2*j-1} }] + real_none_patch_test[{ {}, {2*i}, {2*j-1} }] + real_none_patch_test[{ {}, {2*i-1}, {2*j} }] + real_none_patch_test[{ {}, {2*i}, {2*j} }]) / 4
    end
end

-- make real_reduced_test
local real_reduced_test = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_test[{ {i}, {j} }] = (real_none_test[{ {2*i-1}, {2*j-1} }] + real_none_test[{ {2*i}, {2*j-1} }] + real_none_test[{ {2*i-1}, {2*j} }] + real_none_test[{ {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_test.jpg', image.toDisplayTensor(real_reduced_test))

-- make real_bilinear_test
local real_bilinear_test = torch.Tensor(opt.fineSize, opt.fineSize)
real_bilinear_test = image.scale(real_reduced_test, opt.fineSize, opt.fineSize, bilinear)
real_bilinear_test = real_bilinear_test:float()
image.save('real_bilinear_test.jpg', image.toDisplayTensor(real_bilinear_test))
print(('PSNR btwn real_none_test & real_bilinear_test: %.4f'):format(calPSNR(real_none_test, real_bilinear_test)))
print(('SSIM btwn real_none_test & real_bilinear_test: %.4f'):format(calSSIM(real_none_test, real_bilinear_test)))

-- generate fake_none_patch_test
local inputG_test = torch.Tensor(overlapPatchNumber, 1, opt.patchSize/2, opt.patchSize/2)
inputG_test[{{}, {1}, {}, {}}] = real_reduced_patch_test[{ {}, {}, {}}]
inputG_test = inputG_test:cuda()
local fake_none_patch_test = netG:forward(inputG_test)
fake_none_patch_test = fake_none_patch_test:float()

-- make fake_none_test(ignore overlap)
local fake_none_test = torch.Tensor(opt.fineSize, opt.fineSize)
for i = 1, overlapPatchNumber do
    -- _index: 0 to 14
    -- (i, x_index) = (1, 0) (2, 0) (3, 0) (16, 1) (17, 1)
    x_index = math.floor((i-1) / overlapPatchLine)
    -- (i, y_index) = (1, 0) (2, 1) (3, 2) (16, 0) (17, 1)
    y_index = (i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine

    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            fake_none_test[{ { x_index * opt.overlap + a }, { y_index * opt.overlap + b } }] = fake_none_patch_test[{ {i}, {1}, {a}, {b} }]
        end
    end
end

-- print info about fake_none_test, and save it
fake_none_test = fake_none_test:float()
print(('fake_none_test-max: %.8f  fake_none_test-min: %.8f'):format(fake_none_test:max(), fake_none_test:min()))
print(('fake_none_test-sum: %.8f  fake_none_test-std: %.8f'):format(fake_none_test:sum(), fake_none_test:std()))
print(('PSNR btwn real_none_test & fake_none_test: %.4f'):format(calPSNR(real_none_test, fake_none_test)))
print(('SSIM btwn real_none_test & fake_none_test: %.4f'):format(calSSIM(real_none_test, fake_none_test)))
image.save('fake_none_test.jpg', image.toDisplayTensor(fake_none_test))

-- make fake_none_overlap_test(apply overlap using algorithm)
local fake_none_overlap_test = torch.Tensor(opt.fineSize, opt.fineSize)
fake_none_overlap_test:fill(0)
---- initialization
local overlap_delta_x = torch.Tensor(opt.overlap, opt.patchSize)
local overlap_delta_path_x = torch.Tensor(opt.overlap, opt.patchSize)
local overlap_delta_y = torch.Tensor(opt.patchSize, opt.overlap)
local overlap_delta_path_y = torch.Tensor(opt.patchSize, opt.overlap)
local overlap_index = torch.Tensor(opt.patchSize)
---- for every i,
for i = 1, overlapPatchNumber do
    -- _index: 0 to 14
    -- (i, x_index) = (1, 0) (2, 0) (3, 0) (16, 1) (17, 1)
    x_index = math.floor((i-1) / overlapPatchLine)
    -- (i, y_index) = (1, 0) (2, 1) (3, 2) (16, 0) (17, 1)
    y_index = (i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine

    overlap_delta_x:fill(0)
    overlap_delta_path_x:fill(0)
    overlap_delta_y:fill(0)
    overlap_delta_path_y:fill(0)
    overlap_index:fill(0)

    -- row 1
    if x_index == 0 then
        -- row 1 & column 1 -> ignore overlap
        if y_index == 0 then
            for a = 1, opt.patchSize do
                for b = 1, opt.patchSize do
                    fake_none_overlap_test[{ { x_index * opt.overlap + a }, { y_index * opt.overlap + b } }] = fake_none_patch_test[{ {i}, {1}, {a}, {b} }]
                end
            end
        -- row 1 & other column -> consider overlap on left side
        else
            -- make overlap_delta_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    overlap_delta_y[{ {a}, {b} }] = math.abs(fake_none_patch_test[i-1][1][a][opt.patchSize - opt.overlap + b] - fake_none_patch_test[i][1][a][b])
                end
            end
            -- make overlap_delta_path_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    -- row 1
                    if a == 1 then
                        overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[{ {a}, {b} }]
                    else
                        if b == 1 then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        elseif b == opt.overlap then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b])
                        else
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        end
                    end
                end
            end
            -- calculate last row of overlap_index
            a = opt.patchSize
            for b = 1, opt.overlap do
                if overlap_delta_path_y[a][b] == (overlap_delta_path_y[{ {a}, {} }]):min() then
                    overlap_index[a] = b
                end
            end
            -- calculate other rows of overlap_index
            for a = opt.patchSize-1, 1, -1 do
                if overlap_index[a+1] == 1 then
                    if overlap_delta_path_y[a][1] == math.min(overlap_delta_path_y[a][1], overlap_delta_path_y[a][2]) then
                        overlap_index[a] = 1
                    else
                        overlap_index[a] = 2
                    end
                elseif overlap_index[a+1] == opt.overlap then
                    if overlap_delta_path_y[a][opt.overlap] == math.min(overlap_delta_path_y[a][opt.overlap], overlap_delta_path_y[a][opt.overlap-1]) then
                        overlap_index[a] = opt.overlap
                    else
                        overlap_index[a] = opt.overlap - 1
                    end
                else
                    b = overlap_index[a+1]
                    if overlap_delta_path_y[a][b] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b
                    elseif overlap_delta_path_y[a][b+1] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b + 1
                    else
                        overlap_index[a] = b - 1
                    end
                end
            end
            -- make fake_none_overlap_test which overlap applied, using overlap_index
            for a = 1, opt.patchSize do
                for b = 1, overlap_index[a] do
                    fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i-1}, {1}, {a}, {opt.patchSize - opt.overlap + b} }]
                end
                for b = overlap_index[a] + 1, opt.patchSize do
                    fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i}, {1}, {a}, {b} }]
                end
            end
        end
    -- other rows
    else
        -- column 1 -> consider overlap on top side
        -- overlap of top side
        -- make overlap_delta_x
        for a = 1, opt.overlap do
            for b = 1, opt.patchSize do
                overlap_delta_x[{ {a}, {b} }] = math.abs(fake_none_patch_test[i-1][1][opt.patchSize - opt.overlap + a][b] - fake_none_patch_test[i][1][a][b])
            end
        end
        -- make overlap_delta_path_x
        for b = 1, opt.patchSize do
            for a = 1, opt.overlap do
                -- column 1
                if b == 1 then
                    overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[{ {a}, {b} }]
                else
                    if a == 1 then
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a][b-1], overlap_delta_path_x[a+1][b-1])
                    elseif a == opt.overlap then
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a-1][b-1], overlap_delta_path_x[a][b-1])
                    else
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a-1][b-1], overlap_delta_path_x[a][b-1], overlap_delta_path_x[a+1][b-1])
                    end
                end
            end
        end
        -- calculate last column of overlap_index
        b = opt.patchSize
        for a = 1, opt.overlap do
            if overlap_delta_path_x[a][b] == (overlap_delta_path_x[{ {}, {b} }]):min() then
                overlap_index[b] = a
            end
        end
        -- calculate other columns of overlap_index
        for b = opt.patchSize-1, 1, -1 do
            if overlap_index[b+1] == 1 then
                if overlap_delta_path_x[1][b] == math.min(overlap_delta_path_x[1][b], overlap_delta_path_x[2][b]) then
                    overlap_index[b] = 1
                else
                    overlap_index[b] = 2
                end
            elseif overlap_index[b+1] == opt.overlap then
                if overlap_delta_path_x[opt.overlap][b] == math.min(overlap_delta_path_x[opt.overlap][b], overlap_delta_path_x[opt.overlap-1][b]) then
                    overlap_index[b] = opt.overlap
                else
                    overlap_index[b] = opt.overlap - 1
                end
            else
                a = overlap_index[b+1]
                if overlap_delta_path_x[a][b] == math.min(overlap_delta_path_x[a][b], overlap_delta_path_x[a-1][b], overlap_delta_path_x[a+1][b]) then
                    overlap_index[b] = a
                elseif overlap_delta_path_x[a+1][b] == math.min(overlap_delta_path_x[a][b], overlap_delta_path_x[a-1][b], overlap_delta_path_x[a+1][b]) then
                    overlap_index[b] = a + 1
                else
                    overlap_index[b] = a - 1
                end
            end
        end
        -- make fake_none_overlap_test which overlap applied, using overlap_index
        for b = 1, opt.patchSize do
            for a = 1, overlap_index[b] do
                fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i-overlapPatchLine}, {1}, {opt.patchSize - opt.overlap + a}, {b} }]
            end
            for a = overlap_index[b] + 1, opt.patchSize do
                fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i}, {1}, {a}, {b} }]
            end
        end
        
        -- not row 1 & not column 1 -> consider overlap on both top side and left side
        if y_index ~= 0 then
            -- overlap of left side
            -- make overlap_delta_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    overlap_delta_y[{ {a}, {b} }] = math.abs(fake_none_patch_test[i-1][1][a][opt.patchSize - opt.overlap + b] - fake_none_patch_test[i][1][a][b])
                end
            end
            -- make overlap_delta_path_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    -- row 1
                    if a == 1 then
                        overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[{ {a}, {b} }]
                    else
                        if b == 1 then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        elseif b == opt.overlap then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b])
                        else
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        end
                    end
                end
            end
            -- calculate last row of overlap_index
            a = opt.patchSize
            for b = 1, opt.overlap do
                if overlap_delta_path_y[a][b] == (overlap_delta_path_y[{ {a}, {} }]):min() then
                    overlap_index[a] = b
                end
            end
            -- calculate other rows of overlap_index
            for a = opt.patchSize-1, 1, -1 do
                if overlap_index[a+1] == 1 then
                    if overlap_delta_path_y[a][1] == math.min(overlap_delta_path_y[a][1], overlap_delta_path_y[a][2]) then
                        overlap_index[a] = 1
                    else
                        overlap_index[a] = 2
                    end
                elseif overlap_index[a+1] == opt.overlap then
                    if overlap_delta_path_y[a][opt.overlap] == math.min(overlap_delta_path_y[a][opt.overlap], overlap_delta_path_y[a][opt.overlap-1]) then
                        overlap_index[a] = opt.overlap
                    else
                        overlap_index[a] = opt.overlap - 1
                    end
                else
                    b = overlap_index[a+1]
                    if overlap_delta_path_y[a][b] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b
                    elseif overlap_delta_path_y[a][b+1] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b + 1
                    else
                        overlap_index[a] = b - 1
                    end
                end
            end
            -- make fake_none_overlap_test which overlap applied, using overlap_index
            for a = 1, opt.patchSize do
                for b = 1, overlap_index[a] do
                    fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i-1}, {1}, {a}, {opt.patchSize - opt.overlap + b} }]
                end
                for b = overlap_index[a] + 1, opt.patchSize do
                    fake_none_overlap_test[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_test[{ {i}, {1}, {a}, {b} }]
                end
            end
        end
    end
end

-- print info about fake_none_overlap_test, and save it
fake_none_overlap_test = fake_none_overlap_test:float()
print(('fake_none_overlap_test-max: %.8f  fake_none_overlap_test-min: %.8f'):format(fake_none_overlap_test:max(), fake_none_overlap_test:min()))
print(('fake_none_overlap_test-sum: %.8f  fake_none_overlap_test-std: %.8f'):format(fake_none_overlap_test:sum(), fake_none_overlap_test:std()))
image.save('fake_none_overlap_test.jpg', image.toDisplayTensor(fake_none_overlap_test))

-- print PSNR btwn real_none_test & fake_none_overlap_test
print(('PSNR btwn real_none_test & fake_none_overlap_test: %.4f'):format(calPSNR(real_none_test, fake_none_overlap_test)))
print(('SSIM btwn real_none_test & fake_none_overlap_test: %.4f'):format(calSSIM(real_none_test, fake_none_overlap_test)))

--======================================================================================================================

-- check real image: train
-- get real_none_train(size: 64x64)
local real_none_train = image.load('/CelebA/Img/img_align_celeba/Img/000001.jpg', 1, 'float')
real_none_train = image.scale(real_none_train, opt.fineSize, opt.fineSize)
image.save('real_none_train.jpg', image.toDisplayTensor(real_none_train))
print(('real_none_train-max: %.8f  real_none_train-min: %.8f'):format(real_none_train:max(), real_none_train:min()))
print(('real_none_train-sum: %.8f  real_none_train-std: %.8f'):format(real_none_train:sum(), real_none_train:std()))

-- image size: 64x64 / patchSize: 8x8 / overlap: 4
-- overlap = 4 / overlapPatchLine = 15 / overlapPatchSize = 255
local overlapPatchLine = (opt.fineSize - opt.overlap) / (opt.patchSize - opt.overlap)
local overlapPatchNumber = overlapPatchLine * overlapPatchLine

-- make real_none_patch_train
-- 1 -> (0,0) / 2 -> (0,4) / 3 -> (0,8) / 15 -> (0,56) / 16 -> (4,0) / 17 -> (4,4) / 255 -> (56,56)
local real_none_patch_train = torch.Tensor(overlapPatchNumber, opt.patchSize, opt.patchSize)
for i = 1, overlapPatchNumber do
    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            real_none_patch_train[{ {i}, {a}, {b} }] = real_none_train[{ { math.floor((i-1) / overlapPatchLine) * opt.overlap + a }, { ((i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine) * opt.overlap + b } }]
        end
    end
end

-- make real_reduced_patch_train
local real_reduced_patch_train = torch.Tensor(overlapPatchNumber, opt.patchSize/2, opt.patchSize/2)
for i = 1, opt.patchSize/2 do
    for j = 1, opt.patchSize/2 do
        real_reduced_patch_train[{ {}, {i}, {j} }] = (real_none_patch_train[{ {}, {2*i-1}, {2*j-1} }] + real_none_patch_train[{ {}, {2*i}, {2*j-1} }] + real_none_patch_train[{ {}, {2*i-1}, {2*j} }] + real_none_patch_train[{ {}, {2*i}, {2*j} }]) / 4
    end
end

-- make real_reduced_train
local real_reduced_train = torch.Tensor(opt.fineSize/2, opt.fineSize/2)
for i = 1, opt.fineSize/2 do
    for j = 1, opt.fineSize/2 do
        real_reduced_train[{ {i}, {j} }] = (real_none_train[{ {2*i-1}, {2*j-1} }] + real_none_train[{ {2*i}, {2*j-1} }] + real_none_train[{ {2*i-1}, {2*j} }] + real_none_train[{ {2*i}, {2*j} }]) / 4
    end
end
image.save('real_reduced_train.jpg', image.toDisplayTensor(real_reduced_train))

-- make real_bilinear_train
local real_bilinear_train = torch.Tensor(opt.fineSize, opt.fineSize)
real_bilinear_train = image.scale(real_reduced_train, opt.fineSize, opt.fineSize, bilinear)
real_bilinear_train = real_bilinear_train:float()
image.save('real_bilinear_train.jpg', image.toDisplayTensor(real_bilinear_train))
print(('PSNR btwn real_none_train & real_bilinear_train: %.4f'):format(calPSNR(real_none_train, real_bilinear_train)))
print(('SSIM btwn real_none_train & real_bilinear_train: %.4f'):format(calSSIM(real_none_train, real_bilinear_train)))

-- generate fake_none_patch_train
local inputG_train = torch.Tensor(overlapPatchNumber, 1, opt.patchSize/2, opt.patchSize/2)
inputG_train[{{}, {1}, {}, {}}] = real_reduced_patch_train[{ {}, {}, {}}]
inputG_train = inputG_train:cuda()
local fake_none_patch_train = netG:forward(inputG_train)
fake_none_patch_train = fake_none_patch_train:float()

-- make fake_none_train(ignore overlap)
local fake_none_train = torch.Tensor(opt.fineSize, opt.fineSize)
for i = 1, overlapPatchNumber do
    -- _index: 0 to 14
    -- (i, x_index) = (1, 0) (2, 0) (3, 0) (16, 1) (17, 1)
    x_index = math.floor((i-1) / overlapPatchLine)
    -- (i, y_index) = (1, 0) (2, 1) (3, 2) (16, 0) (17, 1)
    y_index = (i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine

    for a = 1, opt.patchSize do
        for b = 1, opt.patchSize do
            fake_none_train[{ { x_index * opt.overlap + a }, { y_index * opt.overlap + b } }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
        end
    end
end

-- print info about fake_none_train, and save it
fake_none_train = fake_none_train:float()
print(('fake_none_train-max: %.8f  fake_none_train-min: %.8f'):format(fake_none_train:max(), fake_none_train:min()))
print(('fake_none_train-sum: %.8f  fake_none_train-std: %.8f'):format(fake_none_train:sum(), fake_none_train:std()))
print(('PSNR btwn real_none_train & fake_none_train: %.4f'):format(calPSNR(real_none_train, fake_none_train)))
print(('SSIM btwn real_none_train & fake_none_train: %.4f'):format(calSSIM(real_none_train, fake_none_train)))
image.save('fake_none_train.jpg', image.toDisplayTensor(fake_none_train))

-- make fake_none_overlap_train(apply overlap using algorithm)
local fake_none_overlap_train = torch.Tensor(opt.fineSize, opt.fineSize)
fake_none_overlap_train:fill(0)
---- initialization
local overlap_delta_x = torch.Tensor(opt.overlap, opt.patchSize)
local overlap_delta_path_x = torch.Tensor(opt.overlap, opt.patchSize)
local overlap_delta_y = torch.Tensor(opt.patchSize, opt.overlap)
local overlap_delta_path_y = torch.Tensor(opt.patchSize, opt.overlap)
local overlap_index = torch.Tensor(opt.patchSize)
---- for every i,
for i = 1, overlapPatchNumber do
    -- _index: 0 to 14
    -- (i, x_index) = (1, 0) (2, 0) (3, 0) (16, 1) (17, 1)
    x_index = math.floor((i-1) / overlapPatchLine)
    -- (i, y_index) = (1, 0) (2, 1) (3, 2) (16, 0) (17, 1)
    y_index = (i-1) - math.floor((i-1) / overlapPatchLine) * overlapPatchLine

    overlap_delta_x:fill(0)
    overlap_delta_path_x:fill(0)
    overlap_delta_y:fill(0)
    overlap_delta_path_y:fill(0)
    overlap_index:fill(0)

    -- row 1
    if x_index == 0 then
        -- row 1 & column 1 -> ignore overlap
        if y_index == 0 then
            for a = 1, opt.patchSize do
                for b = 1, opt.patchSize do
                    fake_none_overlap_train[{ { x_index * opt.overlap + a }, { y_index * opt.overlap + b } }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
                end
            end
        -- row 1 & other column -> consider overlap on left side
        else
            -- make overlap_delta_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    overlap_delta_y[{ {a}, {b} }] = math.abs(fake_none_patch_train[i-1][1][a][opt.patchSize - opt.overlap + b] - fake_none_patch_train[i][1][a][b])
                end
            end
            -- make overlap_delta_path_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    -- row 1
                    if a == 1 then
                        overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[{ {a}, {b} }]
                    else
                        if b == 1 then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        elseif b == opt.overlap then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b])
                        else
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        end
                    end
                end
            end
            -- calculate last row of overlap_index
            a = opt.patchSize
            for b = 1, opt.overlap do
                if overlap_delta_path_y[a][b] == (overlap_delta_path_y[{ {a}, {} }]):min() then
                    overlap_index[a] = b
                end
            end
            -- calculate other rows of overlap_index
            for a = opt.patchSize-1, 1, -1 do
                if overlap_index[a+1] == 1 then
                    if overlap_delta_path_y[a][1] == math.min(overlap_delta_path_y[a][1], overlap_delta_path_y[a][2]) then
                        overlap_index[a] = 1
                    else
                        overlap_index[a] = 2
                    end
                elseif overlap_index[a+1] == opt.overlap then
                    if overlap_delta_path_y[a][opt.overlap] == math.min(overlap_delta_path_y[a][opt.overlap], overlap_delta_path_y[a][opt.overlap-1]) then
                        overlap_index[a] = opt.overlap
                    else
                        overlap_index[a] = opt.overlap - 1
                    end
                else
                    b = overlap_index[a+1]
                    if overlap_delta_path_y[a][b] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b
                    elseif overlap_delta_path_y[a][b+1] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b + 1
                    else
                        overlap_index[a] = b - 1
                    end
                end
            end
            -- make fake_none_overlap_train which overlap applied, using overlap_index
            for a = 1, opt.patchSize do
                for b = 1, overlap_index[a] do
                    fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i-1}, {1}, {a}, {opt.patchSize - opt.overlap + b} }]
                end
                for b = overlap_index[a] + 1, opt.patchSize do
                    fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
                end
            end
        end
    -- other rows
    else
        -- column 1 -> consider overlap on top side
        -- overlap of top side
        -- make overlap_delta_x
        for a = 1, opt.overlap do
            for b = 1, opt.patchSize do
                overlap_delta_x[{ {a}, {b} }] = math.abs(fake_none_patch_train[i-1][1][opt.patchSize - opt.overlap + a][b] - fake_none_patch_train[i][1][a][b])
            end
        end
        -- make overlap_delta_path_x
        for b = 1, opt.patchSize do
            for a = 1, opt.overlap do
                -- column 1
                if b == 1 then
                    overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[{ {a}, {b} }]
                else
                    if a == 1 then
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a][b-1], overlap_delta_path_x[a+1][b-1])
                    elseif a == opt.overlap then
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a-1][b-1], overlap_delta_path_x[a][b-1])
                    else
                        overlap_delta_path_x[{ {a}, {b} }] = overlap_delta_x[a][b] + math.min(overlap_delta_path_x[a-1][b-1], overlap_delta_path_x[a][b-1], overlap_delta_path_x[a+1][b-1])
                    end
                end
            end
        end
        -- calculate last column of overlap_index
        b = opt.patchSize
        for a = 1, opt.overlap do
            if overlap_delta_path_x[a][b] == (overlap_delta_path_x[{ {}, {b} }]):min() then
                overlap_index[b] = a
            end
        end
        -- calculate other columns of overlap_index
        for b = opt.patchSize-1, 1, -1 do
            if overlap_index[b+1] == 1 then
                if overlap_delta_path_x[1][b] == math.min(overlap_delta_path_x[1][b], overlap_delta_path_x[2][b]) then
                    overlap_index[b] = 1
                else
                    overlap_index[b] = 2
                end
            elseif overlap_index[b+1] == opt.overlap then
                if overlap_delta_path_x[opt.overlap][b] == math.min(overlap_delta_path_x[opt.overlap][b], overlap_delta_path_x[opt.overlap-1][b]) then
                    overlap_index[b] = opt.overlap
                else
                    overlap_index[b] = opt.overlap - 1
                end
            else
                a = overlap_index[b+1]
                if overlap_delta_path_x[a][b] == math.min(overlap_delta_path_x[a][b], overlap_delta_path_x[a-1][b], overlap_delta_path_x[a+1][b]) then
                    overlap_index[b] = a
                elseif overlap_delta_path_x[a+1][b] == math.min(overlap_delta_path_x[a][b], overlap_delta_path_x[a-1][b], overlap_delta_path_x[a+1][b]) then
                    overlap_index[b] = a + 1
                else
                    overlap_index[b] = a - 1
                end
            end
        end
        -- make fake_none_overlap_train which overlap applied, using overlap_index
        for b = 1, opt.patchSize do
            for a = 1, overlap_index[b] do
                fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i-overlapPatchLine}, {1}, {opt.patchSize - opt.overlap + a}, {b} }]
            end
            for a = overlap_index[b] + 1, opt.patchSize do
                fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
            end
        end
        
        -- not row 1 & not column 1 -> consider overlap on both top side and left side
        if y_index ~= 0 then
            -- overlap of left side
            -- make overlap_delta_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    overlap_delta_y[{ {a}, {b} }] = math.abs(fake_none_patch_train[i-1][1][a][opt.patchSize - opt.overlap + b] - fake_none_patch_train[i][1][a][b])
                end
            end
            -- make overlap_delta_path_y
            for a = 1, opt.patchSize do
                for b = 1, opt.overlap do
                    -- row 1
                    if a == 1 then
                        overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[{ {a}, {b} }]
                    else
                        if b == 1 then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        elseif b == opt.overlap then
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b])
                        else
                            overlap_delta_path_y[{ {a}, {b} }] = overlap_delta_y[a][b] + math.min(overlap_delta_path_y[a-1][b-1], overlap_delta_path_y[a-1][b], overlap_delta_path_y[a-1][b+1])
                        end
                    end
                end
            end
            -- calculate last row of overlap_index
            a = opt.patchSize
            for b = 1, opt.overlap do
                if overlap_delta_path_y[a][b] == (overlap_delta_path_y[{ {a}, {} }]):min() then
                    overlap_index[a] = b
                end
            end
            -- calculate other rows of overlap_index
            for a = opt.patchSize-1, 1, -1 do
                if overlap_index[a+1] == 1 then
                    if overlap_delta_path_y[a][1] == math.min(overlap_delta_path_y[a][1], overlap_delta_path_y[a][2]) then
                        overlap_index[a] = 1
                    else
                        overlap_index[a] = 2
                    end
                elseif overlap_index[a+1] == opt.overlap then
                    if overlap_delta_path_y[a][opt.overlap] == math.min(overlap_delta_path_y[a][opt.overlap], overlap_delta_path_y[a][opt.overlap-1]) then
                        overlap_index[a] = opt.overlap
                    else
                        overlap_index[a] = opt.overlap - 1
                    end
                else
                    b = overlap_index[a+1]
                    if overlap_delta_path_y[a][b] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b
                    elseif overlap_delta_path_y[a][b+1] == math.min(overlap_delta_path_y[a][b], overlap_delta_path_y[a][b-1], overlap_delta_path_y[a][b+1]) then
                        overlap_index[a] = b + 1
                    else
                        overlap_index[a] = b - 1
                    end
                end
            end
            -- make fake_none_overlap_train which overlap applied, using overlap_index
            for a = 1, opt.patchSize do
                for b = 1, overlap_index[a] do
                    fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i-1}, {1}, {a}, {opt.patchSize - opt.overlap + b} }]
                end
                for b = overlap_index[a] + 1, opt.patchSize do
                    fake_none_overlap_train[{ {x_index * opt.overlap + a}, {y_index * opt.overlap + b} }] = fake_none_patch_train[{ {i}, {1}, {a}, {b} }]
                end
            end
        end
    end
end

-- print info about fake_none_overlap_train, and save it
fake_none_overlap_train = fake_none_overlap_train:float()
print(('fake_none_overlap_train-max: %.8f  fake_none_overlap_train-min: %.8f'):format(fake_none_overlap_train:max(), fake_none_overlap_train:min()))
print(('fake_none_overlap_train-sum: %.8f  fake_none_overlap_train-std: %.8f'):format(fake_none_overlap_train:sum(), fake_none_overlap_train:std()))
image.save('fake_none_overlap_train.jpg', image.toDisplayTensor(fake_none_overlap_train))

-- print PSNR btwn real_none_train & fake_none_overlap_train
print(('PSNR btwn real_none_train & fake_none_overlap_train: %.4f'):format(calPSNR(real_none_train, fake_none_overlap_train)))
print(('SSIM btwn real_none_train & fake_none_overlap_train: %.4f'):format(calSSIM(real_none_train, fake_none_overlap_train)))

--======================================================================================================================

print(('Total time: %.3f'):format(total_tm:time().real))