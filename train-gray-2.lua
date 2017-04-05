require 'torch'
require 'image'

-- -- convert rgb to grayscale by averaging channel intensities
-- function rgb2gray(im)
-- 	-- Image.rgb2y uses a different weight mixture

-- 	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
-- 	if dim ~= 3 then
-- 		 print('<error> expected 3 channels')
-- 		 return im
-- 	end

-- 	-- a cool application of tensor:select
-- 	local r = im:select(1, 1)
-- 	local g = im:select(1, 2)
-- 	local b = im:select(1, 3)

-- 	local z = torch.Tensor(w, h):zero()

-- 	-- z = z + 0.21r
-- 	z = z:add(0.21, r)
-- 	z = z:add(0.72, g)
-- 	z = z:add(0.07, b)
-- 	return z
-- end

local sample_size = 64


local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_num = 1

local file_name = file_name_route .. '00000' .. tostring(file_num) .. '.jpg'

local image_input_gray = image.load(file_name, 1, 'float')

image_input_gray = image.scale(image_input_gray, sample_size, sample_size)

image.save('image_input_gray.jpg', image.toDisplayTensor(image_input_gray))

print(image_input_gray)