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

local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_num = 1

local file_name = file_name_route .. '00000' .. tostring(file_num) .. '.jpg'

print(file_name)

local image_input_color = image.load(file_name, 3, 'float')
image.save('image_input_color.jpg', image.toDisplayTensor(image_input_color))

local image_input_gray = image.load(file_name, 1, 'float')
image.save('image_input_gray.jpg', image.toDisplayTensor(image_input_gray))

-- rgb2gray(image_input_color)

print(image_input_gray)