require 'torch'
require 'image'

local file_name_route = '/CelebA/Img/img_align_celeba/Img/'

local file_num = 1

local file_name = file_name_route .. '00000' .. tostring(file_num) .. '.jpg'

print(file_name)

local image_input = image.load(file_name, 3, 'float')

print(image_input)