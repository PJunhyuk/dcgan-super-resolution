require 'torch'
require 'image'

local image_black = image.load('image_black.png', 3, 'byte')

print(('image_black-max: %.4f  image_black-min: %.4f'):format(image_black:max(), image_black:min()))
print(('image_black-sum: %.4f  image_black-std: %.4f'):format(image_black:sum(), image_black:std()))

local image_white = image.load('image_white.png', 3, 'byte')

print(('image_white-max: %.4f  image_white-min: %.4f'):format(image_white:max(), image_white:min()))
print(('image_white-sum: %.4f  image_white-std: %.4f'):format(image_white:sum(), image_white:std()))

local image_all4 = image.load('image_all4.png', 3, 'byte')

print(('image_all4-max: %.4f  image_all4-min: %.4f'):format(image_all4:max(), image_all4:min()))
print(('image_all4-sum: %.4f  image_all4-std: %.4f'):format(image_all4:sum(), image_all4:std()))

local image_333333 = image.load('image_333333.png', 3, 'byte')

print(('image_333333-max: %.4f  image_333333-min: %.4f'):format(image_333333:max(), image_333333:min()))
print(('image_333333-sum: %.4f  image_333333-std: %.4f'):format(image_333333:sum(), image_333333:std()))