require 'torch'
require 'image'

local image_all_0 = torch.Tensor(64, 64)
image_all_0:fill(0)
image.save('image_all_0.png', image.toDisplayTensor(image_all_0))

print(('image_all_0-max: %.4f  image_all_0-min: %.4f'):format(image_all_0:max(), image_all_0:min()))
print(('image_all_0-sum: %.4f  image_all_0-std: %.4f'):format(image_all_0:sum(), image_all_0:std()))

local image_all_05 = torch.Tensor(64, 64)
image_all_05:fill(0.5)
image.save('image_all_05.png', image.toDisplayTensor(image_all_05))

print(('image_all_05-max: %.4f  image_all_05-min: %.4f'):format(image_all_05:max(), image_all_05:min()))
print(('image_all_05-sum: %.4f  image_all_05-std: %.4f'):format(image_all_05:sum(), image_all_05:std()))

local image_all_1 = torch.Tensor(64, 64)
image_all_1:fill(1)
image.save('image_all_1.png', image.toDisplayTensor(image_all_1))

print(('image_all_1-max: %.4f  image_all_1-min: %.4f'):format(image_all_1:max(), image_all_1:min()))
print(('image_all_1-sum: %.4f  image_all_1-std: %.4f'):format(image_all_1:sum(), image_all_1:std()))

local image_all__1 = torch.Tensor(64, 64)
image_all__1:fill(-1)
image.save('image_all__1.png', image.toDisplayTensor(image_all__1))

print(('image_all__1-max: %.4f  image_all__1-min: %.4f'):format(image_all__1:max(), image_all__1:min()))
print(('image_all__1-sum: %.4f  image_all__1-std: %.4f'):format(image_all__1:sum(), image_all__1:std()))

local image_all__05 = torch.Tensor(64, 64)
image_all__05:fill(-0.5)
image.save('image_all__05.png', image.toDisplayTensor(image_all__05))

print(('image_all__05-max: %.4f  image_all__05-min: %.4f'):format(image_all__05:max(), image_all__05:min()))
print(('image_all__05-sum: %.4f  image_all__05-std: %.4f'):format(image_all__05:sum(), image_all__05:std()))