require "torch"
require "image"
require "cudnn"

dofile 'vis.lua'

------------------------------------------------------------------------------------

local outputImages = "outputImages/"
local inputImages = "inputImages/"
local imgFileName = "img"
local outFileName = "out"
local fmapFileName = "fmap"
local maskFileName = "mask"
local imgExt = ".jpg"

local imgCnt = 22
local imgCh = 1
local imgH = 125
local imgW = 640

local imgBatch = torch.CudaTensor(imgCnt, imgCh, imgH, imgW)

------------------------------------------------------------------------------------

local function getImages(n, visMask, fMaps, fMapsM)
  local imgOut = torch.CudaTensor(3, imgH, imgW)
  local w = visMask:size(4) 
  local h = visMask:size(3)
  local fMapsImg = torch.ones(#fMaps * h + (#fMaps - 1) * 2, w):cuda()
  local fMapsImgM = torch.ones(#fMaps * h + (#fMaps - 1) * 2, w):cuda()
  --normalize and scale averaged feature maps and intermediate visualization masks 
  for i = 1, #fMaps do
    local min = fMaps[i][n][1]:min()
    local max = fMaps[i][n][1]:max()
    fMaps[i][n][1]:add(-min)
    fMaps[i][n][1]:div(max - min)
    local min = fMapsM[i][n][1]:min()
    local max = fMapsM[i][n][1]:max()
    fMapsM[i][n][1]:add(-min)
    fMapsM[i][n][1]:div(max - min)
    fMapsImg:narrow(1, 1 + (i - 1) * (h + 2), h):
      copy(image.scale(fMaps[i][n][1]:float(), w, h):cuda())
    fMapsImgM:narrow(1, 1 + (i - 1) * (h + 2), h):
      copy(image.scale(fMapsM[i][n][1]:float(), w, h):cuda())
  end
  --overlay visualization mask over the input images
  imgOut[1]:copy(imgBatch[n][1]):add(visMask[n])
  imgOut[2]:copy(imgBatch[n][1]):add(-visMask[n])
  imgOut[3]:copy(imgBatch[n][1]):add(-visMask[n])
  imgOut:clamp(0, 1)
  return imgOut, fMapsImg, fMapsImgM
end

------------------------------------------------------------------------------------

print("Loading model...")
model = torch.load("model.t7b")

print("Loading images...")
for i = 1, 22 do
  imgBatch[i]:copy(image.load(inputImages .. imgFileName .. tostring(i) .. imgExt))
end

print("Generating visualization masks...")
local visMask, fMaps, fMapsM = getVisMask(model:get(1), imgBatch)

print("Saving images...")
for i = 1, 22 do
  local img1, img2, img3 = getImages(i, visMask, fMaps, fMapsM)
  image.save(outputImages..outFileName..tostring(i)..imgExt, img1)
  image.save(outputImages..fmapFileName..tostring(i)..imgExt, img2)
  image.save(outputImages..maskFileName..tostring(i)..imgExt, img3)
end

