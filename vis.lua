require "torch"
require "cutorch"
require "nn"
require "cunn"
require "cudnn"

------------------------------------------------------------------------------------

function getVisMask(mod, idata)
  --do the forward pass through the feature extractor (convolutional layers)
  mod:forward(idata)
  local layersReLU = mod:findModules('cudnn.ReLU')
  local layersConv = mod:findModules('cudnn.SpatialConvolution')
  local mask = nil
  local sum = {}
  local sumUp = {}
  local fMaps = {}
  local fMapsMasked = {}
  --process feature maps
  for i = #layersReLU, 1, -1 do
    --sum all the feature maps at each level
    sum[i] = layersReLU[i].output:sum(2)
    --calculate the dimension of scaled up map
    local w = (layersReLU[i].output:size(3) - 1) * layersConv[i].dW + layersConv[i].kW
    local h = (layersReLU[i].output:size(4) - 1) * layersConv[i].dH + layersConv[i].kH
    fMaps[i] = sum[i]:clone()
    --pointwise multiplication
    if i < #layersReLU then
      sum[i]:cmul(sumUp[i + 1])
    end
    --save intermediate mask
    fMapsMasked[i] = sum[i]:clone() 
    --scale up intermediate mask using deconvolution
    if i > 1 then
      local adjw = layersReLU[i - 1].output:size(3) - w 
      local adjh = layersReLU[i - 1].output:size(4) - h
      local mmUp = nn.SpatialFullConvolution(1, 1,
                                             layersConv[i].kW, 
                                             layersConv[i].kH, 
                                             layersConv[i].dW, 
                                             layersConv[i].dH,
                                             0, 0, adjh, adjw)
      mmUp:cuda()
      mmUp:parameters()[1]:fill(1)
      mmUp:parameters()[2]:fill(0)
      sumUp[i] = mmUp:forward(sum[i])
    else
      local adjw = idata:size(3) - w 
      local adjh = idata:size(4) - h
      local mmUp = nn.SpatialFullConvolution(1, 1,
                                             layersConv[i].kW, 
                                             layersConv[i].kH, 
                                             layersConv[i].dW, 
                                             layersConv[i].dH,
                                             0, 0, adjh, adjw)
      mmUp:cuda()
      mmUp:parameters()[1]:fill(1)
      mmUp:parameters()[2]:fill(0)
      sumUp[i] = mmUp:forward(sum[i])
    end
  end
  --assign output - visualization mask
  local out = sumUp[1]
  --normalize mask to range 0-1
  local omin = torch.min(out, 3):min(4):mul(-1) 
  local omax = torch.max(out, 3):max(4):add(omin)
  out:add(torch.expand(omin:reshape(idata:size(1), 1, 1, 1), idata:size(1), 
                       idata:size(2), idata:size(3), idata:size(4)))
  out:cdiv(torch.expand(omax:reshape(idata:size(1), 1, 1, 1), idata:size(1), 
                       idata:size(2), idata:size(3), idata:size(4))) 
  --return visualization mask, averaged feature maps, and intermediate masks
  return out, fMaps, fMapsMasked
end

