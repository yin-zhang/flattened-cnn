local SpatialSubSamplingPeriodic, parent = torch.class('nn.SpatialSubSamplingPeriodic', 'nn.Module')

--[[
Applies a 2D up-sampling over an input image composed of several input planes.

    The upsampling is done in a periodic fashion.  That is: only output coodinates that are multiple of
    scale_factor has non-zero values.

The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
instance, if the tensor is 4D, then dim 3 is the y dimension and dim 4 is the x.

owidth  = width*scale_factor
oheight  = height*scale_factor
--]]

function SpatialSubSamplingPeriodic:__init(dW,dH,iW,iH)
   parent.__init(self)

   if dW == nil then
       dW = 1
   end
   if dH == nil then
       dH = dW
   end
   if iW == nil then
       iW = 0
   end
   if iH == nil then
       iH = 0
   end
   
   self.dW = dW
   self.dH = dH
   self.iW = iW
   self.iH = iH
   
   if self.dW < 1 or self.dH < 1 then
       error('scale_factor must be no less than 1')
   end

   if math.floor(dW) ~= dW or math.floor(dH) ~= dH or math.floor(iW) ~= iW or math.floor(iH) ~= iH then
       error('dW, dH, iW, iH must be integer')
   end

   if iW < 0 or iW >= dW then
       error('iW must satisfy 0 <= iW < dW')
   end

   if iH < 0 or iH >= dH then
       error('iH must satisfy 0 <= iH < dH')
   end
   
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
   self.usage = nil
end

function SpatialSubSamplingPeriodic:updateOutput(input)
   if input:dim() ~= 4 and input:dim() ~= 3 then
     error('SpatialSubSamplingPeriodic only support 3D or 4D tensors')
   end
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = math.ceil((self.outputSize[ydim] - self.iH) / self.dH)
   self.outputSize[xdim] = math.ceil((self.outputSize[xdim] - self.iW) / self.dW)
   -- Resize the output if needed
   if input:dim() == 3 then
     self.output:resize(self.outputSize[1], self.outputSize[2],
       self.outputSize[3])
   else
     self.output:resize(self.outputSize)
   end
   input.nn.SpatialSubSamplingPeriodic_updateOutput(self, input)
   return self.output
end

function SpatialSubSamplingPeriodic:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialSubSamplingPeriodic_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
