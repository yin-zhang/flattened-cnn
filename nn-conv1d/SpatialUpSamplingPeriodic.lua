local SpatialUpSamplingPeriodic, parent = torch.class('nn.SpatialUpSamplingPeriodic', 'nn.Module')

--[[
Applies a 2D up-sampling over an input image composed of several input planes.

    The upsampling is done in a periodic fashion.  That is: only output coodinates that are multiple of
    scale_factor has non-zero values.

The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
instance, if the tensor is 4D, then dim 3 is the y dimension and dim 4 is the x.

owidth  = width*scale_factor
oheight  = height*scale_factor
--]]

function SpatialUpSamplingPeriodic:__init(xscale,yscale)
   parent.__init(self)

   if xscale == nil then
       xscale = 1
   end
   
   if yscale == nil then
       yscale = xscale
   end
   
   self.x_scale_factor = xscale
   self.y_scale_factor = yscale
   if self.x_scale_factor < 1 or self.y_scale_factor < 1 then
     error('scale_factor must be no less than 1')
   end
   if math.floor(self.x_scale_factor) ~= self.x_scale_factor or math.floor(self.y_scale_factor) ~= self.y_scale_factor then
     error('scale_factor must be integer')
   end
   self.inputSize = torch.LongStorage(4)
   self.outputSize = torch.LongStorage(4)
   self.usage = nil
end

function SpatialUpSamplingPeriodic:updateOutput(input)
   if input:dim() ~= 4 and input:dim() ~= 3 then
     error('SpatialUpSamplingPeriodic only support 3D or 4D tensors')
   end
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = self.outputSize[ydim] * self.y_scale_factor
   self.outputSize[xdim] = self.outputSize[xdim] * self.x_scale_factor
   -- Resize the output if needed
   if input:dim() == 3 then
     self.output:resize(self.outputSize[1], self.outputSize[2],
       self.outputSize[3])
   else
     self.output:resize(self.outputSize)
   end
   input.nn.SpatialUpSamplingPeriodic_updateOutput(self, input)
   return self.output
end

function SpatialUpSamplingPeriodic:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   input.nn.SpatialUpSamplingPeriodic_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
