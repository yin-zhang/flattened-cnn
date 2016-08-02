local LateralMaskedConvolution, parent = torch.class('nn.LateralMaskedConvolution', 'nn.Module')

function LateralMaskedConvolution:__init(nInputPlane, nOutputPlane, mask)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.mask = mask

   self.weight = torch.Tensor(nOutputPlane, nInputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.ones = torch.Tensor()

   self:reset()
end

function LateralMaskedConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function LateralMaskedConvolution:updateOutput(input)
   input = makeContiguous(self, input)
   return input.nn.LateralMaskedConvolution_updateOutput(self, input)
end

function LateralMaskedConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      return input.nn.LateralMaskedConvolution_updateGradInput(self, input, gradOutput)
   end
end

function LateralMaskedConvolution:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return input.nn.LateralMaskedConvolution_accGradParameters(self, input, gradOutput, scale)
end
