local PlanarConvolution, parent = torch.class('nn.PlanarConvolution', 'nn.Module')

function PlanarConvolution:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)

   assert(nInputPlane == nOutputPlane)
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane

   self.kW = kW
   self.kH = kH

   self.weight = torch.Tensor(nInputPlane, kW, kH)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kW, kH)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.ones = torch.Tensor()
   self.finput = torch.Tensor()
   self.fgradWeight = torch.Tensor()

   self:reset()
end

function PlanarConvolution:reset(stdv)
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

function PlanarConvolution:updateOutput(input)
   input = makeContiguous(self, input)
   return input.nn.PlanarConvolution_updateOutput(self, input)
end

function PlanarConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      return input.nn.PlanarConvolution_updateGradInput(self, input, gradOutput)
   end
end

function PlanarConvolution:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return input.nn.PlanarConvolution_accGradParameters(self, input, gradOutput, scale)
end
