local PlanarConvolution, parent = torch.class('nn.PlanarConvolution', 'nn.Module')

function PlanarConvolution:__init(nInputPlane, nOutputPlane, kW, kH)
   parent.__init(self)

   assert(nInputPlane == nOutputPlane)
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane

   self.kH = kH
   self.kW = kW

   self.weight = torch.Tensor(nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kH, kW)
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

function PlanarConvolution:clearState()
   -- don't call set because it might reset referenced tensors
   local function clear(f)
      if self[f] then
         if torch.isTensor(self[f]) then
            self[f] = self[f].new()
         elseif type(self[f]) == 'table' then
            self[f] = {}
         else
            self[f] = nil
         end
      end
   end
   clear('output')
   clear('gradInput')
   clear('_input')
   clear('_gradOutput')
   return self
end
