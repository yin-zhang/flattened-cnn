#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSubSamplingPeriodic.c"
#else

static int nnconv1d_(SpatialSubSamplingPeriodic_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int iW = luaT_getfieldcheckint(L, 1, "iW");
  int iH = luaT_getfieldcheckint(L, 1, "iH");

  // TODO: check argument shapes  
  int yDim = input->nDimension-2;
  int xDim = input->nDimension-1;

  // dims
  int idim = input->nDimension;  // Guaranteed to be between 3 and 5
  int osz0 = output->size[0];
  int osz1 = output->size[1];
  int osz2 = output->size[2];
  int osz3 = 1;
  if (idim > 3) {
    osz3 = output->size[3];
  }

  // get strides
  long *is = input->stride;
  long *os = output->stride;

  // get raw pointers
  real *pin = THTensor_(data)(input);
  real *pout = THTensor_(data)(output);

  // perform the subsampling
  int i0, i1, i2, i3, isrc, idst;
  int iout[4];  // Output indices
  int iin[4];  // Input indices

  for (i0 = 0; i0 < osz0; i0++) {
    iout[0] = i0;
    iin[0] = i0;
    for (i1 = 0; i1 < osz1; i1++) {
      iout[1] = i1;
      iin[1] = i1;
      for (i2 = 0; i2 < osz2; i2++) {
        iout[2] = i2;
        iin[2] = i2;
        for (i3 = 0; i3 < osz3; i3++) {
          iout[3] = i3;
          iin[3] = i3;

          idst = i0*os[0] + i1*os[1] + i2*os[2];
          if (idim > 3) {
            idst += i3*os[3];
          }

          // set the indices for the subsampled dimensions
          iin[xDim] = iout[xDim] * dW + iW;
          iin[yDim] = iout[yDim] * dH + iH;
          
          isrc = iin[0]*is[0] + iin[1]*is[1] + iin[2]*is[2];
          if (idim > 3) {
            isrc += iin[3]*is[3];
          }
          
          pout[idst] = pin[isrc];
        }
      }
    }
  }

  return 1;
}

static int
nnconv1d_(SpatialSubSamplingPeriodic_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int iW = luaT_getfieldcheckint(L, 1, "iW");
  int iH = luaT_getfieldcheckint(L, 1, "iH");
  
  // TODO: check argument shapes  
  int yDim = gradInput->nDimension-2;
  int xDim = gradInput->nDimension-1;

  // dims
  int idim = gradInput->nDimension;  // Guaranteed to be between 3 and 5
  int osz0 = gradOutput->size[0];
  int osz1 = gradOutput->size[1];
  int osz2 = gradOutput->size[2];
  int osz3 = 1;
  if (idim > 3) {
    osz3 = gradOutput->size[3];
  }

  // get strides
  long *is = gradInput->stride;
  long *os = gradOutput->stride;

  // get raw pointers
  real *pin = THTensor_(data)(gradInput);
  real *pout = THTensor_(data)(gradOutput);

  // perform the upsampling
  int i0, i1, i2, i3, isrc, idst, x, y;
  int iin[4];  // Input indices
  int iout[4];  // Output indices

  THTensor_(zero)(gradInput);

  for (i0 = 0; i0 < osz0; i0++) {
    iin[0] = i0;
    iout[0] = i0;
    for (i1 = 0; i1 < osz1; i1++) {
      iin[1] = i1;
      iout[1] = i1;
      for (i2 = 0; i2 < osz2; i2++) {
        iin[2] = i2;
        iout[2] = i2;
        for (i3 = 0; i3 < osz3; i3++) {
          iin[3] = i3;
          iout[3] = i3;

          isrc = i0*os[0] + i1*os[1] + i2*os[2];
          if (idim > 3) {
            isrc += i3*os[3];
          }

          // Now accumulate the gradients from gradOutput
          
          iin[xDim] = dW * iout[xDim] - iW;
          iin[yDim] = dH * iout[yDim] + iH;
          idst = iin[0]*is[0] + iin[1]*is[1] + iin[2]*is[2];
          if (idim > 3) {
            idst += iin[3]*is[3];
          }
          pin[idst] += pout[isrc];
        }
      }
    }
  }

  return 1;
}

static const struct luaL_Reg nnconv1d_(SpatialSubSamplingPeriodic__) [] = {
  {"SpatialSubSamplingPeriodic_updateOutput", nnconv1d_(SpatialSubSamplingPeriodic_updateOutput)},
  {"SpatialSubSamplingPeriodic_updateGradInput", nnconv1d_(SpatialSubSamplingPeriodic_updateGradInput)},
  {NULL, NULL}
};

static void nnconv1d_(SpatialSubSamplingPeriodic_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nnconv1d_(SpatialSubSamplingPeriodic__), "nn");
  lua_pop(L,1);
}


#endif
