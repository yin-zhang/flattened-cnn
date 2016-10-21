
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


/*
 * Description:
 */

__device__ int xlate_idx(int ii, int d1, int d2, int d3, int D2, int D3, int dW, int dH, int iW, int iH)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  if ((w % dW != iW) || (z % dH != iH)) {
    return -1;
  }
  w = w/dW;
  z = z/dH;
  return (((x*d1+y)*D2)+z)*D3+w;
}

__device__ int xlate_idx_inv(int ii, int d1, int d2, int d3, int D2, int D3, int dW, int dH, int iW, int iH)
{
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*dW+iW;
  z = z*dH+iH;
  return (((x*d1+y)*D2)+z)*D3+w;
}

__global__ void downscale(float *input, float *output, long no_elements,
                        int dW, int dH, int iW, int iH, int d1, int d2, int d3, int D2, int D3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = xlate_idx_inv(ii, d1, d2, d3, D2, D3, dW, dH, iW, iH);
  output[ii]=input[ipidx];
}

static int cunnconv1d_SpatialSubSamplingPeriodic_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int iW = luaT_getfieldcheckint(L, 1, "iW");
  int iH = luaT_getfieldcheckint(L, 1, "iH");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  
  THCudaTensor_zero(state, output);

  input = THCudaTensor_newContiguous(state, input);
  // This is for allocating output Tensor
  long no_elements = 1;
  int i;
  for(i = 0; i < input->nDimension - 2; i++){
    no_elements *= input->size[i];
  }
  no_elements *= (input->size[i++] - iH) / dH;
  no_elements *= (input->size[i++] - iW) / dW;

  int d1;
  int d2;
  int d3;

  if (input->nDimension == 3) {
    d1 = output->size[0];
    d2 = output->size[1];
    d3 = output->size[2];
  } else {
    d1 = output->size[1];
    d2 = output->size[2];
    d3 = output->size[3];
  }

  int D2;
  int D3;

  if (input->nDimension == 3) {
    D2 = input->size[1];
    D3 = input->size[2];
  } else {
    D2 = input->size[2];
    D3 = input->size[3];
  }

  float *input_data = THCudaTensor_data(state, input);
  float *output_data = THCudaTensor_data(state, output);

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  downscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (input_data, output_data, no_elements, dW, dH, iW, iH, d1, d2, d3, D2, D3);
  THCudaCheck(cudaGetLastError());

  // final cut:
  THCudaTensor_free(state, input);

  return 1;
}

/*
 * Description:
 */
__global__ void upscale(float *gradInput_data, float *gradOutput_data, long no_elements,
                              int dW, int dH, int iW, int iH, int d1, int d2, int d3, int D2, int D3)
{
  // output offset:
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = xlate_idx(ii, d1, d2, d3, D2, D3, dW, dH, iW, iH);
  if (ipidx >= 0) {
    gradInput_data[ii] += gradOutput_data[ipidx];
  }
}


static int cunnconv1d_SpatialSubSamplingPeriodic_updateGradInput(lua_State *L)
{

  THCState *state = getCutorchState(L);
  THCudaTensor *input =  (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int iW = luaT_getfieldcheckint(L, 1, "iW");
  int iH = luaT_getfieldcheckint(L, 1, "iH");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_zero(state, gradInput);

  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *gradOutput_data = THCudaTensor_data(state, gradOutput);

  long no_elements = 1;
  for(int i = 0; i < gradInput->nDimension; i++){
    no_elements *= gradInput->size[i];
  }

  int d1;
  int d2;
  int d3;

  if (gradInput->nDimension == 3) {
    d1 = gradInput->size[0];
    d2 = gradInput->size[1];
    d3 = gradInput->size[2];
  } else {
    d1 = gradInput->size[1];
    d2 = gradInput->size[2];
    d3 = gradInput->size[3];
  }

  int D2;
  int D3;

  if (gradInput->nDimension == 3) {
    D2 = gradOutput->size[1];
    D3 = gradOutput->size[2];
  } else {
    D2 = gradOutput->size[2];
    D3 = gradOutput->size[3];
  }

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil((float)no_elements / (float)(n_xblocks * nthreads));
  if (n_yblocks > 65535) {
    THError("Input size is too large!  aborting");
  }
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  // kernel:
  upscale<<<blocks, threads, 0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, no_elements,
    dW, dH, iW, iH, d1, d2, d3, D2, D3);
  THCudaCheck(cudaGetLastError());

  return 1;
}


static const struct luaL_Reg cunnconv1d_SpatialSubSamplingPeriodic__ [] = {
   {"SpatialSubSamplingPeriodic_updateOutput", cunnconv1d_SpatialSubSamplingPeriodic_updateOutput},
   {"SpatialSubSamplingPeriodic_updateGradInput", cunnconv1d_SpatialSubSamplingPeriodic_updateGradInput},
   {NULL, NULL}
};


void cunnconv1d_SpatialSubSamplingPeriodic_init(lua_State *L)
{
   luaT_pushmetatable(L, "torch.CudaTensor");
   luaT_registeratname(L, cunnconv1d_SpatialSubSamplingPeriodic__, "nn");
   lua_pop(L,1);
}

