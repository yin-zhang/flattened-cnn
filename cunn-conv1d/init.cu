#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "LateralMaskedConvolution.cu"
#include "LateralConvolution.cu"
#include "VerticalConvolution.cu"
#include "HorizontalConvolution.cu"
#include "PlanarConvolution.cu"
#include "SpatialUpSamplingPeriodic.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunnconv1d(lua_State *L);

int luaopen_libcunnconv1d(lua_State *L)
{
  lua_newtable(L);

  cunnconv1d_LateralMaskedConvolution_init(L);
  cunnconv1d_LateralConvolution_init(L);
  cunnconv1d_VerticalConvolution_init(L);
  cunnconv1d_HorizontalConvolution_init(L);
  cunnconv1d_PlanarConvolution_init(L);
  cunnconv1d_SpatialUpSamplingPeriodic_init(L);

  return 1;
}
