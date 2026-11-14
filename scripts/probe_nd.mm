#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv){@autoreleasepool{
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> q=[dev newCommandQueue];
 int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]);bool integer=atoi(argv[4]),left=atoi(argv[5]);
 auto data=[&](int r,int c,MPSDataType type,int bytes,bool transpose){
   auto desc=[MPSNDArrayDescriptor descriptorWithDataType:type shape:@[@(r),@(c)]];
   if(transpose)[desc transposeDimension:0 withDimension:1];
   auto buf=[dev newBufferWithLength:size_t(r)*c*bytes options:MTLResourceStorageModeShared];
   void *p=buf.contents;
   for(size_t i=0;i<size_t(r)*c;i++){
      int v=i%7-3;
      if(type==MPSDataTypeInt8)((int8_t*)p)[i]=v;
      else if(type==MPSDataTypeFloat32)((float*)p)[i]=v;
   }
   return [[MPSNDArray alloc] initWithBuffer:buf offset:0 descriptor:desc];
 };
 auto a=data(m,k,left?MPSDataTypeInt8:MPSDataTypeFloat32,left?1:4,false);
 auto b=data(n,k,MPSDataTypeInt8,1,true);
 auto y=data(m,n,integer?MPSDataTypeInt32:MPSDataTypeFloat32,4,false);
 auto scale=data(1,1,MPSDataTypeFloat32,4,false);float unit=1;
 [scale writeBytes:&unit strideBytes:nullptr];
 auto desc=[[MPSNDArrayAffineQuantizationDescriptor alloc] initWithDataType:MPSDataTypeInt8 hasZeroPoint:NO hasMinValue:NO];
 auto mm=[[MPSNDArrayQuantizedMatrixMultiplication alloc] initWithDevice:dev leftQuantizationDescriptor:left?desc:nil rightQuantizationDescriptor:desc];
 for(int i=0;i<4;i++){
    auto st=std::chrono::steady_clock::now();
    auto cmd=[q commandBuffer];[mm encodeToCommandBuffer:cmd sourceArrays:left?@[a,b,scale,scale]:@[a,b,scale] resultState:nil destinationArray:y];[cmd commit];[cmd waitUntilCompleted];if(cmd.error){NSLog(@"%@",cmd.error);return 3;}
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count();
    void* results=malloc(size_t(m)*n*4);[y readBytes:results strideBytes:nullptr];double v=integer?((int32_t*)results)[0]:((float*)results)[0];free(results);
    int expected=0;for(int j=0;j<k;j++){int a=j%7-3;expected+=a*a;}
    printf("ND %d %d %d %d %d %.2f ms expected %d actual %.1f\n",m,n,k,integer,left,ms,expected,v);fflush(stdout);if(v!=expected)return 2;
 }
}}
