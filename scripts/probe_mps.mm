#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv) { @autoreleasepool {
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue=[dev newCommandQueue];
  int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]);
  bool integer=atoi(argv[4]), f32=atoi(argv[5]);
  MPSDataType src=integer?MPSDataTypeInt8:(f32?MPSDataTypeFloat32:MPSDataTypeFloat16);
  MPSDataType dst=integer?MPSDataTypeInt32:MPSDataTypeFloat32;
  int bytes=integer?1:(f32?4:2);
  auto input=[&](int rows,int cols,MPSDataType type,int sz){
    auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:cols*sz dataType:type];
    auto buf=[dev newBufferWithLength:size_t(rows)*cols*sz options:MTLResourceStorageModeShared];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
  };
  MPSMatrix *x=input(m,k,src,bytes), *w=input(n,k,src,bytes), *y=input(m,n,dst,4);
  for(MPSMatrix *mat in @[x,w]){
    void* p=mat.data.contents;
    for(size_t i=0;i<mat.rows*mat.columns;++i) {
      int a=int(i%7)-3;
      if(integer)((int8_t*)p)[i]=a;else if(f32)((float*)p)[i]=a;else((__fp16*)p)[i]=a;
    }
  }
  MPSMatrixMultiplication* mm=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n interiorColumns:k alpha:1 beta:0];
  for(int j=0;j<4;j++) {
    auto start=std::chrono::steady_clock::now(); auto cmd=[queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:x rightMatrix:w resultMatrix:y];
    [cmd commit]; [cmd waitUntilCompleted]; if(cmd.error){NSLog(@"%@",cmd.error);return 3;}
    int expected=0;for(int i=0;i<k;i++){int a=i%7-3;expected+=a*a;}
    double out=integer?((int32_t*)y.data.contents)[0]:((float*)y.data.contents)[0];
    printf("MPS %d %d %d %d %d %.2f expected %d actual %.1f\n",m,n,k,integer,f32,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(),expected,out);fflush(stdout);
    if(out!=expected)return 2;
  }
}}
