#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
static BNNSNDArrayDescriptor tensor(size_t rows,size_t cols,BNNSDataType type,unsigned bytes){
    BNNSNDArrayDescriptor t={0};
    t.layout=BNNSDataLayoutRowMajorMatrix;t.size[0]=cols;t.size[1]=rows;
    t.stride[0]=1;t.stride[1]=cols;t.data_type=type;t.data_scale=1;
    t.data=calloc(rows*cols,bytes);
    for(size_t i=0;i<rows*cols;++i){
        int v=i%7-3;
        if(type==BNNSDataTypeInt8)((int8_t*)t.data)[i]=v;
        if(type==BNNSDataTypeFloat16)((__fp16*)t.data)[i]=v;
    }
    return t;
}
int main(int argc,char **argv){
    int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),f=atoi(argv[4]),fi=atoi(argv[5]);
    BNNSNDArrayDescriptor a=tensor(m,k,f?BNNSDataTypeFloat16:BNNSDataTypeInt8,f?2:1),b=tensor(n,k,f?BNNSDataTypeFloat16:BNNSDataTypeInt8,f?2:1),y=tensor(m,n,fi?BNNSDataTypeFloat32:BNNSDataTypeInt32,4);
    ssize_t size=BNNSMatMulWorkspaceSize(false,true,1,&a,&b,&y,NULL);
    printf("format %d %d workspace %zd\n",f,fi,size);fflush(stdout);
    if(size<0)return 1;
    void *scratch=calloc(1,size);
    for(int j=0;j<4;j++){
        uint64_t start=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        int rc=BNNSMatMul(false,true,1,&a,&b,&y,scratch,NULL);
        int expected=0;for(int i=0;i<k;i++){int v=i%7-3;expected+=v*v;}
        double value=fi?((float*)y.data)[0]:((int32_t*)y.data)[0];
        printf("BNNS %d %d %.2f ms expected %d result %.1f status %d\n",f,fi,(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)-start)/1e6,expected,value,rc);fflush(stdout);
        if(rc||value!=expected)return 2;
    }
}
