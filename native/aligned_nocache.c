/* Benchmark-only, bounded aligned SSD reader. Original model and executable unchanged. */
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
static _Atomic unsigned long long logical_bytes, aligned_bytes, calls, errors;
static int under(const char *path,const char *root) {
    if(!root || !*root)return 0;
    size_t n=strlen(root); return !strncmp(path,root,n) && (path[n]=='/' || path[n]==0);
}
static int target(int fd) {
    char path[4096];if(fcntl(fd,F_GETPATH,path))return 0;
    const char *roots=getenv("IMAGE_MLX_UNCACHED_ROOTS");
    if(!roots)return 0;
    while(*roots){
        const char *end=strchr(roots,'\n');size_t n=end?(size_t)(end-roots):strlen(roots);
        if(n && !strncmp(path,roots,n) && (path[n]=='/' || !path[n]))return 1;
        if(!end)break;roots=end+1;
    }
    return 0;
}
static ssize_t fail(int error) {atomic_fetch_add(&errors,1);errno=error;return -1;}
static ssize_t aligned_read(int fd,void *buffer,size_t size,off_t offset) {
    atomic_fetch_add(&calls,1);
    if(fcntl(fd,F_NOCACHE,1)||fcntl(fd,F_RDAHEAD,0))return fail(errno);
    if(offset<0 || size>SSIZE_MAX)return fail(EINVAL);
    if(!size)return 0;
    size_t page=(size_t)getpagesize(), cap=8*1024*1024,done=0;void *raw=NULL;
    int err=posix_memalign(&raw,page,cap);if(err)return fail(err);
    while(done<size){
        if((uint64_t)offset+done>INT64_MAX){err=EOVERFLOW;break;}
        off_t at=offset+done,base=at&~(off_t)(page-1);
        size_t skip=(size_t)(at-base),n=size-done;if(n>cap-skip)n=cap-skip;
        size_t request=(skip+n+page-1)&~(page-1);
        ssize_t got;do {got=pread(fd,raw,request,base);}while(got<0 && errno==EINTR);
        if(got<0){err=errno;break;}
        atomic_fetch_add(&aligned_bytes,(unsigned long long)got);
        if(got<=(ssize_t)skip)break;
        size_t available=(size_t)got-skip;if(n>available)n=available;
        memcpy((char*)buffer+done,(char*)raw+skip,n);done+=n;
        if((size_t)got<request)break;
    }
    free(raw);atomic_fetch_add(&logical_bytes,done);
    if(err){atomic_fetch_add(&errors,1);errno=err;if(!done)return -1;}
    return (ssize_t)done;
}
static ssize_t strict_pread(int fd,void *buffer,size_t size,off_t offset) {
    int saved=errno,active=target(fd);errno=saved;
    return active?aligned_read(fd,buffer,size,offset):pread(fd,buffer,size,offset);
}
static ssize_t strict_preadv(int fd,const struct iovec *iov,int count,off_t offset) {
    int saved=errno,active=target(fd);errno=saved;
    if(!active)return preadv(fd,iov,count,offset);
    if(count<0 || count>IOV_MAX)return fail(EINVAL);
    size_t total=0;for(int i=0;i<count;i++){if(iov[i].iov_len>SSIZE_MAX-total)return fail(EINVAL);total+=iov[i].iov_len;}
    size_t done=0;for(int i=0;i<count;i++){
        if(offset<0 || (uint64_t)offset+done>INT64_MAX)return fail(EINVAL);
        ssize_t n=aligned_read(fd,iov[i].iov_base,iov[i].iov_len,offset+done);
        if(n<0)return done?(ssize_t)done:-1;done+=n;if((size_t)n<iov[i].iov_len)break;
    }return (ssize_t)done;
}
__attribute__((used)) static struct {const void *replacement;const void *original;}
interpose[] __attribute__((section("__DATA,__interpose"))) = {
    {(const void*)strict_pread,(const void*)pread},
    {(const void*)strict_preadv,(const void*)preadv}
};
__attribute__((destructor))static void report(void){
    if(atomic_load(&calls))fprintf(stderr,"STRICT_IO calls=%llu logical_bytes=%llu aligned_bytes=%llu errors=%llu\n",atomic_load(&calls),atomic_load(&logical_bytes),atomic_load(&aligned_bytes),atomic_load(&errors));
}
