#include <sys/mman.h>  
#include <sys/stat.h>  
#include <fcntl.h>  
#include <stdio.h>  
#include <stdlib.h>  
#include <unistd.h>  
#include <error.h>  
#include<string.h>  
#include <stdbool.h>
#include <sys/time.h>

char *creatMMap(const char * filename,struct stat **sb);
char *readMMap(const char * filename, long start,long length) ;
bool writeMMap(const char * filename, char* src, bool sysn) ;

char * creatMMap(const char * filename,struct stat **sb)// failed return nullptr succ mpped;
{
    char *mapped=NULL;
    int fd=0;  
    if ((fd = open(filename,  O_RDONLY | __O_DIRECT)) < 0) {  
        perror("open");  
		return mapped;
    }  
   
    if ((fstat(fd, *sb)) == -1) {  
        perror("fstat");  
		return mapped;
    }  
    // printf memory map context;
    if ((mapped = (char *)mmap(NULL, (*sb)->st_size, PROT_READ , MAP_SHARED, fd, 0)) == (void *)-1) {  
        perror("mmap");  
		return mapped;
    }  
   
    close(fd); 

    return mapped;

}

char *readMMap(const char * filename, long start,long length) // start :开始位置，读取长度；
{   
    struct stat sb;  
    
    struct stat* insb = &sb;
    struct timeval tv;
    struct timeval tv1;
    gettimeofday(&tv, NULL);
    char *mapped = creatMMap(filename,&insb);
    gettimeofday(&tv1, NULL);

    printf("Loading time: %ld micro second", tv1.tv_usec-tv.tv_usec);
    // memcpy(dest,mapped+start,length);
     
    // printf("read =========================\n"); 
    
    // printf("%s\n",mapped); 
    
    // if ((munmap((void *)mapped, sb.st_size)) == -1) {  
    //     perror("munmap");  
    // }

    return mapped;
}

bool closemmap(char *mapped){
    struct stat sb;  
    
    struct stat* insb = &sb;

    if ((munmap((void *)mapped, sb.st_size)) == -1) {  
        perror("munmap");  
    }
}

bool writeMMap(const char * filename, char* src,bool sysn) // src :写入数据,是否同步到本地 sysn；st_size:同步的长度；
{
	struct stat sb;  
    
    struct stat* insb = &sb;
    
    char *mapped = creatMMap(filename,&insb);

    long st_size = sb.st_size;

    memcpy(mapped,(void*)src,strlen(src));
    
    
	if(sysn)
    if ((msync((void *)mapped, st_size, MS_SYNC)) == -1) {  
        perror("msync");  
		return false;
    }  

    if ((munmap((void *)mapped, sb.st_size)) == -1) {  
        perror("munmap");  
    }
    
	return true;
}
  
int main(int argc, char **argv)  
{  
    int fd, nread, i;  
    
    // read memory map   
    // char readmap[5000]={0};
    //printf("read =========================\n"); 
    //char * m = readMMap("/mnt/mem/cora.content", 0, 2000000);
    // printf("%s", m);
    // closemmap(m);

    // /* 修改,同步到磁盘文件 */  
    // char newStr[128]="123456\n";
    // writeMMap("./data.txt",newStr,true);
    //    // printf memory map context;
    // printf("show new =========================\n"); 
    // printf("%s",readMMap("./data.txt",readmap, 0,100));

    // printf("%s", mapped);  
    // printf("show new end =========================\n\n\n"); 

    // while(1)
    // {
    //    printf("show ========================= \n"); 
    //    printf("%s", mapped);
    //    sleep(5);
    // }
    /* 释放存储映射区 */

    return 0;  
}  
