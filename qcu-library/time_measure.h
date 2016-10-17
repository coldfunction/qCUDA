#include <sys/time.h>


#if 1
enum
{
t_RegFatbin = 0,
t_UnregFatbin,
t_RegFunc,
t_Launch,
t_ConfigCall,
t_SetArg,
t_Malloc,
t_MemcpyH2D,
t_MemcpyD2H,
t_Free,
t_GetDev,
t_GetDevCount,
t_SetDev,
t_GetDevProp,
t_DevSync,
t_DevReset,
t_DriverGetVersion,
t_RuntimeGetVersion,
t_EventCreate,
t_EventRecord,
t_EventSync,
t_EventElapsedTime,
t_EventDestroy,
t_GetLastError,
t_myMalloc,
FUNC_MAX_IDX
};

FILE *f;
struct timeval timeval_begin, timeval_end;
unsigned int time_measure[FUNC_MAX_IDX][2] = {{0}};
int t_idx;

#define print(fmt, arg...) \
	fprintf(f, fmt, ##arg)

#define time_init() \
	f = fopen("/tmp/libcudart_dl_out", "a")

#define time_fini() \
	for(t_idx=0; t_idx<FUNC_MAX_IDX; t_idx++){ \
		if(time_measure[t_idx][1]){ \
			print("%.2f ", (double)time_measure[t_idx][0]/time_measure[t_idx][1]); \
		}else{\
			print("0 ");\
		}\
	}\
	print("\n");\
	fclose(f)


#define time_begin() gettimeofday (&timeval_begin, NULL)
#define time_end(idx) 	\
	gettimeofday (&timeval_end, NULL); \
	time_measure[idx][0] += \
		(unsigned int)((timeval_end.tv_sec  - timeval_begin.tv_sec)*1000000 + \
					   (timeval_end.tv_usec - timeval_begin.tv_usec)); \
	time_measure[idx][1] ++;
	
#else
#define print(fmt, arg...) 
#define time_begin()
#define time_end(str)
#endif
