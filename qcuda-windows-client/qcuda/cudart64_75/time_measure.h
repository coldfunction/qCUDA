#include <time.h>
#include <Windows.h>
#include <stdarg.h>

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

extern void start_timer();
extern void stop_timer();
extern double get_microseconds();

FILE *f;
// struct timeval timeval_begin, timeval_end;
double time_measure[FUNC_MAX_IDX][2] = { { 0.0 } };
int t_idx;
LARGE_INTEGER timeval_begin = { 0 }, timeval_end = { 0 }, freq = { 0 };
double ElapsedTime = 0.0;

#define print(fmt, ...) \
	fprintf(f, fmt, ##__VA_ARGS__);

#define time_init() \
	f = fopen("C:\\Users\\sslabuser\\Documents\\libcudart_dl_out.txt", "a") // "/tmp/libcudart_dl_out"

#define time_fini() \
	for (t_idx = 0; t_idx < FUNC_MAX_IDX; t_idx++){ \
		if (time_measure[t_idx][1]){ \
			print("%.2f ", (time_measure[t_idx][0] / time_measure[t_idx][1])); \
		} else { \
			print("0 "); \
		}\
	}\
	print("\n"); \
	fclose(f)

// MSVC defines this in winsock2.h!?
//typedef struct timeval {
//	long tv_sec;
//	long tv_usec;
//} timeval;

// not defined in windows :(
//int gettimeofday(struct timeval * tp, struct timezone * tzp)
//{
//	FILETIME    file_time;
//	SYSTEMTIME  system_time;
//	ULARGE_INTEGER ularge;
//	const unsigned __int64 epoch = ((uint64_t)116444736000000000ULL);
//
//	GetSystemTime(&system_time);
//	SystemTimeToFileTime(&system_time, &file_time);
//	ularge.LowPart = file_time.dwLowDateTime;
//	ularge.HighPart = file_time.dwHighDateTime;
//
//	tp->tv_sec = (long)((ularge.QuadPart - epoch) / 10000000L);
//	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
//
//	return 0;
//}

#define time_begin() start_timer();
#define time_end(idx) 	\
	stop_timer(); \
	ElapsedTime = get_microseconds(); \
	time_measure[idx][0] += ElapsedTime; \
	time_measure[idx][1] ++;

#else
#define print(fmt, arg...) 
#define time_begin()
#define time_end(str)
#endif
