#include "Timer.h"

void start_timer()
{
	start = std::chrono::system_clock::now();
}

void stop_timer()
{
	end = std::chrono::system_clock::now();
}

double get_microseconds()
{
	std::chrono::duration<double> elapsed_seconds = end - start;
	return (double)std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds).count();
}