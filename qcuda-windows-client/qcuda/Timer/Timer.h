#pragma once
#include <iostream>
#include <chrono>
#include <ctime>

std::chrono::time_point<std::chrono::system_clock> start, end;

extern "C" {
	void start_timer();
	void stop_timer();
	double get_microseconds();
}
