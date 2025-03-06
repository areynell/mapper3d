#pragma once

#include <chrono>
#include <iostream>
#include <string>

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
    std::string function_name;

    Timer(const std::string &function_name) : function_name(function_name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        const float seconds = duration.count();
        std::cout << function_name << " took " << seconds * 1000 << " ms\n";
    }
};
