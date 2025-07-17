#pragma once

#include <chrono>
#include <string>

// Enum to represent different logging severities
enum class LogLevel {
    Debug,          // Most verbose
    Info,           // Standard informational messages
    InfoImportant,  // Important informational messages
    Warn,           // Warnings about potential issues
    Error,          // Errors that have occurred
    Off             // Disable all logging
};

// ANSI color codes for terminal output
namespace colors {
inline const char* CYAN = "\033[36m";
inline const char* WHITE = "\033[37m";
inline const char* GREEN = "\033[32m";
inline const char* YELLOW = "\033[33m";
inline const char* RED = "\033[31m";
inline const char* RESET = "\033[0m";
}  // namespace colors

// Helper to convert LogLevel enum to a string for printing.
// Moved outside the class to be accessible by both Timer and Logger.
inline const char* level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "[DEBUG]";
        case LogLevel::Info:
            return "[INFO]";
        case LogLevel::InfoImportant:
            return "[INFO IMPORTANT]";
        case LogLevel::Warn:
            return "[WARN]";
        case LogLevel::Error:
            return "[ERROR]";
        default:
            return "";
    }
}

// Helper to get the color for a given log level.
// Moved outside the class to be accessible by both Timer and Logger.
inline const char* level_to_color(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return colors::CYAN;
        case LogLevel::Info:
            return colors::WHITE;
        case LogLevel::InfoImportant:
            return colors::GREEN;
        case LogLevel::Warn:
            return colors::YELLOW;
        case LogLevel::Error:
            return colors::RED;
        default:
            return colors::RESET;
    }
}

class Logger {
   public:
    // The global log level. Messages with a level >= this will print.
    inline static LogLevel global_log_level = LogLevel::Info;

    // Static function to set the global level from anywhere in the code.
    static void set_global_log_level(LogLevel level) { global_log_level = level; }

    // Simplified logging methods that take a single string.
    static void debug(const std::string& message) { log(LogLevel::Debug, message); }
    static void info(const std::string& message) { log(LogLevel::Info, message); }
    static void info_important(const std::string& message) { log(LogLevel::InfoImportant, message); }
    static void warn(const std::string& message) { log(LogLevel::Warn, message); }
    static void error(const std::string& message) { log(LogLevel::Error, message); }

   private:
    // Private helper function that performs the logging.
    static void log(LogLevel level, const std::string& message) {
        // Only log if the message level is at or above the global level.
        if (level >= global_log_level && global_log_level != LogLevel::Off) {
            // Using fputs, fputc and fprintf for more efficient output
            fputs(level_to_color(level), stdout);   // Set the color
            fputs(level_to_string(level), stdout);  // Print the log level
            fputc(' ', stdout);
            fputs(message.c_str(), stdout);
            fputs(colors::RESET, stdout);
            fputc('\n', stdout);
        }
    }
};

class Timer {
   public:
    // The global log level. Timers with a level >= this will print.
    // Defaults to Off. Set to Info for standard timers, Debug for all timers.
    inline static LogLevel global_log_level = LogLevel::Info;

    // Static function to set the global level from anywhere in the code.
    static void set_global_log_level(LogLevel level) { global_log_level = level; }

    // Constructor now takes an optional LogLevel (defaults to Info)
    Timer(const std::string& function_name, LogLevel level = LogLevel::Info)
        : function_name(function_name), level(level) {
        if (level >= global_log_level && global_log_level != LogLevel::Off) {
            start = std::chrono::high_resolution_clock::now();
        }
    }

    ~Timer() {
        // Only print if the timer's level is at or above the global level
        if (level >= global_log_level && global_log_level != LogLevel::Off) {
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            const float seconds = duration.count();
            // Using fputs, fputc and fprintf for more efficient output
            fputs(level_to_color(level), stdout);   // Set the color
            fputs(level_to_string(level), stdout);  // Print the log level
            fputs(" [TIMER] ", stdout);
            fputs(function_name.c_str(), stdout);
            fprintf(stdout, " took %.3f ms", seconds * 1000.0f);
            fputs(colors::RESET, stdout);
            fputc('\n', stdout);
        }
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;
    std::string function_name;
    LogLevel level;  // The log level of this specific timer instance
};
