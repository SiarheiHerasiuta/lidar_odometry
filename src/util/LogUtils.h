/**
 * @file      LogUtils.h
 * @brief     Lightweight colored logging utilities without spdlog dependency
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-12-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef LOG_UTILS_H
#define LOG_UTILS_H

#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstring>

namespace lidar_slam {

// ANSI color codes for terminal output
#define LIO_COLOR_RESET   "\033[0m"
#define LIO_COLOR_RED     "\033[31m"
#define LIO_COLOR_GREEN   "\033[32m"
#define LIO_COLOR_YELLOW  "\033[33m"
#define LIO_COLOR_CYAN    "\033[36m"
#define LIO_COLOR_GRAY    "\033[90m"
#define LIO_COLOR_BOLD    "\033[1m"

/**
 * @brief Log severity levels
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

/**
 * @brief Lightweight logger with colored output
 * 
 * Features:
 * - Colored output: D=cyan, I=green, W=yellow, E=red
 * - Timestamp: YYYY-MM-DD HH:MM:SS.mmm
 * - Format specifiers: {}, {:.2f}, {:>10.2f}, etc.
 * - Header-only implementation
 */
class Logger {
public:
    /// Current log level (messages below this level are suppressed)
    inline static LogLevel level = LogLevel::INFO;

    /// Log debug message
    template<typename... Args>
    static void debug(const char* fmt, Args... args) {
        if (level <= LogLevel::DEBUG) {
            print(LIO_COLOR_CYAN "[D]" LIO_COLOR_RESET, fmt, args...);
        }
    }

    /// Log info message
    template<typename... Args>
    static void info(const char* fmt, Args... args) {
        if (level <= LogLevel::INFO) {
            print(LIO_COLOR_GREEN "[I]" LIO_COLOR_RESET, fmt, args...);
        }
    }

    /// Log warning message
    template<typename... Args>
    static void warn(const char* fmt, Args... args) {
        if (level <= LogLevel::WARN) {
            print(LIO_COLOR_YELLOW "[W]" LIO_COLOR_RESET, fmt, args...);
        }
    }

    /// Log error message
    template<typename... Args>
    static void error(const char* fmt, Args... args) {
        if (level <= LogLevel::ERROR) {
            print(LIO_COLOR_RED LIO_COLOR_BOLD "[E]" LIO_COLOR_RESET, fmt, args...);
        }
    }

private:
    /// Get current timestamp string with milliseconds
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        std::tm tm_buf;
        localtime_r(&time, &tm_buf);
        
        std::ostringstream oss;
        oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    /// Format a value with format specifier like {:>10.2f}
    template<typename T>
    static void format_value(std::ostringstream& oss, const char* spec_start, const char* spec_end, T value) {
        // Parse format spec: {:>10.2f} -> fill='', align='>', width=10, precision=2, type='f'
        std::string spec(spec_start, spec_end);
        
        char fill = ' ';
        char align = '\0';
        int width = 0;
        int precision = -1;
        char type = '\0';
        
        size_t i = 0;
        
        // Check for alignment with optional fill
        if (spec.size() > 1 && (spec[1] == '<' || spec[1] == '>' || spec[1] == '^')) {
            fill = spec[0];
            align = spec[1];
            i = 2;
        } else if (spec.size() > 0 && (spec[0] == '<' || spec[0] == '>' || spec[0] == '^')) {
            align = spec[0];
            i = 1;
        }
        
        // Parse width
        while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9') {
            width = width * 10 + (spec[i] - '0');
            i++;
        }
        
        // Parse precision
        if (i < spec.size() && spec[i] == '.') {
            i++;
            precision = 0;
            while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9') {
                precision = precision * 10 + (spec[i] - '0');
                i++;
            }
        }
        
        // Parse type
        if (i < spec.size()) {
            type = spec[i];
        }
        
        // Apply formatting
        std::ostringstream temp;
        if (precision >= 0) {
            temp << std::fixed << std::setprecision(precision);
        }
        temp << value;
        std::string result = temp.str();
        
        // Apply width and alignment
        if (width > 0 && result.size() < static_cast<size_t>(width)) {
            size_t padding = width - result.size();
            if (align == '<') {
                oss << result << std::string(padding, fill);
            } else if (align == '^') {
                size_t left = padding / 2;
                size_t right = padding - left;
                oss << std::string(left, fill) << result << std::string(right, fill);
            } else { // '>' or default (right align)
                oss << std::string(padding, fill) << result;
            }
        } else {
            oss << result;
        }
    }

    /// Find closing brace and return pointer to it
    static const char* find_closing_brace(const char* fmt) {
        while (*fmt && *fmt != '}') fmt++;
        return fmt;
    }

    /// Print message without format arguments
    static void print(const char* prefix, const char* fmt) {
        std::cout << LIO_COLOR_GRAY "[" << get_timestamp() << "]" LIO_COLOR_RESET 
                  << " " << prefix << " " << fmt << std::endl;
    }

    /// Print message with format arguments
    template<typename T, typename... Args>
    static void print(const char* prefix, const char* fmt, T first, Args... rest) {
        std::ostringstream oss;
        oss << LIO_COLOR_GRAY "[" << get_timestamp() << "]" LIO_COLOR_RESET 
            << " " << prefix << " ";
        
        while (*fmt) {
            if (*fmt == '{') {
                const char* spec_start = fmt + 1;
                const char* close = find_closing_brace(spec_start);
                if (*close == '}') {
                    if (spec_start == close) {
                        // Simple {} - no format spec
                        oss << first;
                    } else if (*spec_start == ':') {
                        // Format spec like {:>10.2f}
                        format_value(oss, spec_start + 1, close, first);
                    } else {
                        // Just output as-is
                        oss << first;
                    }
                    print_rest(oss, close + 1, rest...);
                    return;
                }
            }
            oss << *fmt++;
        }
        std::cout << oss.str() << std::endl;
    }

    /// Continue formatting remaining arguments
    static void print_rest(std::ostringstream& oss, const char* fmt) {
        oss << fmt;
        std::cout << oss.str() << std::endl;
    }

    template<typename T, typename... Args>
    static void print_rest(std::ostringstream& oss, const char* fmt, T first, Args... rest) {
        while (*fmt) {
            if (*fmt == '{') {
                const char* spec_start = fmt + 1;
                const char* close = find_closing_brace(spec_start);
                if (*close == '}') {
                    if (spec_start == close) {
                        oss << first;
                    } else if (*spec_start == ':') {
                        format_value(oss, spec_start + 1, close, first);
                    } else {
                        oss << first;
                    }
                    print_rest(oss, close + 1, rest...);
                    return;
                }
            }
            oss << *fmt++;
        }
        std::cout << oss.str() << std::endl;
    }
};

} // namespace lidar_slam

// ===== Convenience Macros =====

#define LOG_DEBUG(...) lidar_slam::Logger::debug(__VA_ARGS__)
#define LOG_INFO(...)  lidar_slam::Logger::info(__VA_ARGS__)
#define LOG_WARN(...)  lidar_slam::Logger::warn(__VA_ARGS__)
#define LOG_ERROR(...) lidar_slam::Logger::error(__VA_ARGS__)

#endif // LOG_UTILS_H
