#ifndef SERVER_H
#define SERVER_H

#include <string>
#include <vector>
#include <functional>

// Placeholder for HTTP server library - can be replaced with any C++ HTTP library
// This is intentionally kept simple to avoid exact copying

namespace http {
    struct Request {
        std::string method;
        std::string path;
        std::string query_string;
    };
    
    struct Response {
        int status;
        std::string content_type;
        std::string body;
    };
}

#endif // SERVER_H
