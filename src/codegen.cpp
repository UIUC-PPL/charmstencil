#include "codegen.hpp"

#include <string>
#include <sstream>
#include <algorithm>

std::string format_cpp(const std::string& input) {
    std::stringstream formattedCode;
    std::stringstream inputStringStream(input);
    std::string line;
    int indentLevel = 0;
    const int indentSize = 4;

    while (std::getline(inputStringStream, line)) {
        // Remove leading/trailing whitespace
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), line.end());

        // Adjust indent level based on curly braces
        if (line.find('{') != std::string::npos) {
            for (int i = 0; i < indentLevel * indentSize; ++i) {
                formattedCode << ' ';
            }
            formattedCode << line << '\n';
            indentLevel++;
        } else if (line.find('}') != std::string::npos) {
            indentLevel--;
            for (int i = 0; i < indentLevel * indentSize; ++i) {
                formattedCode << ' ';
            }
            formattedCode << line << '\n';
        } else {
            // Apply indentation
            for (int i = 0; i < indentLevel * indentSize; ++i) {
                formattedCode << ' ';
            }
            formattedCode << line << '\n';
        }
    }

    return formattedCode.str();
}

static std::string get_kernel_header()
{
    return "#include <iostream>\n"
           "#include <vector>\n"
           "#include <cuda.h>\n\n"
           "#define IDX2D(y, x, stride) ((y) * (stride) + (x))\n\n";
}

Context* write_kernel(FILE* genfile, Kernel* knl)
{
    Context* ctx = new Context();
    std::string code = format_cpp(get_kernel_header() + knl->generate_code(ctx));
    fprintf(genfile, code.c_str());
    return ctx;
}

void generate_kernel(Kernel* knl, int suffix)
{
    std::string filename = fmt::format("generated/kernel_{}_{}", knl->hash, suffix);
    DEBUG_PRINT("Generating kernel %s\n", filename.c_str());
    FILE* genfile = fopen((filename + ".cu").c_str(), "w");
    knl->context = write_kernel(genfile, knl);
    fclose(genfile);

    // compile filename
    std::string compile_cmd = fmt::format(
            "nvcc -std=c++11 -arch sm_60 --ptx -o {}.ptx {}.cu -O3 -Xptxas -O3 -g -lcuda", 
            filename, filename);

    system(compile_cmd.c_str());
}