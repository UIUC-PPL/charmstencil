#include "codegen.hpp"

static std::string get_kernel_header()
{
    return "#include <iostream>\n"
           "#include <vector>\n"
           "#include <cuda.h>\n\n";
}

void write_kernel(FILE* genfile, Kernel* knl)
{
    fprintf(genfile, get_kernel_header().c_str());

    Context* ctx = new Context();
    fprintf(genfile, knl->generate_code(ctx).c_str());
}

void generate_kernel(Kernel* knl, int suffix)
{
    std::string filename = fmt::format("kernel_{}_{}", knl->kernel_id, suffix);
    FILE* genfile = fopen((filename + ".cu").c_str(), "w");
    write_kernel(genfile, knl);
    fclose(genfile);

    // compile filename
    std::string compile_cmd = fmt::format(
            "nvcc -std=c++11 -arch sm_60 --ptx -o {}.ptx {}.cu -I$STENCIL_PATH -I$CHARM_PATH -O3 -g -lcuda", 
            filename, filename);

    system(compile_cmd.c_str());
}