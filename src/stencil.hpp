#include <vector>
#include <cstring>
#include "field.hpp"
#include "codegen.hpp"
#include "stencil.decl.h"


class Stencil : public CBase_Stencil
{
private:
    int EPOCH;
    std::vector<char*> graph_cache;
    std::vector<uint32_t> graph_size;

public:
    Stencil_SDAG_CODE
    uint8_t name;
    
    // expects that the number of dimensions and length in each 
    // dimension will be specified at the time of creation
    uint32_t ndims;
    uint32_t odf; 
    
    std::vector<Field> fields;
    std::vector<uint32_t> ghost_depth;  // stores the depth of the ghosts corresponding to each field

    std::vector<uint32_t> local_size;
    std::vector<uint32_t> num_chares;
    std::vector<int> index;
    
    uint32_t start_x, start_y, start_z;
    std::vector<uint32_t> dims;

    Stencil(uint8_t name_, uint32_t ndims_, uint32_t* dims_, uint32_t odf_)
        : EPOCH(0)
        , name(name_)
        , ndims(ndims_)
        , odf(odf_)
    {
        index = std::vector<int>(3);

        index[0] = thisIndex.x;
        index[1] = thisIndex.y;
        index[2] = thisIndex.z;

        local_size = std::vector<uint32_t>(3, 0);
        num_chares = std::vector<uint32_t>(3, 1);
        dims = std::vector<uint32_t>(3, 0);

        for(int i = 0; i < ndims; i++)
            dims[i] = dims_[i];

        uint32_t total_chares = odf * CkNumPes();

        if(ndims == 1)
            num_chares[0] = total_chares; //std::min(odf, dims[0]);
        if(ndims == 2)
        {
            num_chares[0] = sqrt(total_chares); //std::min(odf, dims[0]);
            num_chares[1] = sqrt(total_chares); //std::min(odf, dims[1]);
        }
        if(ndims == 3)
        {
            num_chares[0] = cbrt(total_chares); //std::min(odf, dims[0]);
            num_chares[1] = cbrt(total_chares); //std::min(odf, dims[1]);
            num_chares[2] = cbrt(total_chares); //std::min(odf, dims[1]);
        }

        local_size[0] = dims[0] / num_chares[0];
        local_size[1] = dims[1] / num_chares[1];
        local_size[2] = dims[2] / num_chares[2];

        // for(int i = 0; i < ndims_; i++)
        // {   
        //     if(i == 0)
        //     {
        //         if(num_chares[0] == dims_[0])
        //         {             
        //             local_size[0] = 1;
        //             start_x = thisIndex.x;
        //         }
        //         else if(thisIndex.x < dims_[0] % num_chares[0])
        //         {
        //             local_size[0] = (dims_[0] / num_chares[0]) + 1;
        //             start_x = thisIndex.x * local_size[0];
        //         }
        //         else
        //         {
        //             local_size[0] = (dims_[0] / num_chares[0]);
        //             start_x = std::min(dims_[0] % num_chares[0], (uint32_t) thisIndex.x) * 
        //                 (local_size[0] + 1) + std::max(0,
        //                         (int) (thisIndex.x - dims_[0] % num_chares[0])) * local_size[0];
        //         }
        //     }
        //     else if(i == 1)
        //     {
        //         if(num_chares[1] == dims_[1])
        //         {             
        //             local_size[1] = 1;
        //             start_y = thisIndex.y;
        //         }
        //         else if(thisIndex.y < dims_[1] % num_chares[1])
        //         {
        //             local_size[1] = (dims_[1] / num_chares[1]) + 1;
        //             start_y = thisIndex.y * local_size[1];
        //         }
        //         else
        //         {
        //             local_size[1] = (dims_[1] / num_chares[1]);
        //             start_y = std::min(dims_[1] % num_chares[1], (uint32_t) thisIndex.y) *
        //                 (local_size[1] + 1) + std::max(0,
        //                         (int) (thisIndex.y - dims_[1] % num_chares[1])) * local_size[1];
        //         }
        //     }
        //     else
        //     {
        //         if(num_chares[2] == dims_[2])
        //         {             
        //             local_size[2] = 1;
        //             start_z = thisIndex.z;
        //         }
        //         else if(thisIndex.z < dims_[2] % num_chares[2])
        //         {
        //             local_size[2] = (dims_[2] / num_chares[2]) + 1;
        //             start_z = thisIndex.z * local_size[2];
        //         }
        //         else
        //         {
        //             local_size[2] = (dims_[2] / num_chares[2]);
        //             start_z = std::min(dims_[2] % num_chares[2], (uint32_t) thisIndex.z) *
        //                 (local_size[2] + 1) + std::max(0,
        //                         (int) (thisIndex.z - dims_[2] % num_chares[2])) * local_size[2];
        //         }
        //     }
        // }

        thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start();
    }

    Stencil(CkMigrateMessage* m) {}

    ~Stencil()
    {
        delete_cache();
    }

    void delete_cache()
    {
        for (char* cmd : graph_cache)
            free(cmd);
    }

    void execute_graph(int epoch, int size, char* cmd)
    {
#ifndef NDEBUG
        CkPrintf("execute_graph called\n");
#endif

        // cache the graphs
        uint8_t num_graphs = extract<uint8_t>(cmd);

#ifndef NDEBUG
        CkPrintf("Caching %" PRIu8 " graphs\n", num_graphs);
#endif

        //unique_graphs moved to graph_cache
        for (int i = 0; i < num_graphs; i++)
        {
            uint32_t cmd_size = extract<uint32_t>(cmd);
            char* cmd_copy = (char*) malloc(cmd_size);
            std::memcpy(cmd_copy, cmd, cmd_size);
            graph_cache.push_back(cmd_copy);
            graph_size.push_back(cmd_size);
            cmd += cmd_size;
        }

        uint32_t num_iters = extract<uint32_t>(cmd);

#ifndef NDEBUG
        CkPrintf("Num iterations = %u\n", num_iters);
#endif

        for (int i = 0; i < num_iters; i++)
        {
            uint8_t gid = extract<uint8_t>(cmd);
#ifndef NDEBUG
        CkPrintf("Running graph %u\n", gid);
#endif
            char* graph = graph_cache[gid];
            uint32_t cmd_size = graph_size[gid];
            
            bool generate_code = extract<bool>(graph);
             
            // FIXME this probably moves the pointer in cache?
            if (generate_code)
            {
                compute_fun_t compute_f = get_compute_fun(
                        graph, cmd_size, ndims, ghost_depth, local_size, num_chares);
#ifndef NDEBUG
                CkPrintf("Calling compute function\n");
#endif
                compute_f(fields, num_chares, index);
            }
            else
                interpret_graph(graph);
        }
    }

    void interpret_graph(char* cmd)
    {
        //OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
        Operation oper = lookup_operation(extract<uint8_t>(cmd));

#ifndef NDEBUG
        CkPrintf("interpret_graph called\n");
#endif

        switch (oper)
        {
            case Operation::create:
            {
                uint8_t fname = extract<uint8_t>(cmd);
                uint8_t depth = extract<uint8_t>(cmd);
                uint32_t total_local_size = 1;
#ifndef NDEBUG
                CkPrintf("Create field %" PRIu8 " with depth %" PRIu8 "\n", fname, depth);
#endif
                if (fname >= fields.size())
                {
                    fields.resize(fname + 1);
                    ghost_depth.resize(fname + 1);
                }

                for (int i = 0; i < ndims; i++)
                    total_local_size *= (local_size[i] + 
                            ((index[i] == 0 || index[i] == num_chares[i] - 1) ? depth : 2 * depth));

                fields[fname] = Field(total_local_size);
                ghost_depth[fname] = (uint32_t) depth;
            }
        }
    }
    
    void receiveGhost(int field_name,int dir,int size,int* res){
        
    }
};

#include "stencil.def.h"
