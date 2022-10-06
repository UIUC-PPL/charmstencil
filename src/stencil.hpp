#include <vector>
#include <cstring>
#include "codegen.hpp"
#include "stencil.decl.h"


class Field
{
public:
    std::vector<double> data;
    Field(uint32_t data_size)
    {
        data.reserve(data_size);
    }
};


class Stencil : public CBase_Stencil
{
private:
    int EPOCH;
    std::vector<char*> graph_cache;

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

        if(ndims > 0)
            num_chares[0] = std::min(odf, dims_[0]);
        if(ndims > 1)
            num_chares[1] = std::min(odf, dims_[1]);
        if(ndims > 2)
            num_chares[2] = std::min(odf, dims_[2]);

        for(int i = 0; i < ndims_; i++)
        {   
            if(i == 0)
            {
                if(num_chares[0] == dims_[0])
                {             
                    local_size[0] = 1;
                    start_x = thisIndex.x;
                }
                else if(thisIndex.x < dims_[0] % num_chares[0])
                {
                    local_size[0] = (dims_[0] / num_chares[0]) + 1;
                    start_x = thisIndex.x * local_size[0];
                }
                else
                {
                    local_size[0] = (dims_[0] / num_chares[0]);
                    start_x = std::min(dims_[0] % num_chares[0], (uint32_t) thisIndex.x) * 
                        (local_size[0] + 1) + std::max(0,
                                (int) (thisIndex.x - dims_[0] % num_chares[0])) * local_size[0];
                }
            }
            else if(i == 1)
            {
                if(num_chares[1] == dims_[1])
                {             
                    local_size[1] = 1;
                    start_y = thisIndex.y;
                }
                else if(thisIndex.y < dims_[1] % num_chares[1])
                {
                    local_size[1] = (dims_[1] / num_chares[1]) + 1;
                    start_y = thisIndex.y * local_size[1];
                }
                else
                {
                    local_size[1] = (dims_[1] / num_chares[1]);
                    start_y = std::min(dims_[1] % num_chares[1], (uint32_t) thisIndex.y) *
                        (local_size[1] + 1) + std::max(0,
                                (int) (thisIndex.y - dims_[1] % num_chares[1])) * local_size[1];
                }
            }
            else
            {
                if(num_chares[2] == dims_[2])
                {             
                    local_size[2] = 1;
                    start_z = thisIndex.z;
                }
                else if(thisIndex.z < dims_[2] % num_chares[2])
                {
                    local_size[2] = (dims_[2] / num_chares[2]) + 1;
                    start_z = thisIndex.z * local_size[2];
                }
                else
                {
                    local_size[2] = (dims_[2] / num_chares[2]);
                    start_z = std::min(dims_[2] % num_chares[2], (uint32_t) thisIndex.z) *
                        (local_size[2] + 1) + std::max(0,
                                (int) (thisIndex.z - dims_[2] % num_chares[2])) * local_size[2];
                }
            }
        }

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
        // cache the graphs
        uint8_t num_graphs = extract<uint8_t>(cmd);

        //unique_graphs moved to graph_cache
        for (int i = 0; i < num_graphs; i++)
        {
            uint32_t cmd_size = extract<uint32_t>(cmd);
            //generate(cmd, ndims, ghost_depth, local_size, index, num_chares);
            // char* cmd_copy = (char*) malloc(cmd_size);
            // std::memcpy(cmd_copy, cmd, cmd_size);
            // graph_cache.push_back(cmd_copy);
            // cmd += cmd_size;
        }

        // uint32_t num_iters = extract<uint32_t>(cmd);
        // for (int i = 0; i < num_iters; i++)
        // {
        //     uint8_t gid = extract<uint8_t>(cmd);
        //     char* graph = graph_cache[gid];
        //     // FIXME this probably moves the pointer in cache?
        //     traverse_graph(graph);
        // }
    }

    
    void receiveGhost(int field_name,int dir,int size,int* res){
        
    }
};

#include "stencil.def.h"
