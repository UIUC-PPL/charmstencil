#include <vector>
#include <cstring>
#include "field.hpp"
#include "codegen.hpp"
#include "stencil.decl.h"


#define DOWN 0
#define UP 1
#define LEFT 2
#define RIGHT 3
#define FRONT 4
#define BACK 5


CProxy_CodeGenCache codegen_proxy;


class CodeGenCache : public CBase_CodeGenCache
{
private:
    double start;
    CmiNodeLock lock;
    std::unordered_map<size_t, compute_fun_t> cache;

public:
    CodeGenCache()
    {
        lock = CmiCreateLock();
    }

    ~CodeGenCache()
    {
        CmiDestroyLock(lock);
    }

    compute_fun_t lookup(size_t hash)
    {
        // lock and find in cache else load from FS
        CmiLock(lock);
        auto find = cache.find(hash);
        compute_fun_t fun;
        if (find == std::end(cache))
        {
            // load the function
            fun = load_compute_fun(hash);
            cache[hash] = fun;
        }
        else
        {
            fun = find->second;
        }
        CmiUnlock(lock);
        return fun;
    }

    void start_time(double time)
    {
        start = time;
    }

    void end_time(double time)
    {
        CkPrintf("Total time = %f\n", time - start);
    }
};


class Stencil : public CBase_Stencil
{
private:
    int EPOCH;
    uint32_t itercount;
    uint8_t curr_gid;
    uint32_t num_iters;
    char* itercmd;
    bool sent_ghosts;
    std::vector<char*> graph_cache;
    std::vector<uint32_t> graph_size;
    std::vector<ck::future<bool>> futures;
    std::unordered_map<size_t, compute_fun_t> fun_cache;

    double comp_time;
    double recv_ghost_time;

public:
    Stencil_SDAG_CODE
    uint8_t name;
    
    // expects that the number of dimensions and length in each 
    // dimension will be specified at the time of creation
    uint32_t ndims;
    uint32_t odf; 
    int num_nbrs;
    int num_ghost_recv;
    
    std::vector<Field> fields;
    std::vector<uint32_t> ghost_depth;  // stores the depth of the ghosts corresponding to each field

    std::vector<uint32_t> local_size;
    std::vector<uint32_t> step;
    std::vector<uint32_t> num_chares;
    std::vector<int> index;
    
    uint32_t start_x, start_y, start_z;
    std::vector<uint32_t> dims;

    Stencil(uint8_t name_, uint32_t ndims_, uint32_t* dims_, uint32_t odf_)
        : EPOCH(0)
        , name(name_)
        , ndims(ndims_)
        , odf(odf_)
        , num_nbrs(0)
        , num_ghost_recv(0)
        , itercmd(nullptr)
        , num_iters(0)
        , sent_ghosts(false)
        , recv_ghost_time(0)
    {
        index = std::vector<int>(3);

        index[0] = thisIndex.x;
        index[1] = thisIndex.y;
        index[2] = thisIndex.z;

        local_size = std::vector<uint32_t>(3, 0);
        step = std::vector<uint32_t>(3, 0);
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

        // internal chares
        for (int i = 0; i < ndims; i++)
        {
            if (index[i] > 0)
                num_nbrs++;
            if (index[i] < num_chares[i] - 1)
                num_nbrs++;
        }

        // FIXME assuming ghost depth = 1
        for (int i = 0; i < ndims; i++)
            step[i] = (index[i] == num_chares[i] - 1) || index[i] == 0 ? local_size[i] + 1 :
                local_size[i] + 2;

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

        ck::future<bool> fut;
        futures.push_back(fut);

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
            char* cmd_check = cmd_copy + 1;
            graph_cache.push_back(cmd_copy);
            graph_size.push_back(cmd_size);
            cmd += cmd_size;
        }

        num_iters = extract<uint32_t>(cmd);
        itercmd = cmd;
        run_next_iteration();
    }

    void run_next_iteration()
    {
        if (itercount++ < num_iters)
        {
#ifndef NDEBUG
            if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0)
                CkPrintf("Running iteration %u\n", itercount);
#endif
            if (itercount == 3)
            {
                double start_time = CkTimer();
                CkCallback cb(CkReductionTarget(CodeGenCache, start_time), codegen_proxy[0]);
                contribute(sizeof(double), &start_time, CkReduction::min_double, cb);
            }

            curr_gid = extract<uint8_t>(itercmd);
            char* graph = graph_cache[curr_gid];
            uint32_t cmd_size = graph_size[curr_gid];

            bool gen = extract<bool>(graph);
             
            if (gen)
            {
                // FIXME handle multiple fields
                size_t num_ghost_fields = extract<size_t>(graph);
                uint8_t fname = extract<uint8_t>(graph);
                send_ghosts(fname);
            }
            else
                interpret_graph(graph);
        }
        else
        {
            double end_time = CkTimer();
            CkCallback cb(CkReductionTarget(CodeGenCache, end_time), codegen_proxy[0]);
            contribute(sizeof(double), &end_time, CkReduction::max_double, cb);
            itercount = 0;
            num_iters = 0;
            itercmd = nullptr;
            futures[EPOCH++].set(true);
#ifndef NDEBUG
            CkPrintf("Compute time = %f\n", comp_time);
            CkPrintf("Done epoch %i\n", EPOCH);
#endif
        }
    }

    void wait(ck::future<bool> done)
    {
        ck::wait_all(futures.begin(), futures.end());
        for (auto& f: futures)
            f.release();
        futures.clear();
        done.set(true);
    }

    void call_compute(uint8_t gid)
    {
        char* graph = graph_cache[gid];
        uint32_t cmd_size = graph_size[gid];

        bool gen = extract<bool>(graph);
         
        if (gen)
        {
            // FIXME handle multiple fields
            size_t num_ghost_fields = extract<size_t>(graph);
            if (num_ghost_fields != 1)
                CkAbort("Not implemented");
            uint8_t fname = extract<uint8_t>(graph);

            compute_fun_t compute_f;
            size_t hash = extract<size_t>(graph);
            // first check local cache
            auto find = fun_cache.find(hash);
            if (find == std::end(fun_cache))
            {
                // now check node level cache
                CodeGenCache* codegen_cache = codegen_proxy.ckLocalBranch();
                compute_f = codegen_cache->lookup(hash);
                // insert in local cache
                fun_cache[hash] = compute_f;
            }
            else
            {
                compute_f = find->second;
            }

            //CkPrintf("Local size = %i, %i, %i\n", local_size[0], local_size[1], local_size[2]);

            double start_comp = CkTimer();
            compute_f(fields, num_chares, index, local_size);
            comp_time += (CkTimer() - start_comp);
        }

#ifndef NDEBUG
        if (index[0] == 0 && index[1] == 0 && index[2] == 0)
            CkPrintf("Iteration %i done, Mem usage = %i\n", itercount, CmiMemoryUsage());
#endif
    }

    void interpret_graph(char* cmd)
    {
        //OperandType operand_type = lookup_type(extract<uint8_t>(cmd));
        Operation oper = lookup_operation(extract<uint8_t>(cmd));

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

        run_next_iteration();
    }

    void populate_ghosts(Field &f, int start_x, int stop_x, int start_y, int stop_y,
            int start_z, int stop_z, double* ghost_data, bool print = false)
    {
        int idx = 0;
        for (int z = start_z; z < stop_z; z++)
            for (int y = start_y; y < stop_y; y++)
                for (int x = start_x; x < stop_x; x++)
                {
                    ghost_data[idx++] = f.data[x + step[0] * y + step[0] * step[1] * z];
                    //if (print && thisIndex.x == 0 && thisIndex.y == 1 && thisIndex.z == 1)
                    //{
                    //    CkPrintf("Coordinate = (%i, %i, %i), Flat = %i\n", x, y, z, x + step[0] * y + step[0] * step[1] * z);
                    //}
                }
    }

    // FIXME doesn't do diagonal neighbors
    void send_ghosts(uint8_t fname)
    {
        double ghost_start = CkTimer();
        int start_x, start_y, start_z;
        int stop_x, stop_y, stop_z;
        //CkPrintf("Step sizes = (%i, %i, %i)\n", step[0], step[1], step[2]);
        if (ndims == 3)
        {
            // FIXME different sizes in different dimensions
            Field &f = fields[fname];
            uint32_t ghost_size = local_size[0] * local_size[0];
            double* ghost_data = (double*) malloc(ghost_size * sizeof(double));

            // down
            if (index[2] > 0)
            {
                start_x = index[0] == 0 ? 0 : 1;
                start_y = index[1] == 0 ? 0 : 1;
                start_z = 1;
                stop_x = start_x + local_size[0];
                stop_y = start_y + local_size[1];
                stop_z = start_z + 1;
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data, true);
                thisProxy(thisIndex.x, thisIndex.y, thisIndex.z - 1).receive_ghost(
                    fname, UP, ghost_size, ghost_data);
            }
            if (index[2] < num_chares[2] - 1)
            {
                start_x = index[0] == 0 ? 0 : 1;
                start_y = index[1] == 0 ? 0 : 1;
                start_z = index[2] == 0 ? local_size[2] - 1 : local_size[2];
                stop_x = start_x + local_size[0];
                stop_y = start_y + local_size[1];
                stop_z = start_z + 1;
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data);
                thisProxy(thisIndex.x, thisIndex.y, thisIndex.z + 1).receive_ghost(
                    fname, DOWN, ghost_size, ghost_data);
            }
            if (index[0] > 0)
            {
                start_x = 1;
                start_y = index[1] == 0 ? 0 : 1;
                start_z = index[2] == 0 ? 0 : 1;
                stop_x = start_x + 1;
                stop_y = start_y + local_size[1];
                stop_z = start_z + local_size[2];
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data);
                thisProxy(thisIndex.x - 1, thisIndex.y, thisIndex.z).receive_ghost(
                    fname, RIGHT, ghost_size, ghost_data);
            }
            if (index[0] < num_chares[0] - 1)
            {
                start_x = index[0] == 0 ? local_size[0] - 1 : local_size[0];
                start_y = index[1] == 0 ? 0 : 1;
                start_z = index[2] == 0 ? 0 : 1;
                stop_x = start_x + 1;
                stop_y = start_y + local_size[1];
                stop_z = start_z + local_size[2];
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data);
                thisProxy(thisIndex.x + 1, thisIndex.y, thisIndex.z).receive_ghost(
                    fname, LEFT, ghost_size, ghost_data);
            }
            if (index[1] > 0)
            {
                start_x = index[0] == 0 ? 0 : 1;
                start_y = 1;
                start_z = index[2] == 0 ? 0 : 1;
                stop_x = start_x + local_size[0];
                stop_y = start_y + 1;
                stop_z = start_z + local_size[2];
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data);
                thisProxy(thisIndex.x, thisIndex.y - 1, thisIndex.z).receive_ghost(
                    fname, BACK, ghost_size, ghost_data);
            }
            if (index[1] < num_chares[1] - 1)
            {
                start_x = index[0] == 0 ? 0 : 1;
                start_y = index[1] == 0 ? local_size[1] - 1 : local_size[1];
                start_z = index[2] == 0 ? 0 : 1;
                stop_x = start_x + local_size[0];
                stop_y = start_y + 1;
                stop_z = start_z + local_size[2];
                populate_ghosts(f, start_x, stop_x, start_y, stop_y, start_z, stop_z,
                        ghost_data);
                thisProxy(thisIndex.x, thisIndex.y + 1, thisIndex.z).receive_ghost(
                    fname, FRONT, ghost_size, ghost_data);
            }

            free(ghost_data);
        }
        else
        {
            CkAbort("Not implemented");
        }

        //CkPrintf("Time for sending ghosts = %f\n", CkTimer() - ghost_start);

        sent_ghosts = true;
        if (num_ghost_recv == num_nbrs)
        {
            num_ghost_recv = 0;
            sent_ghosts = false;
            call_compute(curr_gid);
            run_next_iteration();
        }
    }
    
    // TODO implement multiple fields in same message
    void receive_ghost(uint8_t fname, int dir, int size, double* data)
    {
        //CkPrintf("Processing ghost at (%i, %i, %i) from %i\n", thisIndex.x, 
        //        thisIndex.y, thisIndex.z, dir);
        double start_recv = CkTimer();
        int start_x, start_y, start_z;
        int stop_x, stop_y, stop_z;
        if (ndims == 3)
        {
            Field &f = fields[fname];
            switch (dir)
            {
                case DOWN:
                {
                    start_x = index[0] == 0 ? 0 : 1;
                    start_y = index[1] == 0 ? 0 : 1;
                    start_z = 0;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + 1;
                    break;
                }
                case UP:
                {
                    start_x = index[0] == 0 ? 0 : 1;
                    start_y = index[1] == 0 ? 0 : 1;
                    start_z = index[2] == 0 ? local_size[2] : local_size[2] + 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + 1;
                    break;
                }
                case LEFT:
                {
                    start_x = 0;
                    start_y = index[1] == 0 ? 0 : 1;
                    start_z = index[2] == 0 ? 0 : 1;
                    stop_x = start_x + 1;
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + local_size[2];
                    break;
                }
                case RIGHT:
                {
                    start_x = index[0] == 0 ? local_size[0] : local_size[0] + 1;
                    start_y = index[1] == 0 ? 0 : 1;
                    start_z = index[2] == 0 ? 0 : 1;
                    stop_x = start_x + 1;
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + local_size[2];
                    break;
                }
                case FRONT:
                {
                    start_x = index[0] == 0 ? 0 : 1;
                    start_y = 0;
                    start_z = index[2] == 0 ? 0 : 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + 1;
                    stop_z = start_z + local_size[2];
                    break;
                }
                case BACK:
                {
                    start_x = index[0] == 0 ? 0 : 1;
                    start_y = index[1] == 0 ? local_size[1] : local_size[1] + 1;
                    start_z = index[2] == 0 ? 0 : 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + 1;
                    stop_z = start_z + local_size[2];
                    break;
                }
            }

            int idx = 0;
            for (int z = start_z; z < stop_z; z++)
                for (int y = start_y; y < stop_y; y++)
                    for (int x = start_x; x < stop_x; x++)
                        f.data[x + step[0] * y + step[0] * step[1] * z] = data[idx++];
        }
        else
        {
            CkAbort("Not implemented");
        }

        recv_ghost_time += CkTimer() - start_recv;

        if (++num_ghost_recv == num_nbrs && sent_ghosts)
        {
            num_ghost_recv = 0;
            sent_ghosts = false;
            //CkPrintf("Total recv time = %f\n", recv_ghost_time);
            call_compute(curr_gid);
            run_next_iteration();
        }
    }
};

#include "stencil.def.h"
