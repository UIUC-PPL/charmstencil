#include <vector>
#include <cstring>
#include "codegen.hpp"
#include "stencil.decl.h"


#define IDX3D(x, y, z, step) ((x) + (y) * (step)[0] + (z) * (step)[0] * (step)[1])
#define IDX2D(x, y, step) ((x) + (y) * (step)[0])

#define BACK 0
#define FRONT 1
#define LEFT 2
#define RIGHT 3
#define DOWN 4
#define UP 5


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

class StencilParams
{
public:
    uint8_t ndims;
    uint32_t dims[3];
    uint8_t num_fields;
    uint8_t odf;
    uint32_t ghost_depth[3];

    StencilParams(uint8_t ndims_, uint32_t* dims_, uint8_t num_fields_,
        uint8_t odf_, uint32_t* ghost_depth_)
        :   ndims(ndims_),
            num_fields(num_fields_),
            odf(odf_)
    {
        for (int i = 0; i < ndims; i++)
        {
            dims[i] = dims_[i];
            ghost_depth[i] = ghost_depth_[i];
        }
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
    uint32_t ghost_size;

    double comp_time;
    double recv_ghost_time;

public:
    Stencil_SDAG_CODE
    uint8_t name;
    
    // expects that the number of dimensions and length in each 
    // dimension will be specified at the time of creation
    uint8_t num_fields;
    uint32_t ndims;
    uint32_t odf; 
    int num_nbrs;
    int num_ghost_recv;
    bool is_gpu;
    
    double** fields;
    //std::vector<double*> fields;
    std::vector<uint32_t> ghost_depth;  // stores the depth of the ghosts corresponding to each field

    uint32_t local_size[3];
    uint32_t step[3];
    uint32_t num_chares[3];
    int index[3];
    
    uint32_t start_x, start_y, start_z;
    uint32_t dims[3];
    int init_count;
    bool bounds[6];

    bool is_sync;
    ck::future<bool> sync_future;
    uint32_t sync_epoch;
    
    double* send_ghosts;

    Stencil(uint8_t name_, uint32_t ndims_, uint32_t* dims_, uint32_t odf_,
        uint8_t num_fields_, uint32_t* ghost_depth_, double* boundary_)
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
        , is_gpu(true)
        , itercount(0)
        , init_count(0)
        , is_sync(false)
    {
        index[0] = thisIndex.x;
        index[1] = thisIndex.y;
        index[2] = thisIndex.z;

        for (int i = 0; i < ndims; i++)
        {
            local_size[i] = 0;
            step[i] = 0;
            num_chares[i] = 1;
            dims[i] = dims_[i];
        }

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
            {
                num_nbrs++;
                bounds[2 * i] = false;
            }
            else
                bounds[2 * i] = true;
            if (index[i] < num_chares[i] - 1)
            {
                num_nbrs++;
                bounds[2 * i + 1] = false;
            }
            else
                bounds[2 * i + 1] = true;
        }

        // FIXME handle different ghost depths for different fields
        for (int i = 0; i < ndims; i++)
            step[i] = local_size[i] + 2 * ghost_depth_[0];

        create_fields(num_fields_, ghost_depth_, boundary_);

        // FIXME fix non cube decompositions
        ghost_size = ghost_depth[0] * local_size[0] * local_size[0];
        //CkPrintf("Ghost size = %u\n", ghost_size);
        send_ghosts = (double*) malloc(sizeof(double) * ghost_size);


        thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start();
    }

    Stencil(CkMigrateMessage* m) {}

    ~Stencil()
    {
        delete_cache();
        free(send_ghosts);
        for (int i = 0; i < num_fields; i++)
            free(fields[i]);
        free(fields);
        //free(recv_ghosts);
    }

    void delete_cache()
    {
        for (char* cmd : graph_cache)
            free(cmd);
    }

    void execute_graph(uint32_t epoch, int size, char* cmd)
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
        // TODO avoid making a copy, use persistent messages
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
        //CkPrintf("Iters = %u\n", num_iters);
        itercmd = cmd;
        run_next_iteration();
        //thisProxy[thisIndex].iterate();
    }

    void run_next_iteration()
    {
        //CkPrintf("Iterating %u, %u\n", itercount, num_iters);

        if (itercount++ < num_iters)
        {
#ifndef NDEBUG
            if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0)
                CkPrintf("Running iteration %u\n", itercount);
#endif
            /*if (itercount == 1)
            {
                double start_time = CkTimer();
                CkCallback cb(CkReductionTarget(CodeGenCache, start_time), codegen_proxy[0]);
                contribute(sizeof(double), &start_time, CkReduction::min_double, cb);
            }*/

            curr_gid = extract<uint8_t>(itercmd);
            char* graph = graph_cache[curr_gid];
            uint32_t cmd_size = graph_size[curr_gid];

            bool gen = extract<bool>(graph);
             
            if (gen)
            {
                // FIXME handle multiple fields
                size_t num_ghost_fields = extract<size_t>(graph);
                std::vector<uint8_t> fnames;
                for (int i = 0; i < num_ghost_fields; i++)
                    fnames.push_back(extract<uint8_t>(graph));
                if (fnames.size() > 0)
                {
                    send_ghost_data(fnames);
                    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).iterate(curr_gid);
                }
                else
                {
                    num_ghost_recv = 0;
                    call_compute(curr_gid);
                    run_next_iteration();
                }
            }
        }
        else
        {
            //double end_time = CkTimer();
            //CkCallback cb(CkReductionTarget(CodeGenCache, end_time), codegen_proxy[0]);
            //contribute(sizeof(double), &end_time, CkReduction::max_double, cb);
            itercount = 0;
            num_iters = 0;
            itercmd = nullptr;
            ++EPOCH;
#ifndef NDEBUG
            //CkPrintf("Compute time = %f\n", comp_time);
            CkPrintf("Done epoch %u\n", EPOCH);
#endif
            if (is_sync && sync_epoch == EPOCH)
            {
                sync_future.set(true);
                is_sync = false;
            }
            thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start();
        }
    }

    void wait(ck::future<bool> done, uint32_t last_epoch)
    {
        if (EPOCH == last_epoch)
            done.set(true);
        else
        {
            sync_future = done;
            is_sync = true;
            sync_epoch = last_epoch;
        }
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
            if (num_ghost_fields > 1)
                CkAbort("Not implemented");
            uint8_t fname = extract<uint8_t>(graph);

            compute_fun_t compute_f;
            size_t hash = extract<size_t>(graph);
            CkPrintf("lookup hash = %lu\n", hash);

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

            compute_f(fields, num_chares, index, local_size, dims);
            //comp_time += (CkTimer() - start_comp);
        }

#ifndef NDEBUG
        if (index[0] == 0 && index[1] == 0 && index[2] == 0)
            CkPrintf("Iteration %i done, Mem usage = %i\n", itercount, CmiMemoryUsage());
#endif
    }

    void create_fields(uint8_t num_fields_, uint32_t* ghost_depth_, double* boundary_)
    {
        num_fields = num_fields_;
        //fields.resize(num_fields);
        fields = (double**) malloc(sizeof(double*) * num_fields);
        ghost_depth.resize(num_fields);
        for (uint8_t fname = 0; fname < num_fields; fname++)
        {
            double bc = boundary_[fname];
            uint32_t depth = ghost_depth_[fname];
            uint32_t total_local_size = 1;
#ifndef NDEBUG
            CkPrintf("Create field %" PRIu8 " with depth %" PRIu8 " bc %f\n", 
                fname, depth, bc);
#endif

            for (int i = 0; i < ndims; i++)
                total_local_size *= (local_size[i] + 2 * depth);

            fields[fname] = (double*) malloc(sizeof(double) * total_local_size);
            std::memset(fields[fname], 0, sizeof(double) * total_local_size);
            double* f = fields[fname];
            if (ndims == 3)
            {
                for (int x = depth; x < local_size[0] + depth; x++)
                    for (int y = depth; y < local_size[1] + depth; y++)
                        f[IDX3D(x, y, depth, step)] = bc;

                for (int x = depth; x < local_size[0] + depth; x++)
                    for (int y = depth; y < local_size[1] + depth; y++)
                        f[IDX3D(x, y, local_size[2] + depth, step)] = bc;

                for (int x = depth; x < local_size[0] + depth; x++)
                    for (int z = depth; z < local_size[2] + depth; z++)
                        f[IDX3D(x, depth, z, step)] = bc;

                for (int x = depth; x < local_size[0] + depth; x++)
                    for (int z = depth; z < local_size[2] + depth; z++)
                        f[IDX3D(x, local_size[1] + depth, z, step)] = bc;

                for (int y = depth; y < local_size[1] + depth; y++)
                    for (int z = depth; z < local_size[2] + depth; z++)
                        f[IDX3D(depth, y, z, step)] = bc;

                for (int y = depth; y < local_size[1] + depth; y++)
                    for (int z = depth; z < local_size[2] + depth; z++)
                        f[IDX3D(local_size[2] + depth, y, z, step)] = bc;
            }

            ghost_depth[fname] = depth;
        }
    }

    void pack_ghosts3d(double* f, double* gh, int depth, int startx, int starty, int startz,
        int stopx, int stopy, int stopz)
    {
        int i = 0;
        for(int x = startx; x < stopx; x++)
            for(int y = starty; y < stopy; y++)
                for(int z = startz; z < stopz; z++)
                    gh[i++] = f[IDX3D(x, y, z, step)];
    }

    void pack_ghosts2d(double* f, double* gh, int depth, int startx, int starty,
        int stopx, int stopy)
    {
        int i = 0;
        for(int x = startx; x < stopx; x++)
            for(int y = starty; y < stopy; y++)
                gh[i++] = f[IDX2D(x, y, step)];
    }

    void unpack_ghosts3d(double* f, double* gh, int depth, int startx, int starty, int startz,
        int stopx, int stopy, int stopz)
    {
        int i = 0;
        for(int x = startx; x < stopx; x++)
            for(int y = starty; y < stopy; y++)
                for(int z = startz; z < stopz; z++)
                    f[IDX3D(x, y, z, step)] = gh[i++];
    }

    void unpack_ghosts2d(double* f, double* gh, int depth, int startx, int starty,
        int stopx, int stopy)
    {
        int i = 0;
        for(int x = startx; x < stopx; x++)
            for(int y = starty; y < stopy; y++)
                f[IDX2D(x, y, step)] = gh[i++];
    }

    // FIXME doesn't do diagonal neighbors
    void send_ghost_data(std::vector<uint8_t> &fnames)
    {
        double ghost_start = CkTimer();
        int startx, starty, startz;
        int stopx, stopy, stopz;

        // TODO send all ghosts in a single message
        for (int i = 0; i < fnames.size(); i++)
        {
            uint8_t fname = fnames[i];
            int depth = ghost_depth[fname];
            if (ndims == 3)
            {
                // FIXME different sizes in different dimensions
                double* f = fields[fname];

                // down
                if (!bounds[DOWN])
                {
                    startx = 1;
                    starty = 1;
                    startz = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + local_size[1];
                    stopz = startz + 1;
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z - 1).receive_ghost(
                        itercount, fname, UP, ghost_size, send_ghosts);
                }
                if (!bounds[UP])
                {
                    startx = 1;
                    starty = 1;
                    startz = local_size[2];
                    stopx = startx + local_size[0];
                    stopy = starty + local_size[1];
                    stopz = startz + 1;
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z + 1).receive_ghost(
                        itercount, fname, DOWN, ghost_size, send_ghosts);
                }
                if (!bounds[LEFT])
                {
                    startx = 1;
                    starty = 1;
                    startz = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    stopz = startz + local_size[2];
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x, thisIndex.y - 1, thisIndex.z).receive_ghost(
                        itercount, fname, RIGHT, ghost_size, send_ghosts);
                }
                if (!bounds[RIGHT])
                {
                    startx = local_size[0];
                    starty = 1;
                    startz = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    stopz = startz + local_size[2];
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x, thisIndex.y + 1, thisIndex.z).receive_ghost(
                        itercount, fname, LEFT, ghost_size, send_ghosts);
                }
                if (!bounds[FRONT])
                {
                    startx = 1;
                    starty = 1;
                    startz = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    stopz = startz + local_size[2];
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x + 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, BACK, ghost_size, send_ghosts);
                }
                if (!bounds[BACK])
                {
                    startx = 1;
                    starty = local_size[1];
                    startz = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    stopz = startz + local_size[2];
                    pack_ghosts3d(f, send_ghosts, depth, startx, starty, startz, stopx, stopy, stopz);
                    thisProxy(thisIndex.x - 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, FRONT, ghost_size, send_ghosts);
                }
            }
            else if (ndims == 2)
            {
                // FIXME different sizes in different dimensions
                double* f = fields[fname];

                if (!bounds[LEFT])
                {
                    startx = 1;
                    starty = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    pack_ghosts2d(f, send_ghosts, depth, startx, starty, stopx, stopy);
                    thisProxy(thisIndex.x, thisIndex.y - 1, thisIndex.z).receive_ghost(
                        itercount, fname, RIGHT, ghost_size, send_ghosts);
                }
                if (!bounds[RIGHT])
                {
                    startx = local_size[0];
                    starty = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    pack_ghosts2d(f, send_ghosts, depth, startx, starty, stopx, stopy);
                    thisProxy(thisIndex.x, thisIndex.y + 1, thisIndex.z).receive_ghost(
                        itercount, fname, LEFT, ghost_size, send_ghosts);
                }
                if (!bounds[FRONT])
                {
                    startx = 1;
                    starty = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    pack_ghosts2d(f, send_ghosts, depth, startx, starty, stopx, stopy);
                    thisProxy(thisIndex.x + 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, BACK, ghost_size, send_ghosts);
                }
                if (!bounds[BACK])
                {
                    startx = 1;
                    starty = local_size[1];
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    pack_ghosts2d(f, send_ghosts, depth, startx, starty, stopx, stopy);
                    thisProxy(thisIndex.x - 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, FRONT, ghost_size, send_ghosts);
                }
            }
            else
            {
                CkAbort("Not implemented");
            }
        }
    }

    // TODO implement multiple fields in same message
    void process_ghost(uint8_t fname, int dir, int size, double* data)
    {
        //CkPrintf("Processing ghost at (%i, %i, %i) from %i\n", thisIndex.x, 
        //        thisIndex.y, thisIndex.z, dir);
        double start_recv = CkTimer();
        int startx, starty, startz, stopx, stopy, stopz;
        double* f = fields[fname];
        int depth = ghost_depth[fname];
        if (ndims == 3)
        {
            switch (dir)
            {
                case DOWN:
                {
                    startx = 1;
                    starty = 1;
                    startz = 0;
                    stopx = startx + local_size[0];
                    stopy = starty + local_size[1];
                    stopz = startz + 1;
                    break;
                }
                case UP:
                {
                    start_x = 1;
                    start_y = 1;
                    start_z = local_size[2] + 1;
                    stopx = startx + local_size[0];
                    stopy = starty + local_size[1];
                    stopz = startz + 1;
                    break;
                }
                case LEFT:
                {
                    startx = 1;
                    starty = 0;
                    startz = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    stopz = startz + local_size[2];
                    break;
                }
                case RIGHT:
                {
                    startx = 1;
                    starty = local_size[0] + 1;
                    startz = 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    stopz = startz + local_size[2];
                    break;
                }
                case FRONT:
                {
                    startx = local_size[1] + 1;
                    starty = 1;
                    startz = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    stopz = startz + local_size[2];
                    break;
                }
                case BACK:
                {
                    startx = 0;
                    starty = 1;
                    startz = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    stopz = startz + local_size[2];
                    break;
                }
            }
            unpack_ghosts3d(f, data, depth, startx, starty, startz, stopx, stopy, stopz);
        }
        else if (ndims == 2)
        {
            switch (dir)
            {
                case LEFT:
                {
                    startx = 1;
                    starty = 0;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    break;
                }
                case RIGHT:
                {
                    startx = 1;
                    starty = local_size[0] + 1;
                    stopx = startx + local_size[0];
                    stopy = starty + 1;
                    break;
                }
                case FRONT:
                {
                    startx = local_size[1] + 1;
                    starty = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    break;
                }
                case BACK:
                {
                    startx = 0;
                    starty = 1;
                    stopx = startx + 1;
                    stopy = starty + local_size[1];
                    break;
                }
            }
            unpack_ghosts2d(f, data, depth, startx, starty, stopx, stopy);
        }
        else
        {
            CkAbort("Not implemented");
        }

        recv_ghost_time += CkTimer() - start_recv;
    }
};

#include "stencil.def.h"
