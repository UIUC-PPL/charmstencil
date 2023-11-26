#include <vector>
#include <cstring>
#include "field.hpp"
#include "codegen.hpp"
#include "stencil.decl.h"


#define LEFT 0
#define RIGHT 1
#define FRONT 2
#define BACK 3
#define DOWN 4
#define UP 5


extern void invoke_fb_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_ud_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_rl_unpacking_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_fb_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_ud_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_rl_packing_kernel(double* f, double* ghost_data, int ghost_depth, int startx, 
    int starty, int startz, int stepx, int stepy, uint32_t* local_size, cudaStream_t stream);
extern void invoke_init_fields(double** fields, uint8_t num_fields, uint32_t total_size, cudaStream_t stream);
extern CUfunction load_kernel(size_t &hash);
extern void launch_kernel(void** args, uint32_t* local_size, int* block_sizes, 
    CUfunction& compute_kernel, cudaStream_t& stream);


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
            fun = load_kernel(hash);
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

    cudaStream_t compute_stream;
    cudaStream_t comm_stream;
    
    double* send_ghosts[27];
    double* recv_ghosts[27];

    Stencil(uint8_t name_, uint32_t ndims_, uint32_t* dims_, uint32_t odf_,
        uint8_t num_fields_, uint32_t* ghost_depth_)
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
        , init_count(0)
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

        create_fields(num_fields_, ghost_depth_);

        // FIXME handle different ghost depths for different fields
        for (int i = 0; i < ndims; i++)
            step[i] = local_size[i] + 2 * ghost_depth[0];

        
        // FIXME fix non cube decompositions
        ghost_size = ghost_depth[0] * local_size[0] * local_size[0];
        CkPrintf("Ghost size = %u\n", ghost_size);

        //send_ghosts.resize(num_nbrs);
        //recv_ghosts.resize(num_nbrs);

        hapiCheck(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, 0));
        hapiCheck(cudaStreamCreateWithPriority(&comm_stream, cudaStreamDefault, -1));
        

        for (int i = 0; i < num_nbrs; i++)
        {
            hapiCheck(cudaMalloc((void**)&send_ghosts[i], sizeof(double) * ghost_size));
            hapiCheck(cudaMalloc((void**)&recv_ghosts[i], sizeof(double) * ghost_size));
        }  
    }

    Stencil(CkMigrateMessage* m) {}

    ~Stencil()
    {
        delete_cache();
        for (int i = 0; i < num_nbrs; i++)
        {
            hapiCheck(cudaFree(send_ghosts[i]));
            hapiCheck(cudaFree(recv_ghosts[i]));
        }
        //free(send_ghosts);
        //free(recv_ghosts);
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
        itercmd = cmd;
        run_next_iteration();
        //thisProxy[thisIndex].iterate();
    }

    void run_next_iteration()
    {
        if (itercount++ < num_iters)
        {
#ifndef NDEBUG
            if(thisIndex.x == 0 && thisIndex.y == 0 && thisIndex.z == 0)
                CkPrintf("Running iteration %u\n", itercount);
#endif
            if (itercount == 1)
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
                std::vector<uint8_t> fnames;
                for (int i = 0; i < num_ghost_fields; i++)
                    fnames.push_back(extract<uint8_t>(graph));
                send_ghost_data(fnames);
                thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).iterate(curr_gid);
            }
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

            // TODO don't hardcode this
            int block_sizes[3] = {8, 8, 8};
            //double** fields_ptr = fields.begin();
            void* args[num_fields + 3];
            for(int i = 0; i < num_fields; i++)
                args[i] = &fields[i];
            args[num_fields] = &num_chares;
            args[num_fields + 1] = &index;
            args[num_fields + 2] = &local_size;
            double start_comp = CkTimer();
            launch_kernel(args, local_size, block_sizes, compute_f, compute_stream);
            //compute_f(fields, num_chares, index, local_size);
            comp_time += (CkTimer() - start_comp);
        }

#ifndef NDEBUG
        if (index[0] == 0 && index[1] == 0 && index[2] == 0)
            CkPrintf("Iteration %i done, Mem usage = %i\n", itercount, CmiMemoryUsage());
#endif
    }

    void create_fields(uint8_t num_fields_, uint32_t* ghost_depth_)
    {
        num_fields = num_fields_;
        //fields.resize(num_fields);
        fields = (double**) malloc(sizeof(double*) * num_fields);
        ghost_depth.resize(num_fields);
        for (uint8_t fname = 0; fname < num_fields; fname++)
        {
            uint32_t depth = ghost_depth_[fname];
            uint32_t total_local_size = 1;
#ifndef NDEBUG
            CkPrintf("Create field %" PRIu8 " with depth %" PRIu8 "\n", fname, depth);
#endif

            for (int i = 0; i < ndims; i++)
                total_local_size *= (local_size[i] + 2 * depth);

            hapiCheck(cudaMalloc((void**) &(fields[fname]), sizeof(double) * total_local_size));
            ghost_depth[fname] = depth;
            if (is_gpu)
                invoke_init_fields(fields, num_fields, total_local_size, compute_stream);
        }
    }

    // FIXME doesn't do diagonal neighbors
    void send_ghost_data(std::vector<uint8_t> &fnames)
    {
        double ghost_start = CkTimer();
        int start_x, start_y, start_z;
        int stop_x, stop_y, stop_z;
        //CkPrintf("Step sizes = (%i, %i, %i)\n", step[0], step[1], step[2]);

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
                    start_x = 1;
                    start_y = 1;
                    start_z = 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + 1;
                    invoke_ud_packing_kernel(f, send_ghosts[DOWN], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z - 1).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[DOWN], comm_stream));
                }
                if (!bounds[UP])
                {
                    start_x = 1;
                    start_y = 1;
                    start_z = local_size[2];
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + 1;
                    invoke_ud_packing_kernel(f, send_ghosts[UP], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x, thisIndex.y, thisIndex.z + 1).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[UP], comm_stream));
                }
                if (!bounds[LEFT])
                {
                    start_x = 1;
                    start_y = 1;
                    start_z = 1;
                    stop_x = start_x + 1;
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + local_size[2];
                    invoke_rl_packing_kernel(f, send_ghosts[LEFT], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x, thisIndex.y - 1, thisIndex.z).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[LEFT], comm_stream));
                }
                if (!bounds[RIGHT])
                {
                    start_x = local_size[0];
                    start_y = 1;
                    start_z = 1;
                    stop_x = start_x + 1;
                    stop_y = start_y + local_size[1];
                    stop_z = start_z + local_size[2];
                    invoke_rl_packing_kernel(f, send_ghosts[RIGHT], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x, thisIndex.y + 1, thisIndex.z).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[RIGHT], comm_stream));
                }
                if (!bounds[FRONT])
                {
                    start_x = 1;
                    start_y = 1;
                    start_z = 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + 1;
                    stop_z = start_z + local_size[2];
                    invoke_fb_packing_kernel(f, send_ghosts[FRONT], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x + 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[FRONT], comm_stream));
                }
                if (!bounds[BACK])
                {
                    start_x = 1;
                    start_y = local_size[1];
                    start_z = 1;
                    stop_x = start_x + local_size[0];
                    stop_y = start_y + 1;
                    stop_z = start_z + local_size[2];
                    invoke_fb_packing_kernel(f, send_ghosts[BACK], depth, start_x,
                        start_y, start_z, step[0], step[1], local_size, compute_stream);
                    thisProxy(thisIndex.x - 1, thisIndex.y, thisIndex.z).receive_ghost(
                        itercount, fname, UP, ghost_size, CkDeviceBuffer(send_ghosts[BACK], comm_stream));
                }
            }
            else
            {
                CkAbort("Not implemented");
            }
        }
    }
    
    void receive_ghost(int iter, uint8_t fname, int dir, int &size, double* &data, CkDeviceBufferPost *devicePost)
    {
        data = recv_ghosts[dir];
        devicePost[0].cuda_stream = comm_stream;
    }

    // TODO implement multiple fields in same message
    void process_ghost(uint8_t fname, int dir, int size, double* data)
    {
        //CkPrintf("Processing ghost at (%i, %i, %i) from %i\n", thisIndex.x, 
        //        thisIndex.y, thisIndex.z, dir);
        double start_recv = CkTimer();
        int startx, starty, startz;
        double* recv_ghost = recv_ghosts[dir];
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
                    invoke_ud_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
                case UP:
                {
                    start_x = 1;
                    start_y = 1;
                    start_z = local_size[2] + 1;
                    invoke_ud_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
                case LEFT:
                {
                    startx = 1;
                    starty = 0;
                    startz = 1;
                    invoke_rl_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
                case RIGHT:
                {
                    startx = 1;
                    starty = local_size[0] + 1;
                    startz = 1;
                    invoke_rl_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
                case FRONT:
                {
                    startx = local_size[1] + 1;
                    starty = 1;
                    startz = 1;
                    invoke_fb_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
                case BACK:
                {
                    startx = 0;
                    starty = 1;
                    startz = 1;
                    invoke_fb_unpacking_kernel(f, recv_ghost, depth, 
                        startx, starty, startz, step[0], step[1], local_size, compute_stream);
                    break;
                }
            }
        }
        else
        {
            CkAbort("Not implemented");
        }

        recv_ghost_time += CkTimer() - start_recv;
    }
};

#include "stencil.def.h"
