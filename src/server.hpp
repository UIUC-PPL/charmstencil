#include <stack>
#include <cinttypes>
#include <variant>
#include <queue>
#include <unordered_map>
#include "stencil.hpp"


std::unordered_map<uint8_t, CProxy_Stencil> stencil_table;
std::stack<uint8_t> client_ids;


class Server
{
public:
    static void initialize()
    {
        for (int16_t i = 255; i >= 0; i--)
            client_ids.push((uint8_t) i);
    }

    inline static void insert(uint8_t name, CProxy_Stencil st)
    {
#ifndef NDEBUG
        CkPrintf("Created stencil %" PRIu8 " on server\n", name);
#endif
        stencil_table.emplace(name, st);
    }

    inline static void remove(uint8_t name)
    {
        stencil_table.erase(name);
#ifndef NDEBUG
        CkPrintf("Deleted stencil %" PRIu8 " on server\n", name);
#endif
    }

    inline static uint8_t get_client_id()
    {
        if (client_ids.empty())
            CmiAbort("Too many clients connected to the server");
        uint8_t client_id = client_ids.top();
        client_ids.pop();
        return client_id;
    }

    static CProxy_Stencil lookup(uint8_t name, uint32_t ndims, uint32_t* dims, 
            uint32_t odf)
    {
        auto find = stencil_table.find(name);
        if (find == std::end(stencil_table))
        {
#ifndef NDEBUG
            CkPrintf("Stencil %" PRIu8 " not found\n", name);
            CkPrintf("Active stencils: ");
            for (auto it: stencil_table)
                CkPrintf("%" PRIu8 ", ", it.first);
            CkPrintf("\n");
#endif
            return create_stencil(name, ndims, dims, odf);
        }
        return find->second;
    }

    static void operation_handler(char* msg)
    {
        /*
         *        stencil name
        1. epoch
        cmd size
        2. number of new unique graphs
        for each new graph
            1. size of graph
            2. graph
        size of graph epochs section of cmd
        number of graph epochs
        3. array of graph index

         */
#ifndef NDEBUG
        CkPrintf("Operation handler called\n");
#endif
         char* cmd = msg + CmiMsgHeaderSizeBytes;
         uint8_t name = extract<uint8_t>(cmd);
         uint32_t epoch = extract<uint32_t>(cmd);
         uint32_t cmd_size = extract<uint32_t>(cmd);
         uint8_t ndims = extract<uint8_t>(cmd);
#ifndef NDEBUG
        CkPrintf("%" PRIu8 ", %u, %" PRIu8 "\n", name, epoch, ndims);
#endif
         uint32_t dims[ndims];

         for(int i = 0; i < ndims; i++)
             dims[i] = extract<uint32_t>(cmd);

         uint8_t odf = extract<uint8_t>(cmd);
         CProxy_Stencil st = lookup(name, ndims, dims, odf);
         st.receive_graph(epoch, cmd_size, cmd);
    }

    static void fetch_handler(char* msg)
    {
    }

    static void delete_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        uint8_t name = extract<uint8_t>(cmd);
        remove(name);
    }

    static void connection_handler(char* msg)
    {
        uint8_t client_id = get_client_id();
        CcsSendReply(1, (void*) &client_id);
    }

    static void disconnection_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        uint8_t client_id = extract<uint8_t>(cmd);
        client_ids.push(client_id);
#ifndef NDEBUG
        CkPrintf("Disconnected %" PRIu8 " from server\n", client_id);
#endif
    }

    inline static void exit_server(char* msg)
    {
        CkExit();
    }

    static CProxy_Stencil create_stencil(uint8_t name, uint32_t ndims, uint32_t* dims, 
            uint32_t odf)
    {
        uint32_t num_chare_x, num_chare_y, num_chare_z;
        num_chare_x = num_chare_y = num_chare_z = 1;

        uint32_t total_chares = odf * CkNumPes();

        if(ndims == 1)
            num_chare_x = total_chares; //std::min(odf, dims[0]);
        if(ndims == 2)
        {
            num_chare_x = sqrt(total_chares); //std::min(odf, dims[0]);
            num_chare_y = sqrt(total_chares); //std::min(odf, dims[1]);
        }
        if(ndims == 3)
        {
            num_chare_x = cbrt(total_chares); //std::min(odf, dims[0]);
            num_chare_y = cbrt(total_chares); //std::min(odf, dims[1]);
            num_chare_z = cbrt(total_chares); //std::min(odf, dims[1]);
        }

#ifndef NDEBUG
        CkPrintf("Creating stencil %" PRIu8 " of size (%u, %u, %u)\n", 
                name, num_chare_x, num_chare_y, num_chare_z);
#endif

        CProxy_Stencil new_stencil = CProxy_Stencil::ckNew(
                name, ndims, dims, odf, num_chare_x, num_chare_y, num_chare_z);
        insert(name, new_stencil);
        return new_stencil;
    }
};

