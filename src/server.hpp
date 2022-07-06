#include <stack>
#include <cinttypes>
#include <variant>
#include <queue>
#include <unordered_map>
#include "ast.hpp"


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

    static CProxy_Stencil& lookup(uint8_t name)
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
            return create_stencil();
        }
        return find->second;
    }

    static void operation_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        uint8_t name = extract<uint8_t>(cmd);
        uint32_t epoch = extract<uint32_t>(cmd);
        uint32_t size = extract<uint32_t>(cmd);
        CProxy_Stencil& st = lookup(name);
        st.receive_graph(epoch, size, cmd);
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
};

