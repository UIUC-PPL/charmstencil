#include <stack>
#include <cinttypes>
#include <queue>
#include <unordered_map>
#include "stencil.hpp"


CProxy_Stencil stencil;
CcsDelayedReply operation_reply;
CcsDelayedReply fetch_reply;


class Server
{
public:

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
        int size = extract<int>(cmd);
        //char* msg_cpy = (char*) malloc(size);
        operation_reply = CcsDelayReply();
        //memcpy(msg_cpy, cmd, size);
        codegen_proxy.receive(size, cmd, stencil);
        //CcsSendReply(1, &res);
    }

    static void fetch_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        int name = extract<int>(cmd);
        fetch_reply = CcsDelayReply();
        stencil.gather(name);
    }

    static void connection_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        int odf = extract<int>(cmd);
        create_stencil(odf);
        CkPrintf("Connected to server odf = %i\n", odf);
        char res;
        CcsSendReply(1, &res);
    }

    static void disconnection_handler(char* msg)
    {
        char* cmd = msg + CmiMsgHeaderSizeBytes;
        stencil.ckDestroy();
#ifndef NDEBUG
        CkPrintf("Disconnected from server\n");
#endif
        char res;
        CcsSendReply(1, &res);
    }

    static CProxy_Stencil create_stencil(int odf)
    {
        int num_chare_x, num_chare_y;

        int total_chares = odf * CkNumPes();

        num_chare_x = sqrt(total_chares); //std::min(odf, dims[0]);
        num_chare_y = sqrt(total_chares); //std::min(odf, dims[1]);


#ifndef NDEBUG
        CkPrintf("Creating stencil\n");
#endif

        stencil = CProxy_Stencil::ckNew(num_chare_x, num_chare_y, num_chare_x, num_chare_y);
        return stencil;
    }

};

