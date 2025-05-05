#include "server.hpp"
#include "converse.h"
#include "conv-ccs.h"

#include "server.decl.h"

class Main : public CBase_Main
{
public:
    Main(CkArgMsg* msg) 
    {
        register_handlers();
        codegen_proxy = CProxy_CodeGenCache::ckNew();
#ifndef NDEBUG
        CkPrintf("Initialization done\n");
#endif
    }

    void register_handlers()
    {
        CcsRegisterHandler("connect", (CmiHandler) Server::connection_handler);
        CcsRegisterHandler("disconnect", (CmiHandler) Server::disconnection_handler);
        CcsRegisterHandler("operation", (CmiHandler) Server::operation_handler);
        CcsRegisterHandler("fetch", (CmiHandler) Server::fetch_handler);
    }
};

#include "server.def.h"

