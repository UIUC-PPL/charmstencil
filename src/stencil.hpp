#include <vector>
#include <cstring>
#include "stencil.decl.h"


using scalar_t = std::variant<int, double>


template<class T>
inline T extract(char* &msg, bool increment=true)
{
    T arg = *(reinterpret_cast<T*>(msg));
    if (increment)
        msg += sizeof(T);
    return arg;
}


enum class Operation : uint8_t
{
    noop = 0,
    create = 1,
    add = 2,
    sub = 3,
    mul = 4,
    norm = 5,
    getitem = 6,
    setitem = 7,
    exchange_ghosts = 8
};


enum class OperandType : uint8_t
{
    field = 0,
    slice = 1,
    tuple = 2,
    int_t = 3,
    double_t = 4
};


Operation lookup_operation(uint8_t opcode)
{
    return static_cast<Operation>(opcode);
}


OperandType lookup_type(uint8_t typecode)
{
    return static_cast<OperandType>(typecode);
}


struct Slice1D
{
    int start;
    int stop;
    int step;
};


struct Slice
{
    Slice1D index[3];
};


class Field
{
public:
    std::vector<double> data;
    Field(uint32_t data_size){
        double temp;
        for(int i=0;i<data_size;i++)
            data.push_back(temp);
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
    
    uint32_t size_x,size_y,size_z;
    uint32_t start_x,start_y,start_z;
    vector<uint32_t> dims(3,1);
    Stencil(uint8_t name_,uint32_t ndims_,uint32_t dims_[ndims_],uint32_t odf_)
        : EPOCH(0)
        , name(name_)
        , ndims(ndims_)
        , odf(odf_)
    {
        uint32_t num_chare_x,num_chare_y,num_chare_z;
        num_chare_x=num_chare_y=num_chare_z=1;

        for(int i=0;i<ndims;i++)
            dims[i]=dims_[i];

        if(ndims>0)
            num_chare_x=min(odf,dims_[0]);
        if(ndims>1)
            num_chare_y=min(odf,dims_[1]);
        if(ndims>2)
            num_chare_z=min(odf,dims_[2]);

        for(int i=0;i<ndims_;i++){    
            if(i==0){
                if(num_chare_x==dims_[0]){             
                    size_x=1;
                    start_x=thisIndex.x;
                }
                else if(thisIndex.x < dims_[0]%num_chare_x){
                    size_x=(dims_[0]/num_chare_x)+1;
                    start_x=thisIndex.x*size_x;
                }
                else{
                    size_x=(dims_[0]/num_chare_x);
                    start_x=std::min(dims_[0]%num_chare_x,thisIndex.x)*(size_x+1)+std::max(0,thisIndex.x-dims_[0]%num_chare_x)*size_x;
                }
            }
            else if(i==1){
                if(num_chare_y==dims_[1]){             
                    size_y=1;
                    start_y=thisIndex.y;
                }
                else if(thisIndex.y < dims_[1]%num_chare_y){
                    size_y=(dims_[1]/num_chare_y)+1;
                    start_y=thisIndex.y*size_y;
                }
                else{
                    size_y=(dims_[1]/num_chare_y);
                    start_y=std::min(dims_[1]%num_chare_y,thisIndex.y)*(size_y+1)+std::max(0,thisIndex.y-dims_[1]%num_chare_y)*size_y;
                }
            }
            else{
                if(num_chare_z==dims_[2]){             
                    size_z=1;
                    start_z=thisIndex.z;
                }
                else if(thisIndex.z < dims_[2]%num_chare_z){
                    size_z=(dims_[2]/num_chare_z)+1;
                    start_z=thisIndex.z*size_z;
                }
                else{
                    size_z=(dims_[2]/num_chare_z);
                    start_z=std::min(dims_[2]%num_chare_z,thisIndex.z)*(size_z+1)+std::max(0,thisIndex.z-dims_[2]%num_chare_x)*size_x;
                }
            }
        }

        thisProxy(thisIndex.x, thisIndex.y, thisIndex.z).start()
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
            char* cmd_copy = (char*) malloc(cmd_size);
            std::memcpy(cmd_copy, cmd, cmd_size);
            graph_cache.push_back(cmd_copy);
            cmd += cmd_size;
        }

        uint32_t num_iters = extract<uint32_t>(cmd);
        for (int i = 0; i < num_iters; i++)
        {
            uint8_t gid = extract<uint8_t>(cmd);
            char* graph = graph_cache[gid];
            // FIXME this probably moves the pointer in cache?
            traverse_graph(graph);
        }
    }

    void traverse_graph(char* cmd)
    {
        uint8_t opcode = extract<uint8_t>(cmd);
        Operation oper = lookup_operation(opcode);
        switch (oper)
        {
            case Operation::norm: {
                // TODO how to implement this?
            }
            case Operation::setitem: {
                uint32_t fname = extract<uint32_t>(cmd);
                Field &f = fields[fname];
                uint32_t depth=ghost_depth[fname];
                Slice key = get_slice(cmd);
                // wrap_slice(key, st);
                
                uint32_t step_x,step_y,step_z;
                if(ndims>0)
                    step_x=size_x+2*depth;
                if(ndims>1)
                    step_y=size_y+2*depth;
                if(ndims>2)
                    step_z=size_z+2*depth;

                for (int k = 0; (key.index[2].start+k*key.index[2].step) < key.index[2].stop; k++)
                    for (int j = 0; (key.index[1].start+j*key.index[1].step) < key.index[1].stop; j++)
                        for (int i = 0; (key.index[0].start+i*key.index[0].step) < key.index[0].stop; i++)
                        {
                            f.data[step_x * step_y * (key.index[2].start+k*key.index[2].step) + step_x * (key.index[1].start+j*key.index[1].step) + (key.index[0].start+i*key.index[0].step)] = traverse_loop(cmd, key, i, j, k);
                        }
            }
            case Operation::exchange_ghosts: {
                uint32_t num_exchange=extract<uint_32>(cmd);
                

                for(int i=0;i<num_exchange;i++){
                    uint32_t offset_x,offset_y,offset_z;

                        }
                    }
                }
            }
        }
    }

    double traverse_loop(char* cmd, Slice &set_slice, int i, int j, int k)
    {
        uint8_t opcode = extract<uint8_t>(cmd);
        Operation oper = lookup_operation(opcode);

        switch (oper)
        {
            case Operation::noop: {
                OperandType type = lookup_type(extract<uint8_t>(cmd));
                if (type == OperandType::int_t)
                    return extract<int>(cmd);
                else if (type == OperandType::double_t)
                    return extract<double>(cmd);
            }
            
            case Operation:create: {
                // extract data for ghost_depth
                
                //uint32_t depth;

                uint_32 size=1;

                if(ndims>0)
                    size*=(size_x+2*depth);
                else
                    size*=size_y;

                if(ndims>1)
                    size*=(size_y+2*depth);
                else
                    size*=size_y;

                if(ndims>2)
                    size*=(size_z+2*depth);
                else
                    size*=size_z;

                Field f=Field(size);
                fields.push_back(f);
                ghost_depth.push_back(depth);
                return f.data[0];
            }

            case Operation:add: {
                double ans;
                uint8_t len=extract<uint8_t>(cmd);
                uint8_t operand_type=lookup_type(extract<uint8_t>(cmd));

                switch(operand_type)
                {
                    case OperandType:field:{
                        ans=traverse_loop(cmd, key, i, j, k);
                    }

                    case OperandType:double_t:{
                        ans=extract<double>(cmd);
                    }
                }

                for(int i=1;i<len;i++){
                    operand_type=lookup_type(extract<uint8_t>(cmd));
                    switch(operand_type)
                    {
                        case OperandType:field:{
                            ans+=traverse_loop(cmd, key, i, j, k);
                        }

                        case OperandType:double_t:{
                            ans+=extract<double>(cmd);
                        }
                    }
                }

                return ans;
            }

            case Operation:sub: {
                double ans;
                uint8_t len=extract<uint8_t>(cmd);
                uint8_t operand_type=lookup_type(extract<uint8_t>(cmd));

                switch(operand_type)
                {
                    case OperandType:field:{
                        ans=traverse_loop(cmd, key, i, j, k);
                    }

                    case OperandType:double_t:{
                        ans=extract<double>(cmd);
                    }
                }

                for(int i=1;i<len;i++){
                    operand_type=lookup_type(extract<uint8_t>(cmd));
                    switch(operand_type)
                    {
                        case OperandType:field:{
                            ans-=traverse_loop(cmd, key, i, j, k);
                        }

                        case OperandType:double_t:{
                            ans-=extract<double>(cmd);
                        }
                    }
                }

                return ans;
            }

            case Operation:mul: {
                double ans;
                uint8_t len=extract<uint8_t>(cmd);
                uint8_t operand_type=lookup_type(extract<uint8_t>(cmd));

                switch(operand_type)
                {
                    case OperandType:field:{
                        ans=traverse_loop(cmd, key, i, j, k);
                    }

                    case OperandType:double_t:{
                        ans=extract<double>(cmd);
                    }
                }

                for(int i=1;i<len;i++){
                    operand_type=lookup_type(extract<uint8_t>(cmd));
                    switch(operand_type)
                    {
                        case OperandType:field:{
                            ans*=traverse_loop(cmd, key, i, j, k);
                        }

                        case OperandType:double_t:{
                            ans*=extract<double>(cmd);
                        }
                    }
                }

                return ans;
            }

            case Operation:getitem: {
                Slice key;
                uint32_t fname=extract<uint32_t>(cmd);
                Field &f=fields[fname];
                key=get_slice(cmd);
                
                uint32_t step_x,step_y,step_z;
                if(ndims>0)
                    step_x=size_x+2*depth;
                if(ndims>1)
                    step_y=size_y+2*depth;
                if(ndims>2)
                    step_z=size_z+2*depth;

                return f[step_x*step_y*(key.index[2].start+k*key.index[2].step)+step_x*(key.index[1].start+j*key.index[1].step)+(key.index[0].start+i*key.index[0].step)];
            }

        }
    }

    Slice get_slice(char* &cmd){
        uint8_t operandType = extract<uint8_t>(cmd); 
        OperandType type= lookup_type(operandType);

        Slice key;

        // switch (type)
        // {
        //     case OperandType::tuple {
        //         // assumes that tuple will deliver the right number of slices or ints
        //         uint8_t innerType = extract<uint8_t>(cmd); 
        //         OperandType type_= lookup_type(innerType);
                
        //         switch(type_)
        //         {
        //             case OperandType::slice {
        //                 for(int i=0;i<ndims;i++){
        //                     if(i==0){
        //                         uint32_t temp_start = extract<uint32_t>(cmd);
        //                         uint32_t temp_end = extract<uint32_t>(cmd);

        //                         if((start_x <= temp_start) && (start_x < temp_end)){
        //                             key.index[0].start=temp_start-start_x;
        //                             key.index[0].stop = min(temp_end,start_x + size_x);
        //                         }
        //                         else{
        //                             key.index[0].start=0;
        //                             key.index[0].top=0;
        //                         }
        //                         key.index[0].step = extract<uint32_t>(cmd);
        //                     }
        //                     else if(i==1){
        //                         uint32_t temp_start = extract<uint32_t>(cmd);
        //                         uint32_t temp_end = extract<uint32_t>(cmd);

        //                         if((start_y >= temp_start) && (start_x < temp_end)){
        //                             key.index[1].start=temp_start-start_y;
        //                             key.index[1].stop = min(temp_end,start_x + size_x);
        //                         }
        //                         else{
        //                             key.index[0].start=0;
        //                             key.index[0].top=0;
        //                         }
        //                         key.index[1].step = extract<uint32_t>(cmd);
        //                     }
        //                     else{

        //                     }
        //                 }
                        
        //                 for(i=ndims;i<3;i++){
        //                     key.index[i].start=0;
        //                     key.index[i].stop=1;
        //                     key.index[i].step=1;
        //                 }
        //             }

        //         }
        //     }
        //     case OperandType::slice{
        //         key.index[0].start=extract<uint32_t>(cmd);
        //         key.index[0].stop=extract<uint32_t>(cmd);
        //         key.index[0].step=extract<uint32_t>(cmd);

        //         key.index[1].start=0;
        //         key.index[1].stop=1;
        //         key.index[1].step=1;

        //         key.index[2].start=0;
        //         key.index[2].stop=1;
        //         key.index[2].step=1;
        //     }
        // }
        // return key;
    }
    
    void receiveGhost(int field_name,int dir,int size,int res[size]){
        
    }
};

#include "stencil.def.h"
