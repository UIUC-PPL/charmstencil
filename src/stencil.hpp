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
    std::vector<scalar_t> data;
};


class Stencil : public CBase_Stencil
{
private:
    int EPOCH;
    std::vector<char*> graph_cache;

public:
    Stencil_SDAG_CODE
    uint8_t name;
    std::vector<Field> fields;

    Stencil(uint8_t name_)
        : EPOCH(0)
        , name(name_)
    {
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
                Slice key = get_slice(cmd);
                wrap_slice(key, st);

                for (int i = key.index[2].start; i < key.index[2].stop; 
                        i += key.index[2].step)
                    for (int j = key.index[1].start; j < key.index[1].stop; 
                            j += key.index[1].step)
                        for (int k = key.index[0].start; k < key.index[0].stop; 
                                k += key.index[0].step)
                        {
                            f.data[size_j * size_k * i + size_k * j + k] = traverse_loop(
                                    cmd, key, i, j, k);
                        }
            }
            case Operation::exchange_ghosts: {
            }
        }
    }

    scalar_t traverse_loop(char* cmd, Slice &set_slice, int i, int j, int k)
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
        }
    }
};

#include "stencil.def.h"
