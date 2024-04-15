__host__ __device__ __forceinline__ uint32_t currentHash(uint32_t id)
{
    return id * 11;
}
__device__ __forceinline__ uint32_t toHashEntry(uint32_t row, uint32_t col)
{
    return (row << 27) + col;
}
__device__ __forceinline__ uint32_t hashEntryToColumn(uint32_t hashEntry)
{
    return hashEntry & 0x7FFFFFF;
}
__device__ __forceinline__ uint32_t hashEntryToRow(uint32_t hashEntry)
{
    return hashEntry >> 27;
}
template <size_t MAX_ROW_COUNT>
struct HashMapNoValue
{
private:
    uint32_t limit;

public:
    __device__ INDEX_TYPE UNUSED() const { return INT_MAX; }
    INDEX_TYPE *ids;
    INDEX_TYPE *occupancyPerRow;
    INDEX_TYPE *occupancy;
    // no default values or else union does not work
    int reserved;
    uint32_t capacity;
    __device__ void init(bool mainThread)
    {
        for (int i = threadIdx.x; i < capacity; i += blockDim.x)
            ids[i] = UNUSED();

        for (int i = threadIdx.x; i < MAX_ROW_COUNT; i += blockDim.x)
            occupancyPerRow[i] = 0;

        if (mainThread)
        {
            *occupancy = 0;
            limit = capacity;
        }
    }
    __device__ __forceinline__ void operator[](INDEX_TYPE id)
    {
        INDEX_TYPE hashed_id = currentHash(id);
        INDEX_TYPE map_id = hashed_id % getSize();
        do
        {
            auto entry = ids[map_id];
            if (entry == id)
                return;

            if (entry == UNUSED())
            {
                auto old_id = atomicCAS(ids + map_id, UNUSED(), id);

                if (old_id == UNUSED() || old_id == id)
                {
                    if (old_id == UNUSED())
                    {
                        atomicAdd_block(occupancy, 1);
                        atomicAdd_block(&occupancyPerRow[idToRow(id)], 1);
                    }
                    return;
                }
            }

            map_id = (map_id + 1) % getSize();
        } while (true);
    }

    __device__ __forceinline__ void limitSize(uint32_t limit)
    {
        this->limit = min(limit, capacity);
    }

    __device__ __forceinline__ INDEX_TYPE coordToId(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        return toHashEntry(rowA, colB);
    }

    __device__ __forceinline__ static INDEX_TYPE idToRow(INDEX_TYPE id) { return hashEntryToRow(id); }

    __device__ __forceinline__ static INDEX_TYPE idToCol(INDEX_TYPE id) { return hashEntryToColumn(id); }

    __device__ __forceinline__ void at(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        this->operator[](coordToId(rowA, colB));
    }

    __device__ __forceinline__ void atDirect(INDEX_TYPE rowA, INDEX_TYPE colB)
    {
        if (ids[colB] != UNUSED())
            return;

        INDEX_TYPE retVal = atomicCAS(&ids[colB], UNUSED(), coordToId(rowA, colB));
        if (retVal == UNUSED())
        {
            atomicAdd_block(occupancy, 1);
            atomicAdd_block(&occupancyPerRow[rowA], 1);
        }
    }

    __device__ __forceinline__ size_t getSize() const { return limit; }
};
