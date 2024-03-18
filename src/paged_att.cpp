#include "thaDNN.hpp"
#include "thaBLAS.hpp"
#include "hip_helper.hpp"

#include <hip/hip_runtime.h>

void initMemoryManager(MemoryManager *mm, int page_size, int row_size_in_bytes)
{
    mm->counter = 0;
    mm->page_size = page_size;
    mm->row_size_in_bytes = row_size_in_bytes;
    mm->table.clear();
}

void initRunStatePaged(RunStatePaged* sp, int page_size, int row_size_in_bytes)
{
    sp->n_pages = 0;
    sp->page_keys.clear();
    sp->page_ptr.clear();
    sp->page_size = page_size;
    sp->row_size_in_bytes = row_size_in_bytes;
}

long long requestNewPage(MemoryManager *mm, int gid, float* &new_page_ptr)
{
    long long new_page_id = ++mm[gid].counter;
    CHECK_HIP(hipMalloc(&new_page_ptr, mm[gid].row_size_in_bytes * mm[gid].page_size));
    mm[gid].table[new_page_id] = new_page_ptr;
    return new_page_id;
}

void extendPage(RunStatePaged* sp, MemoryManager* mm, int gid, int pos) {

    int total_row = sp->page_size * sp->n_pages;

    // if need more memory page
    if (total_row < (pos + 2)) // TODO: +2 for future placeholder
    {
        float* new_page_ptr;
        long long new_page_id = requestNewPage(mm, gid, new_page_ptr);
        sp->page_keys.push_back(new_page_id);
        sp->page_ptr.push_back(new_page_ptr);
        ++sp->n_pages;
    }
}

float* getPage(RunStatePaged* sp, int pos)
{
    int page_pos = (pos+1) / sp->page_size;
    float* page = sp->page_ptr[page_pos];
    float* row_pos = page + ((pos+1) % sp->page_size) * sp->row_size_in_bytes;
    return row_pos;
}

void freeRunStatePaged(RunStatePaged* sp, MemoryManager* mm, int gid, int pos) {

    for(int p=0 ; p<sp->n_pages ; ++p)
    {
        float* page_ptr = sp->page_ptr[p];
        CHECK_HIP(hipFree(page_ptr));

        long long page_id = sp->page_keys[p];
        mm->table.erase(page_id);
    }

    sp->page_keys.clear();
    sp->page_ptr.clear();
    sp->n_pages = 0;
}
