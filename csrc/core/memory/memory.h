#ifndef CSRC_CORE_MEMORY_MEMORY_H
#define CSRC_CORE_MEMORY_MEMORY_H

#include <cstddef>

namespace memory {

template <typename Tp, typename Device>
struct malloc_mem_op {
    /// @brief memory allocation for multi-device
    /// 
    /// Inputs:
    /// @param device  : the type of device
    /// @param p_data  : the input pointer
    /// @param size    : the size of the memory
    void operator()(const Device* device, Tp*& p_data, const size_t size);
};

template <typename Tp, typename Device>
struct free_mem_op {
    /// @brief memory free operator for multi-device
    /// 
    /// Inputs:
    /// @param device  : the type of device
    /// @param p_data  : the input pointer
    void operator()(const Device* device, Tp*& p_data);
};

template <typename Tp, typename Dev_dst, typename Dev_src>
struct copy_mem_op {
    /// @brief memory copy operator for multi-device
    /// 
    /// Inputs:
    /// @param dev_dst : the type of device of p_dst
    /// @param dev_src : the type of device of p_src
    /// @param p_dst   : the destination pointer
    /// @param p_src   : the source pointer
    /// @param size    : the size of the memory
    void operator()(
        const Dev_dst* dev_dst, 
        const Dev_src* dev_src, 
        Tp* p_dst, 
        const Tp* p_src, 
        const size_t size
    );
};

template <typename Tp, typename Device>
struct set_mem_op {
    /// @brief memory set operator for multi-device
    ///
    /// Inputs:
    /// @param device  : the type of device
    /// @param p_data  : the input pointer
    /// @param value   : the value to set
    /// @param size    : the size of the memory
    void operator()(const Device* device, Tp* p_data, const Tp value, const size_t size);
};

} // namespace memory

#endif