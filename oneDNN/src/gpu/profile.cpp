/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <vector>

#include "gpu/profile.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/ocl/profile.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "sycl/profile.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

static setting_t<bool> profile {false};

bool is_profiling_enabled() {
    return profile.get();
}

status_t get_profile_info_impl(
        const std::unordered_map<uint64_t, profile_entry_t> &stamp2entry,
        uint64_t *nsecs, uint64_t *cycles) {
    int idx = 0;
    for (auto &kv : stamp2entry) {
        auto &e = kv.second;
        double freq = e.freq / e.kernel_count;
        nsecs[idx] = e.nsec;
        cycles[idx] = freq * e.nsec / 1e9;
        idx++;
    }
    return status::success;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

using dnnl::impl::status_t;

extern "C" status_t DNNL_API dnnl_impl_gpu_set_profiling(int flag) {
    using namespace dnnl::impl;
    dnnl::impl::gpu::profile.set((bool)flag);
    return status::success;
}

extern "C" status_t DNNL_API dnnl_impl_gpu_reset_profiling() {
    using namespace dnnl::impl;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    return dnnl::impl::gpu::ocl::reset_profiling();
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    return dnnl::impl::sycl::reset_profiling();
#endif
    return status::unimplemented;
}

extern "C" status_t DNNL_API dnnl_impl_gpu_get_profile_info(
        int *num_entries, uint64_t *nsecs, uint64_t *cycles) {

    using namespace dnnl::impl;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    return dnnl::impl::gpu::ocl::get_profile_info(num_entries, nsecs, cycles);
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    return dnnl::impl::sycl::get_profile_info(num_entries, nsecs, cycles);
#endif
    return status::unimplemented;
}
