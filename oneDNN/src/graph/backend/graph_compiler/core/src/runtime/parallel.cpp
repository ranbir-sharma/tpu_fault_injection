/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <memory>
#include "config.hpp"
#include "context.hpp"
#include <runtime/generic_val.hpp>
#include <runtime/parallel.hpp>
#include <util/simple_math.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
#include <tbb/global_control.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_arena.h>
#endif

#include <runtime/thread_locals.hpp>
#ifdef SC_KERNEL_PROFILE
#include <atomic>
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#include <omp.h>
#endif

using namespace dnnl::impl::graph::gc;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#error "unimplemented"
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB

static int &get_default_threads() {
    static int num_threads = oneapi::tbb::info::default_concurrency();
    return num_threads;
}
extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, generic_val *), uint64_t flags,
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, generic_val *args) {
    runtime::thread_local_buffer_t::tls_buffer_.additional_->is_main_thread_
            = true;
    oneapi::tbb::task_arena arena(get_default_threads());
    arena.execute([&] {
        tbb::parallel_for(
                begin, end, step,
                [&](int64_t i) { pfunc(rtl_ctx, module_env, i, args); },
                tbb::simple_partitioner());
    });
}

static int get_num_threads() {
    return get_default_threads();
}

static void set_num_threads(int num) {
    get_default_threads() = num;
}

static int get_thread_num() {
    return tbb::this_task_arena::current_thread_index();
}
static int get_in_parallel() {
    return 0;
}

#else

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#define get_num_threads omp_get_max_threads
#define set_num_threads omp_set_num_threads
#define get_thread_num omp_get_thread_num
#define get_in_parallel omp_in_parallel
#else
static int get_num_threads() {
    return 1;
}

static void set_num_threads(int num) {}
static int get_thread_num() {
    return 0;
}
static int get_in_parallel() {
    return 0;
}
#endif

#ifdef SC_KERNEL_PROFILE
static std::atomic<int> instance_cnt = {0};
static thread_local int instance_id = instance_cnt++;
#endif

// omp or sequential
extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, generic_val *), uint64_t flags,
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, generic_val *args) {
#ifdef SC_KERNEL_PROFILE
    int parent_instance_id = instance_id;
#endif
    runtime::thread_local_buffer_t::tls_buffer_.additional_->is_main_thread_
            = true;

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#pragma omp parallel for
#endif
    for (int64_t i = begin; i < end; i += step) {
        auto &tls = runtime::thread_local_buffer_t::tls_buffer_;
#ifdef SC_KERNEL_PROFILE
        tls.additional_->instance_id_ = parent_instance_id;
        tls.additional_->linear_thread_id_ = get_thread_num();
#endif
        pfunc(rtl_ctx, module_env, i, args);
    }
}

#endif
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
thread_pool_table sc_pool_table {&sc_parallel_call_cpu_with_env_impl, nullptr,
        &get_num_threads, &set_num_threads, &get_thread_num, &get_in_parallel};
}
} // namespace graph
} // namespace impl
} // namespace dnnl
