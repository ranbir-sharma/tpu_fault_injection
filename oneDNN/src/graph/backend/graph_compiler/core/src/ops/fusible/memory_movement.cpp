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

#include <assert.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "memory_movement.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/outer_loop_generator.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <unordered_map>
#include <util/math_utils.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
ir_module_ptr reshape_op_t::get_func(context_ptr ctx) {
    top_level_anchor_generator_t gen;
    attrs_.set(op_attr_key::no_fuse, true);
    auto ret = fusible_op_get_func(this, gen, ctx, true);
    return ret;
}

transpose_op_t::transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : order_(attrs.get<std::vector<int>>("order")) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    assert(info_.inputs_.size() == 1);
    assert(order_.size() == info_.inputs_[0]->details_.get_plain_dims().size());
    auto out_format = attrs.get_or_else("out_format", sc_data_format_t());
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        auto in_dims = info_.inputs_[0]->details_.get_plain_dims();
        sc_dims out_dims(in_dims.size());
        for (size_t i = 0; i < in_dims.size(); ++i) {
            out_dims[i] = in_dims[order_[i]];
        }
        info_.outputs_[0]->details_.set_plain_dims(out_dims);
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_format(out_format);
    }
    attrs_ = attrs;
    op_name_ = "transpose";
}

transpose_op_t::transpose_op_t(graph_tensor_ptr v, std::vector<int> &order)
    : order_(order) {
    info_.inputs_.emplace_back(std::move(v));
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    op_name_ = "transpose";
}

shape_rl_vec transpose_op_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &out_dims = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < in_dims.size(); i++) {
        if (is_dynamic_dim(in_dims[order_[i]])) {
            assert(is_dynamic_dim(out_dims[i]));
            ret.emplace_back(in_dims[order_[i]], out_dims[i]);
        }
    }
    return ret;
}

void transpose_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    COMPILE_ASSERT(!info_.inputs_[0]->details_.get_format().is_any(),
            "cannot infer output format with any input format");
    auto in_format = info_.inputs_[0]->details_.get_format();
    auto in_format_code = in_format.format_code_;
    int batch_dims = info_.inputs_[0]->details_.get_plain_dims().size()
            - in_format_code.norig_dims();
    std::unordered_map<int, int> order_map;
    for (size_t i = 0; i < order_.size(); ++i) {
        COMPILE_ASSERT((order_[i] >= batch_dims)
                        == (static_cast<int>(i) >= batch_dims),
                "Permutation on batch dims is not supported.")
        if (order_[i] >= batch_dims && static_cast<int>(i) >= batch_dims) {
            order_map[order_[i] - batch_dims] = i - batch_dims;
        }
    }

    std::vector<int> storage_args;
    for (int i = 0; i < in_format_code.ndims(); ++i) {
        int axis = in_format_code.get(i);
        storage_args.push_back(order_map[axis]);
    }
    auto out_format = sc_data_format_t(storage_args, in_format.blocks_);

    in_formats.push_back(std::vector<sc_data_format_t> {in_format});
    out_formats.push_back(std::vector<sc_data_format_t> {out_format});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void transpose_op_t::prepare_fusion_data(fdata_map &fdmap) {
    throw std::runtime_error("Not implemented");
}

void transpose_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    throw std::runtime_error("Not implemented");
}

void transpose_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    throw std::runtime_error("Not implemented");
}

void compute_block_transpose(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, const std::vector<int> &axis, size_t wkld) {
    std::vector<expr> iters(src[0]->nslice_dims());
    std::vector<expr> src_idx(src[0]->nslice_dims());
    std::vector<expr> dst_idx(src[0]->nslice_dims());

    for (unsigned i = 0; i < src[0]->nslice_dims(); i++) {
        iters[i] = builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx());
        src_idx[i] = iters[i];
    }
    dst_idx = src_idx;
    std::swap(dst_idx[axis[0]], dst_idx[axis[1]]);
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");

    stmt cur;
    expr indexed_target = builder::make_indexing(dst.tptr_, dst_idx);
    expr indexed_input = builder::make_indexing(src[0]->tptr_, src_idx);
    cur = make_stmt<assign_node_t>(indexed_target, indexed_input);
    for (int64_t i = src[0]->nslice_dims() - 1; i >= 0; i--) {
        auto body = make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(iters[i], expr(0),
                src[0]->get_shape()[i], expr(1), std::move(body), true,
                for_type::NORMAL);
    }
    bld->emit(cur);
}

void transpose_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    throw std::runtime_error("Not implemented");
}

size_t transpose_op_t::compute_workload(
        const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    return fusible_op_t::compute_workload(ins, outs)
            * workload_penalty_coefficient;
}

sc_dims tensor_view_op_t::get_shapes() const {
    return info_.outputs_[0]->details_.get_blocking_dims();
}

std::vector<expr> tensor_view_op_t::get_shapes_expr() {
    return info_.outputs_[0]->details_.get_blocking_dims_expr(
            get_owner_graph());
}

tensor_view_op_t::tensor_view_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "tensor_view";
    COMPILE_ASSERT(ins.size() == 1, "Reshape takes 1 input");
    info_.inputs_ = ins;
    auto cache_input_format = ins[0]->details_.get_format();
    attrs_ = attrs;
    auto shapes = attrs_.get<sc_dims>("shape");
    auto format = attrs.get_or_else("format", sc_data_format_t());
    int total_shape1 = 1, total_shape2 = 1, total_shape3 = 1;
    for (auto &dim : sc_data_format_t::get_padded_plain_shapes(
                 ins[0]->details_.get_blocking_dims(), cache_input_format)) {
        total_shape1 *= dim;
    }
    for (auto &dim : shapes) {
        total_shape2 *= dim;
    }
    if (!outs.empty()) {
        for (auto &dim : sc_data_format_t::get_padded_plain_shapes(
                     outs[0]->details_.get_blocking_dims(),
                     outs[0]->details_.get_format())) {
            total_shape3 *= dim;
        }
    }
    COMPILE_ASSERT(is_dynamic() || total_shape1 == total_shape2
                    || (!outs.empty() && total_shape1 == total_shape3),
            "Wrong total size of input shapes, can not do reshape plain dims "
            "from " << utils::print_vector(ins[0]->details_.get_plain_dims())
                    << " to " << utils::print_vector(shapes));
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(shapes);
        info_.outputs_[0]->details_.set_format(format);
        shapes_ = shapes;
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
        format = info_.outputs_[0]->details_.get_format();
        // changed to get dynamically in need.
        // shapes_ = outs[0]->details_.get_blocking_dims();
    }
    if (cache_input_format.is_any()) {
        cache_input_format
                = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                        ins[0]->details_.get_plain_dims().size()));
    }
    attrs_["cache_input_format"] = cache_input_format;
    if (format.is_any()) {
        format = sc_data_format_t(sc_data_format_kind_t::get_plain_by_dims(
                info_.outputs_[0]->details_.get_plain_dims().size()));
    }
    attrs_["format"] = format;
    if (is_dynamic()
            && count_dynamic_dims(get_inputs()[0]->details_.get_plain_dims())
                    != count_dynamic_dims(
                            get_outputs()[0]->details_.get_plain_dims())) {
        attrs_.set(op_attr_key::no_fuse, true);
    }
}

tensor_view_op_t::tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes)
    : tensor_view_op_t({std::move(v)}, {}, {{"shape", shapes}, {}}) {}

bool tensor_view_op_t::try_penetrate(
        sc_data_format_t &new_output_format) const {
    auto input_plain_shapes = info_.inputs_[0]->details_.get_plain_dims();
    auto input_blocking_shapes = info_.inputs_[0]->details_.get_blocking_dims();
    auto input_format = info_.inputs_[0]->details_.get_format();
    auto output_plain_shapes = info_.outputs_[0]->details_.get_plain_dims();
    auto input_size = input_plain_shapes.size();
    auto output_size = output_plain_shapes.size();
    bool inp_short = input_size < output_size;
    auto &short_plain_shapes
            = inp_short ? input_plain_shapes : output_plain_shapes;
    auto &long_plain_shapes
            = inp_short ? output_plain_shapes : input_plain_shapes;
    auto &short_size = inp_short ? input_size : output_size;
    auto &long_size = inp_short ? output_size : input_size;
    bool can_penetrate = true;
    std::unordered_map<size_t, size_t> long_to_short;
    // if inp_short, inp blk idx to out blk idx
    std::unordered_map<size_t, size_t> inp_blk_map;
    size_t short_idx = 0, long_idx = 0;
    while (short_idx < short_size) {
        int64_t acc_shape = long_plain_shapes[long_idx];
        long_to_short[long_idx] = short_idx;
        long_idx++;
        while (long_idx < long_size
                && (acc_shape < short_plain_shapes[short_idx]
                        || long_plain_shapes[long_idx] == 1)) {
            acc_shape *= long_plain_shapes[long_idx];
            long_to_short[long_idx] = short_idx;
            long_idx++;
        }

        if (acc_shape != short_plain_shapes[short_idx]) {
            can_penetrate = false;
            break;
        }
        // blocking of short format is big than corresponding long plain dims.
        if (inp_short) {
            auto blk_idx_list
                    = input_format.format_code_.collect_blocking_index(
                            short_idx);
            if (!blk_idx_list.empty()) {
                inp_blk_map[short_idx] = long_idx - 1;
                if (input_format.blocks_[blk_idx_list[0]]
                        > long_plain_shapes[long_idx - 1]) {
                    can_penetrate = false;
                    break;
                }
            }
        }
        short_idx++;
    }

    if (can_penetrate) {
        if (!inp_short) {
            auto &input_code = input_format.format_code_;
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            int out_count[sc_data_format_kind_t::MAX_DIMS] = {0};
            size_t blk_idx = 0;
            auto remain_blocks = output_plain_shapes;
            for (int i = 0; i < input_code.ndims(); i++) {
                auto new_idx = long_to_short[input_code.get(i)];
                new_code.set(i, new_idx);
                out_count[new_code.get(i)]++;
                if (out_count[new_code.get(i)] > 1
                        && blk_idx < input_size - output_size) {
                    new_format.blocks_[blk_idx++] = remain_blocks[new_idx];
                }
                remain_blocks[new_idx] /= input_plain_shapes[input_code.get(i)];
            }
            new_code.set(sc_data_format_kind_t::MAX_DIMS,
                    input_format.format_code_.get(
                            sc_data_format_kind_t::MAX_DIMS));
            size_t inp_blk_idx = 0;
            while (inp_blk_idx < 4 && blk_idx < 4
                    && input_format.blocks_[inp_blk_idx] > 0) {
                new_format.blocks_[blk_idx++]
                        = input_format.blocks_[inp_blk_idx++];
            }
            new_output_format = new_format;
            if (!is_dynamic()) {
                if (math_utils::get_dims_product(input_blocking_shapes)
                        != math_utils::get_dims_product(
                                sc_data_format_t::get_blocking_shapes(
                                        output_plain_shapes,
                                        new_output_format))) {
                    return false;
                }
            }
            return true;
        } else {
            sc_data_format_t new_format;
            auto &new_code = new_format.format_code_;
            for (int i = 0; i < static_cast<int>(output_plain_shapes.size());
                    i++) {
                new_code.set(i, i);
            }
            int inp_plain_idx = input_format.format_code_.norig_dims();
            int inp_blk_size = input_format.format_code_.ndims()
                    - input_format.format_code_.norig_dims();
            for (int i = 0; i < inp_blk_size; i++) {
                new_code.set(i + static_cast<int>(output_plain_shapes.size()),
                        inp_blk_map[input_format.format_code_.get(
                                i + inp_plain_idx)]);
            }
            new_code.set(sc_data_format_kind_t::MAX_DIMS,
                    input_format.format_code_.get(
                            sc_data_format_kind_t::MAX_DIMS));
            new_format.blocks_ = input_format.blocks_;
            new_output_format = new_format;
            if (!is_dynamic()) {
                if (math_utils::get_dims_product(input_blocking_shapes)
                        != math_utils::get_dims_product(
                                sc_data_format_t::get_blocking_shapes(
                                        output_plain_shapes,
                                        new_output_format))) {
                    return false;
                }
            }
            return true;
        }
    }
    new_output_format = info_.outputs_[0]->details_.get_format();
    return false;
}

shape_rl_vec tensor_view_op_t::get_dynamic_shape_relations() const {
    auto rl_axis_pair = attrs_.get_or_else(
            "temp.rl_axis_pair", std::vector<std::pair<int, int>>());
    auto in_dims = get_inputs()[0]->details_.get_plain_dims();
    auto out_dims = get_outputs()[0]->details_.get_plain_dims();
    shape_rl_vec ret;
    for (auto &it : rl_axis_pair) {
        if (is_dynamic_dim(in_dims[it.first])
                || is_dynamic_dim(out_dims[it.second])) {
            ret.emplace_back(in_dims[it.first], out_dims[it.second]);
        }
    }
    return ret;
}

void tensor_view_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    sc_data_format_t output_format;
    // temp workaround
    assert(!attrs_.get<sc_data_format_t>("format").is_any());
    bool query_by_dispatch_key = false;
    if (is_dynamic()) {
        auto key_set = get_dispatch_key_set();
        key_set->for_each_key_process([&](const op_dispatch_key_base_t *key) {
            auto dkey = static_cast<const op_dispatch_key_t *>(key);
            if (!query_by_dispatch_key
                    && dkey->in_out_formats_[0]
                            == info_.inputs_[0]->details_.get_format()) {
                query_by_dispatch_key = true;
                output_format = dkey->in_out_formats_[1];
            }
        });
    }
    if (query_by_dispatch_key) {
        out_formats.push_back({output_format});
        in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    } else if (attrs_.has_key("expand_dim")
            && info_.inputs_[0]->details_.get_format()
                    == attrs_.get<sc_data_format_t>("cache_input_format")) {
        out_formats.push_back({attrs_.get<sc_data_format_t>("format")});
        in_formats.push_back({info_.inputs_[0]->details_.get_format()});
    } else {
        bool can_penetrate = try_penetrate(output_format);
        if (can_penetrate) {
            out_formats.push_back({output_format});
            in_formats.push_back({info_.inputs_[0]->details_.get_format()});
        } else {
            out_formats.push_back({attrs_.get<sc_data_format_t>("format")});
            in_formats.push_back(
                    {attrs_.get<sc_data_format_t>("cache_input_format")});
        }
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void tensor_view_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}

slice_range_list infer_tensor_view_slice(sc_graph_t &graph,
        const slice_range_list &known_ranges_list,
        const std::vector<expr> &src_tv_dims,
        const std::vector<expr> &dst_tv_dims) {
    slice_range_list ret;
    for (auto known_ranges : known_ranges_list) {
        slice_range consistent_tv_slice;
        auto src_dims = src_tv_dims, dst_dims = dst_tv_dims;
        while (!dst_dims.empty() && !src_dims.empty()) {
            if (!slice_expr_equals(dst_dims.back(), src_dims.back())) break;
            consistent_tv_slice.insert(
                    consistent_tv_slice.begin(), known_ranges.back());
            known_ranges.pop_back();
            src_dims.pop_back();
            dst_dims.pop_back();
        }

        if (consistent_tv_slice.size() == dst_tv_dims.size()) {
            ret.emplace_back(consistent_tv_slice);
            continue;
        }

        bool slice_stop = false;
        // flatten index
        expr flatten_idx = 0;
        // total length of static dim
        sc_dim total_len = 1;
        // accumulater src dims
        expr acc_src_dim_expr = 1;
        const int dyn_len = -2;
        for (int i = src_dims.size() - 1; i >= 0; i--) {
            if (slice_stop) {
                // check whether slice is full on last several dims
                if (get_const_as_int(
                            known_ranges[i].second.checked_as<constant_c>())
                        != 1)
                    // if tensor_view deals with inconsequence slice, it will
                    // return empty slice range list to tell fusion manager not
                    // to fuse it
                    return slice_range_list {};
            }
            auto src_expr = src_dims[i];
            if (!(known_ranges[i].first.isa<constant_c>()
                        && get_const_as_int(
                                   known_ranges[i]
                                           .first.checked_as<constant_c>())
                                == 0
                        && slice_expr_equals(
                                known_ranges[i].second, src_expr))) {
                slice_stop = true;
            }
            if (known_ranges[i].second.isa<constant_c>()) {
                total_len *= get_const_as_int(
                        known_ranges[i].second.checked_as<constant_c>());
            } else {
                total_len *= dyn_len;
            }
            flatten_idx
                    = flatten_idx + known_ranges[i].first * acc_src_dim_expr;
            acc_src_dim_expr = acc_src_dim_expr * src_expr;
        }
        // deflatten to new shape
        slice_range reshape_ranges;
        sc_dims acc_dst_dim;
        std::vector<expr> acc_dst_dim_expr;
        sc_dim tmp_acc = 1;
        expr tmp_acc_expr = 1;
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            tmp_acc *= !dst_dims[i].isa<constant>()
                    ? dyn_len
                    : get_expr_as_int(dst_dims[i]);
            tmp_acc_expr = tmp_acc_expr * dst_dims[i];
            acc_dst_dim.emplace_back(tmp_acc);
            acc_dst_dim_expr.emplace_back(tmp_acc_expr);
        }
        std::reverse(acc_dst_dim.begin(), acc_dst_dim.end());
        std::reverse(acc_dst_dim_expr.begin(), acc_dst_dim_expr.end());
        std::vector<expr> dst_idx;
        for (unsigned i = 0; i < dst_dims.size() - 1; i++) {
            expr cur_idx = flatten_idx / acc_dst_dim_expr[i + 1];
            dst_idx.emplace_back(cur_idx);
            flatten_idx = flatten_idx % acc_dst_dim_expr[i + 1];
        }
        slice_stop = false;
        for (int64_t i = static_cast<int64_t>(dst_dims.size()) - 1; i >= 0;
                i--) {
            if (!slice_stop && abs(total_len) >= abs(acc_dst_dim[i])) {
                reshape_ranges.emplace_back(
                        std::make_pair(expr(0), dst_dims[i]));
                if (total_len == acc_dst_dim[i]) slice_stop = true;
            } else {
                if (!slice_stop) slice_stop = true;
                if (i == static_cast<int64_t>(dst_dims.size()) - 1) {
                    reshape_ranges.emplace_back(std::make_pair(
                            flatten_idx, expr(dim2unsigned(total_len))));
                } else {
                    reshape_ranges.emplace_back(std::make_pair(dst_idx[i],
                            expr(std::max(UINT64_C(1),
                                    dim2unsigned(
                                            total_len / acc_dst_dim[i + 1])))));
                }
            }
        }
        std::reverse(reshape_ranges.begin(), reshape_ranges.end());
        constant_folder_t f;
        auto_caster_t ca;
        for (auto &r : reshape_ranges) {
            r.first = ca(r.first).remove_const();
        }
        reshape_ranges.insert(reshape_ranges.end(), consistent_tv_slice.begin(),
                consistent_tv_slice.end());
        ret.emplace_back(reshape_ranges);
    }
    return ret;
}

void tensor_view_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
        return;
    }
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    if (known_ranges_map.empty()) return;
    slice_range_list known_ranges_list = known_ranges_map[0];

    if (fsmap.get(get_outputs()[0]).empty()) {
        auto &graph = get_owner_graph();
        // src
        auto src_dims
                = info_.inputs_[0]->details_.get_blocking_dims_expr(graph);
        // dst
        auto dst_dims
                = info_.outputs_[0]->details_.get_blocking_dims_expr(graph);

        auto tv_slice = infer_tensor_view_slice(
                graph, known_ranges_list, src_dims, dst_dims);

        if (tv_slice.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        fsmap.get(get_outputs()[0]) = tv_slice;
    }
}

void tensor_view_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (share_gt_with_op<output_op>(get_inputs()[0])) {
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
        return;
    }
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        auto &graph = get_owner_graph();
        // src
        auto src_dims
                = info_.inputs_[0]->details_.get_blocking_dims_expr(graph);
        // dst
        auto dst_dims
                = info_.outputs_[0]->details_.get_blocking_dims_expr(graph);
        // NOTE: pre_slice_ranges use shapes as src_dims
        auto tv_slice = infer_tensor_view_slice(
                graph, known_ranges_list, dst_dims, src_dims);
        if (tv_slice.empty()) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        fsmap.get(get_inputs()[0]) = tv_slice;
        // recursively pre-infer
        info_.inputs_[0]
                ->producer_owner_->dyn_cast<fusible_op_t>()
                ->pre_slice_ranges(fsmap, stat_map);
    }
}

bound_axis infer_tensor_view_binding_axis(const bound_axis &src_axis,
        const sc_dims &src_dims, const sc_dims &dst_dims,
        const std::vector<int> &expand_dims = {}) {
    bound_axis dst_axis, tv_axis_map;

    sc_dims acc_src_dims(src_dims.size()), acc_dst_dims(dst_dims.size());
    sc_dim tmp_acc = 1;
    std::transform(src_dims.begin(), src_dims.end(), acc_src_dims.begin(),
            [&tmp_acc](const sc_dim &d) {
                tmp_acc *= d;
                return tmp_acc;
            });
    tmp_acc = 1;
    std::transform(dst_dims.begin(), dst_dims.end(), acc_dst_dims.begin(),
            [&tmp_acc](const sc_dim &d) {
                tmp_acc *= d;
                return tmp_acc;
            });
    // compare src and dst
    int j = 0;
    for (size_t i = 0; i < acc_src_dims.size(); i++) {
        std::vector<int> axis;
        while (true) {
            axis.emplace_back(j);
            if (acc_src_dims[i] <= acc_dst_dims[j]) {
                if (acc_src_dims[i] == acc_dst_dims[j]) { j++; }
                break;
            }
            j++;
        }
        tv_axis_map.emplace_back(axis);
    }

    for (auto &bd_ax : src_axis) {
        std::vector<int> ret;
        for (auto &ax : bd_ax) {
            if (ax == -1
                    || expand_dims.end()
                            != std::find(expand_dims.begin(), expand_dims.end(),
                                    ax)) {
                ret.emplace_back(-1);
            } else {
                ret.insert(ret.end(), tv_axis_map[ax].begin(),
                        tv_axis_map[ax].end());
            }
        }
        // remove duplicated axis.
        std::sort(ret.begin(), ret.end());
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
        dst_axis.emplace_back(ret);
    }
    return dst_axis;
}

void tensor_view_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    auto known_axis_map = search_known_bound_axis(this, bdax_map);
    if (!bdax_map.get(get_outputs()[0]).empty()) return;
    // src
    auto src_dims = info_.inputs_[0]->details_.get_blocking_dims();
    // dst
    auto shapes = get_shapes();
    auto ths = this;
    bound_axis known_block_axis(known_axis_map[0].size());
    std::transform(known_axis_map[0].begin(), known_axis_map[0].end(),
            known_block_axis.begin(), [&ths](const std::vector<int> &bd_ax) {
                return transform_axis_plain2blocking(
                        ths->get_inputs()[0]->details_, bd_ax);
            });
    auto blocking_bd_axis = infer_tensor_view_binding_axis(
            known_block_axis, src_dims, shapes);
    bound_axis plain_bd_axis(blocking_bd_axis.size());
    std::transform(blocking_bd_axis.begin(), blocking_bd_axis.end(),
            plain_bd_axis.begin(), [&ths](const std::vector<int> &bd_ax) {
                return transform_axis_blocking2plain(
                        ths->get_outputs()[0]->details_, bd_ax);
            });
    bdax_map.get(get_outputs()[0]) = plain_bd_axis;
    set_unknown_axis_binding(this, known_axis_map, bdax_map);
}

void tensor_view_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    auto &outaxis = bdax_map.get(get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre bind axis")
    auto &input = get_inputs()[0];
    auto &inpaxis = bdax_map.get(input);

    if (inpaxis.empty()) {
        // src
        auto src_dims = input->details_.get_blocking_dims();
        // dst
        auto shapes = get_shapes();
        auto ths = this;
        bound_axis known_block_axis(outaxis.size());
        std::transform(outaxis.begin(), outaxis.end(), known_block_axis.begin(),
                [&ths](const std::vector<int> &bd_ax) {
                    return transform_axis_plain2blocking(
                            ths->get_outputs()[0]->details_, bd_ax);
                });
        auto blocking_bd_axis
                = infer_tensor_view_binding_axis(outaxis, shapes, src_dims,
                        attrs_.get_or_else("expand_dim", std::vector<int> {}));
        bound_axis plain_bd_axis(blocking_bd_axis.size());
        std::transform(blocking_bd_axis.begin(), blocking_bd_axis.end(),
                plain_bd_axis.begin(), [&ths](const std::vector<int> &bd_ax) {
                    return transform_axis_blocking2plain(
                            ths->get_inputs()[0]->details_, bd_ax);
                });
        inpaxis = plain_bd_axis;
        if (auto bd_op
                = input->producer_owner_
                          ->dyn_cast<op_traits::mixed_partition_acceptable>()) {
            bd_op->pre_binding_axis(bdax_map);
        }
    }
}

sc_dims tensor_view_op_t::get_bwise_fuse_shrink_dims() {
    auto old_dims = info_.inputs_[0]->details_.get_blocking_dims();
    auto new_dims = get_shapes();
    sc_dims bw_dims;
    int offset
            = std::min(op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
                               info_.inputs_[0]),
                    op_traits::batchwise_shrinkable_t::get_shrinkable_offset(
                            info_.outputs_[0]));
    int common_size = std::min(old_dims.size(), new_dims.size());
    for (int i = 0; i < std::min(common_size, offset); i++) {
        if (old_dims[i] == new_dims[i])
            bw_dims.emplace_back(new_dims[i]);
        else
            break;
    }
    return bw_dims;
}

sc_op_ptr tensor_view_op_t::bw_shrinked_copy(
        gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) {
    auto ins = get_inputs()[0];
    auto cache_input_format = ins->details_.get_format();
    COMPILE_ASSERT(bw_lt_map.haskey(ins),
            "tensor_view_op: new input graph tensor not found in map")
    auto plain_shape = sc_data_format_t::get_padded_plain_shapes(
            bw_lt_map.get(ins)->details_.get_blocking_dims(),
            cache_input_format);
    return op_traits::batchwise_shrinkable_t::bw_shrinked_copy(
            bw_lt_map, shrinked_graph, {{"shape", plain_shape}});
}

void tensor_view_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {}

reshape_op_t::reshape_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "reshape";
    COMPILE_ASSERT(ins.size() == 1, "Reshape copy takes 1 input");
    info_.inputs_ = ins;
    attrs_ = attrs;
    auto &shapes = attrs_.get<sc_dims>("shape");
    int total_shape1 = 1, total_shape2 = 1;
    for (auto &dim : ins[0]->details_.get_plain_dims()) {
        total_shape1 *= dim;
    }
    for (auto &dim : shapes) {
        total_shape2 *= dim;
    }
    COMPILE_ASSERT(total_shape1 == total_shape2,
            "Wrong total size of input shapes, can not do reshape plain dims "
            "from " << utils::print_vector(ins[0]->details_.get_plain_dims())
                    << " to " << utils::print_vector(shapes));
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.dtype_ = ins[0]->details_.dtype_;
        info_.outputs_[0]->details_.set_plain_dims(shapes);
        shapes_ = shapes;
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
        shapes_ = outs[0]->details_.get_plain_dims();
    }
}
void reshape_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}
void reshape_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // fake infer slice
    std::vector<std::pair<expr, expr>> ranges;
    auto &shapes = info_.outputs_[0]->details_.get_plain_dims();
    ranges.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); i++) {
        ranges.emplace_back(expr(0), expr(dim2unsigned(shapes[i])));
    }
    fsmap.get(get_outputs()[0]).push_back(ranges);
}
void reshape_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = fdmap.get(info_.inputs_[0]);
    in_detail0.use_count_++;
}
void reshape_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(
            info_.outputs_[0]->details_.get_plain_dims().size())});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}
void reshape_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dsts,
        const std::vector<const tensor_slice *> &inputs) {
    auto *src = inputs[0];
    auto *dst = dsts[0];
    // accumulate src tensor size
    std::vector<expr> src_accsize {expr(dim2unsigned(1))};
    expr src_size = expr(dim2unsigned(1));
    // accumulate dst tensor size
    std::vector<expr> dst_accsize {expr(dim2unsigned(1))};
    expr dst_size = expr(dim2unsigned(1));
    // outer nested loop vars
    expr iters = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
    // the indices for the input tensor.
    std::vector<expr> src_idx(src->nslice_dims());
    // the indices for the output tensor.
    std::vector<expr> dst_idx(dst->nslice_dims());
    uint64_t total_size = 1;
    for (auto it : shapes_) {
        total_size *= it;
    }
    for (int64_t i = inputs.at(0)->nslice_dims() - 1; i > 0; i--) {
        src_size = src_size * src->get_shape()[i];
        src_accsize.emplace_back(src_size);
    }

    for (auto i = dst->nslice_dims() - 1; i > 0; i--) {
        dst_size = dst_size * dst->get_shape()[i];
        dst_accsize.emplace_back(dst_size);
    }
    std::reverse(src_accsize.begin(), src_accsize.end());
    std::reverse(dst_accsize.begin(), dst_accsize.end());

    for (int i = 0; i < (int)src->nslice_dims(); i++) {
        if (i == 0) {
            src_idx[i] = iters / src_accsize[i] + src->get_offset()[i];
        } else {
            src_idx[i] = iters % src_accsize[i - 1] / src_accsize[i]
                    + src->get_offset()[i];
        }
    }
    for (int i = 0; i < (int)dst->nslice_dims(); i++) {
        if (i == 0) {
            dst_idx[i] = iters / dst_accsize[i] + dst->get_offset()[i];
        } else {
            dst_idx[i] = iters % dst_accsize[i - 1] / dst_accsize[i]
                    + dst->get_offset()[i];
        }
    }
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");

    expr indexed_target = builder::make_indexing(dst->tptr_, {dst_idx});
    expr indexed_input = builder::make_indexing(src->tptr_, src_idx);
    stmt_c cur = builder::make_assign_unattached(indexed_target, indexed_input);
    auto body = builder::make_stmts_unattached(
            std::vector<stmt_c> {std::move(cur)});
    cur = builder::make_for_loop_unattached(iters, expr(0), expr(total_size),
            expr(1), std::move(body), true, for_type::NORMAL);
    constant_folder_t folder;
    cur = folder(cur);
    bld->emit(cur.remove_const());
}

split_op_t::split_op_t(graph_tensor_ptr v, int dim, const sc_dims &shapes)
    : dim_(dim), shapes_(shapes) {
    attrs_.set(op_attr_key::no_fuse, true);
    info_.inputs_.emplace_back(std::move(v));
    for (unsigned i = 0; i < shapes_.size(); i++) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
    }
}

void split_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.reserve(info_.outputs_.size());
    for (size_t i = 0; i < out_formats.size(); ++i) {
        out_formats[i].push_back({info_.inputs_[0]->details_.get_format()});
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void split_op_t::prepare_fusion_data(fdata_map &fdmap) {
    auto &in_detail0 = info_.inputs_[0];
    fdmap.get(in_detail0).use_count_++;
    COMPILE_ASSERT(info_.outputs_.size() > 1,
            "Split op output size should bigger than 1.\n");
    auto dims = in_detail0->details_.get_blocking_dims();
    auto dims_size = dims.size();
    COMPILE_ASSERT(dims_size > dim_, "Split dim is not available.\n");
    sc_dim total_split = 0;
    for (auto num : shapes_) {
        total_split += num;
    }
    COMPILE_ASSERT(total_split == dims[dim_],
            "Split shapes are not matched with input.\n");
    for (unsigned i = 0; i < info_.outputs_.size(); i++) {
        auto &output = info_.outputs_[i];
        sc_dims out_dims(dims_size);
        std::vector<expr> tmp_shape;
        auto &outdetail = fdmap.get(output);
        for (unsigned j = 0; j < dims_size; j++) {
            if (j != dim_) {
                tmp_shape.emplace_back(dim2unsigned(dims[j]));
                out_dims.emplace_back(
                        info_.inputs_[0]->details_.get_blocking_dims()[j]);
            } else {
                tmp_shape.emplace_back(dim2unsigned(shapes_[i]));
                out_dims.emplace_back(shapes_[i]);
            }
        }
    }
}

void split_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    if (known_ranges_map.empty()) return;
    size_t slice_size = known_ranges_map[0].size();
    slice_range_list split_ranges_list = known_ranges_map[0];
    for (size_t i = 0; i < get_outputs().size(); i++) {
        fsmap.get(get_outputs()[i]).resize(slice_size);
        for (size_t n = 0; n < slice_size; n++) {
            for (size_t j = 0; j < split_ranges_list.at(n).size(); j++) {
                if (j == dim_) {
                    // Due to query stage, split shapes should be matched
                    // with input.
                    fsmap.get(get_outputs()[i])
                            .at(n)
                            .emplace_back(std::make_pair(
                                    expr(0), dim2unsigned(shapes_.at(i))));
                } else {
                    fsmap.get(get_outputs()[i])
                            .at(n)
                            .emplace_back(std::make_pair(
                                    split_ranges_list.at(n).at(j).first,
                                    split_ranges_list.at(n).at(j).second));
                }
            }
        }
    }
}

void split_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

void compute_block_split(const std::vector<const tensor_slice *> &src,
        const std::vector<tensor_slice *> &dst, unsigned dim,
        const sc_dims &shapes, size_t wkld = 0UL) {
    // outer nested loop vars
    std::vector<expr> outer_iter(dim);
    // inner nested loop vars
    std::vector<std::vector<expr>> inner_iter(
            static_cast<uint64_t>(src[0]->nbase_dims()) - dim);
    // the indices for multiple inputs. First dim: the input, Second
    // dim: the dimemsions in the tensor
    std::vector<std::vector<expr>> src_idx(dst.size());
    // the indices for the output tensor. Cause concat is a assign op,
    // we need number of src indexes.
    std::vector<std::vector<expr>> dst_idx(dst.size());
    for (int64_t i = 0; i < src[0]->nbase_dims(); i++) {
        if (i < dim) { // outer loop
            // make the loop var for the for-loop
            outer_iter[i] = builder::make_var(datatypes::index,
                    std::string("_fuseiter") + fusion_create_idx());
            for (unsigned j = 0; j < dst.size(); j++) {
                src_idx[j].emplace_back(outer_iter[i]);
                dst_idx[j].emplace_back(outer_iter[i]);
            }
        } else { // inner loop
            expr cur;
            for (unsigned j = 0; j < dst.size(); j++) {
                inner_iter[i - dim].emplace_back(builder::make_var(
                        datatypes::index,
                        std::string("_fuseiter") + fusion_create_idx()));
                dst_idx[j].emplace_back(inner_iter[i - dim][j]);
                if (i == dim) {
                    if (j == 0) {
                        cur = 0;
                    } else {
                        cur = cur + dst[j - 1]->get_shape()[i];
                    }
                    src_idx[j].emplace_back(inner_iter[i - dim][j] + cur);
                } else {
                    src_idx[j].emplace_back(inner_iter[i - dim][j]);
                }
            }
        }
    }
    expr indexed_target;
    expr indexed_input;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    std::vector<stmt> tcur;
    for (unsigned j = 0; j < dst.size(); j++) {
        indexed_target = builder::make_indexing(dst[j]->tptr_, dst_idx[j]);
        indexed_input = builder::make_indexing(src[0]->tptr_, src_idx[j]);
        stmt cur = make_stmt<assign_node_t>(indexed_target, indexed_input);
        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
        for (int64_t i = src[0]->nslice_dims() - 1; i >= dim; i--) {
            auto body = make_stmt<stmts_node_t>(
                    std::vector<stmt> {std::move(cur)});
            cur = make_stmt<for_loop_node_t>(inner_iter[i - dim][j], expr(0),
                    dst[j]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        tcur.emplace_back(std::move(cur));
    }
    if (dim) {
        stmt cur = make_stmt<stmts_node_t>(std::move(tcur));
        for (int i = dim - 1; i >= 0; i--) {
            stmt body;
            if (cur.isa<for_loop>()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
            } else {
                body = cur;
            }
            cur = make_stmt<for_loop_node_t>(outer_iter[i], expr(0),
                    src[0]->get_shape()[i], expr(1), std::move(body), true,
                    for_type::NORMAL);
        }
        bld->emit(cur);
    } else {
        for (auto &cur : tcur) {
            bld->emit(cur);
        }
    }
}

void split_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    compute_block_split(inputs, dst, dim_, shapes_, wkld);
}

OP_REGISTER(transpose_op_t, transpose)
OP_REGISTER(tensor_view_op_t, tensor_view)
OP_REGISTER(reshape_op_t, reshape)
OP_REGISTER(reorder_op_t, reorder)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
