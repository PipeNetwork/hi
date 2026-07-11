#[cfg(feature = "native-cuda")]
mod native {
    use std::ffi::c_void;
    use std::os::raw::{c_float, c_int};

    use anyhow::{Result, bail};

    use crate::runtime::{DeviceBuffer, Stream, check_last_error};

    unsafe extern "C" {
        fn hi_cuda_launch_rms_norm(
            input: *const c_void,
            weight: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            eps: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_layer_norm(
            input: *const c_void,
            weight: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            eps: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_silu_mul_f32_f16(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            output_f16: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_silu_mul(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gelu(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gelu_mul(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_softcap(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            cap: f32,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_add(
            left: *const c_void,
            right: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_add_rowwise(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_copy_row_f32(
            input: *const c_void,
            output: *mut c_void,
            row: c_int,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_add_scaled_row_in_place(
            output: *mut c_void,
            row_values: *const c_void,
            row: c_int,
            rows: c_int,
            cols: c_int,
            scale: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_qwen_ssm_streaming_step(
            qkv: *const c_void,
            gate: *const c_void,
            conv_weight: *const c_void,
            ba: *const c_void,
            ba_alpha: *const c_void,
            dt_bias: *const c_void,
            a_log: *const c_void,
            norm_weight: *const c_void,
            conv_ring: *mut c_void,
            recurrent_state: *mut c_void,
            output: *mut c_void,
            conv_next: c_int,
            conv_len: c_int,
            conv_kernel: c_int,
            conv_dim: c_int,
            state_size: c_int,
            time_step_rank: c_int,
            group_count: c_int,
            head_v_dim: c_int,
            packed_qkvz: c_int,
            kv_group_round_robin: c_int,
            eps: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_moe_topk_router(
            scores: *const c_void,
            output_ids: *mut c_void,
            output_weights: *mut c_void,
            rows: c_int,
            experts: c_int,
            top_k: c_int,
            norm_topk: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cast_f32_to_f16(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q4_0_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q4_k_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q6_k_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q5_k_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_iq4_nl_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q8_0_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q2_k_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_q3_k_to_f16(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cast_f32_to_bf16(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cast_f16_to_f32(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cast_bf16_to_f32(
            input: *const c_void,
            output: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gather_rows_f16_to_f32(
            matrix: *const c_void,
            row_ids: *const c_void,
            output: *mut c_void,
            row_count: c_int,
            cols: c_int,
            matrix_rows: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gather_rows_bf16_to_f32(
            matrix: *const c_void,
            row_ids: *const c_void,
            output: *mut c_void,
            row_count: c_int,
            cols: c_int,
            matrix_rows: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gather_rows_f32_to_f32(
            matrix: *const c_void,
            row_ids: *const c_void,
            output: *mut c_void,
            row_count: c_int,
            cols: c_int,
            matrix_rows: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gather_quant_rows(
            matrix: *const c_void,
            row_ids: *const c_void,
            output: *mut c_void,
            row_count: c_int,
            row_bytes: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_dequantize_matrix(
            input: *const c_void,
            output: *mut c_void,
            elements: c_int,
            quant_type: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_quantize_q8_row(
            x: *const c_void,
            xq: *mut c_void,
            dx: *mut c_void,
            xsum: *mut c_void,
            k: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_quantize_q8_rows(
            x: *const c_void,
            xq: *mut c_void,
            dx: *mut c_void,
            xsum: *mut c_void,
            m: c_int,
            k: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gqa_paged_decode_attention(
            q8: c_int,
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            out_or_partials: *mut c_void,
            positions: *const c_void,
            d_position: *const c_void,
            position: c_int,
            batch_count: c_int,
            split_count: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_gqa_split_decode_merge(
            partials: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            heads: c_int,
            split_count: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_kquant_dp4a_gemm(
            dtype: c_int,
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            m: c_int,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_moe_grouped_dp4a_gemv(
            dtype: c_int,
            expert_ptrs: *const c_void,
            route_ids: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            pairs: c_int,
            top_k: c_int,
            act_per_pair: c_int,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_moe_scatter_reduce(
            down: *const c_void,
            route_weights: *const c_void,
            out: *mut c_void,
            rows: c_int,
            top_k: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_moe_add_rows_scaled_by_sigmoid(
            values: *const c_void,
            gates: *const c_void,
            out: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q4_k_a8_gemm(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            out: *mut c_void,
            m: c_int,
            n: c_int,
            k: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q4_0_dp4a_gemv(
            weight: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_mxfp4_gemv(
            weight: *const c_void,
            x: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_nvfp4_gemv(
            weight: *const c_void,
            x: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q4_k_dp4a_gemv(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q5_k_dp4a_gemv(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q6_k_dp4a_gemv(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q2_k_dp4a_gemv(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q3_k_dp4a_gemv(
            weights: *const c_void,
            xq: *const c_void,
            dx: *const c_void,
            xsum: *const c_void,
            y: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q6_k_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q4_k_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q5_k_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q3_k_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_q2_k_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq4_nl_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq4_xs_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq3_s_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq2_xxs_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq2_s_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq2_xs_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq1_s_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq1_m_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_iq3_xxs_gemv(
            weights: *const c_void,
            x: *const c_void,
            output: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_rope(
            values: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            rot_dim: c_int,
            base: c_float,
            scale: c_float,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_rope_with_offset(
            values: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            rot_dim: c_int,
            base: c_float,
            scale: c_float,
            position_offset: c_int,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_rope_batched_with_offset(
            values: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            rot_dim: c_int,
            base: c_float,
            scale: c_float,
            position_offset: c_int,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_rope_batched_with_offset_devpos(
            values: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            rot_dim: c_int,
            base: c_float,
            scale: c_float,
            d_position_offset: *const c_void,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_rope_batched_positions(
            values: *mut c_void,
            positions: *const c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            rot_dim: c_int,
            base: c_float,
            scale: c_float,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_mrope(
            values: *mut c_void,
            pos_t: *const c_void,
            pos_h: *const c_void,
            pos_w: *const c_void,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            base: c_float,
            scale: c_float,
            section_t: c_int,
            section_h: c_int,
            section_w: c_int,
            section_e: c_int,
            split_half: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_vision_rope(
            values: *mut c_void,
            pos_h: *const c_void,
            pos_w: *const c_void,
            seq_len: c_int,
            heads: c_int,
            head_dim: c_int,
            base: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_kv_cache(
            values: *const c_void,
            cache: *mut c_void,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            max_seq: c_int,
            start_pos: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_kv_cache_batched(
            values: *const c_void,
            cache: *mut c_void,
            batch_count: c_int,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            max_seq: c_int,
            start_pos: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_paged_kv_cache(
            values: *const c_void,
            pages: *mut c_void,
            page_table: *const c_void,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            start_pos: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_paged_kv_cache_batched(
            values: *const c_void,
            pages: *mut c_void,
            page_table: *const c_void,
            batch_count: c_int,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            start_pos: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_paged_kv_cache_batched_devpos(
            values: *const c_void,
            pages: *mut c_void,
            page_table: *const c_void,
            batch_count: c_int,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            d_start_pos: *const c_void,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_paged_kv_cache_batched_positions(
            values: *const c_void,
            pages: *mut c_void,
            page_table: *const c_void,
            positions: *const c_void,
            batch_count: c_int,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_write_paged_kv_cache_q8_batched(
            values: *const c_void,
            pages: *mut c_void,
            scales: *mut c_void,
            page_table: *const c_void,
            positions: *const c_void,
            d_start_pos: *const c_void,
            start_pos: c_int,
            batch_count: c_int,
            row_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_copy_paged_kv_cache_prefix_batched(
            pages: *mut c_void,
            page_table: *const c_void,
            batch_count: c_int,
            token_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_copy_paged_kv_cache_prefix_batched_q8(
            pages: *mut c_void,
            scales: *mut c_void,
            page_table: *const c_void,
            batch_count: c_int,
            token_count: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            page_size: c_int,
            page_table_len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_causal_attention(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_causal_attention_batched(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_causal_attention(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_causal_attention(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_causal_attention_batched(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_causal_attention_batched(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flashtile_causal_attention_batched(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_paged_prefill_causal_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_paged_prefill_causal_attention_batched_q8(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_wmma_paged_prefill_causal_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_wmma_paged_prefill_causal_attention_batched_q8(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_fa2_paged_prefill_causal_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_fa2_paged_prefill_causal_attention_batched_q8(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            query_offset: c_int,
            batch_count: c_int,
            chunk_len: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_wmma_causal_attention_batched(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_full_attention(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_window_attention(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            window_start: *const c_void,
            window_end: *const c_void,
            output: *mut c_void,
            seq_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cached_decode_attention(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            output: *mut c_void,
            position: c_int,
            max_seq: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_cached_decode_attention(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            output: *mut c_void,
            position: c_int,
            max_seq: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_paged_decode_attention(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_paged_decode_attention(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention_q8(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_paged_decode_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_paged_decode_attention_batched_positions(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            positions: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_paged_decode_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention_batched(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            position: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention_batched_devpos(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            d_position: *const c_void,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention_batched_positions(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            page_table: *const c_void,
            positions: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_tiled_paged_decode_attention_batched_positions_q8(
            q: *const c_void,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const c_void,
            v_scales: *const c_void,
            page_table: *const c_void,
            positions: *const c_void,
            d_start_pos: *const c_void,
            start_pos: c_int,
            output: *mut c_void,
            batch_count: c_int,
            page_size: c_int,
            page_table_len: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            window: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_cached_decode_attention_batched(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            position: c_int,
            max_seq: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_flash_cached_decode_attention_batched(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            output: *mut c_void,
            batch_count: c_int,
            position: c_int,
            max_seq: c_int,
            heads: c_int,
            kv_heads: c_int,
            qk_head_dim: c_int,
            v_head_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_argmax(
            logits: *const c_void,
            output_token: *mut c_void,
            len: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_argmax_last_row(
            logits: *const c_void,
            output_token: *mut c_void,
            rows: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_argmax_batched_last_token(
            logits: *const c_void,
            output_tokens: *mut c_void,
            batch_count: c_int,
            seq_len: c_int,
            cols: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_sample_last_row(
            logits: *const c_void,
            output_token: *mut c_void,
            rows: c_int,
            cols: c_int,
            temperature: c_float,
            top_p: c_float,
            top_k: c_int,
            sample: c_float,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_sample_batched_last_token(
            logits: *const c_void,
            output_tokens: *mut c_void,
            samples: *const c_void,
            batch_count: c_int,
            seq_len: c_int,
            cols: c_int,
            temperature: c_float,
            top_p: c_float,
            top_k: c_int,
            gpu_ranked: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn hi_cuda_launch_select_batched_last_token_per_row(
            logits: *const c_void,
            output_tokens: *mut c_void,
            samples: *const c_void,
            temperatures: *const c_void,
            top_ps: *const c_void,
            top_ks: *const c_void,
            batch_count: c_int,
            seq_len: c_int,
            cols: c_int,
            gpu_ranked: c_int,
            stream: *mut c_void,
        ) -> c_int;
    }

    pub fn launch_rms_norm(
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "rms_norm rows")?;
        ensure_len(cols, "rms_norm cols")?;
        launch_status(unsafe {
            hi_cuda_launch_rms_norm(
                input.as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                eps,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rms_norm")
    }

    pub fn launch_layer_norm(
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        bias: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "layer_norm rows")?;
        ensure_len(cols, "layer_norm cols")?;
        launch_status(unsafe {
            hi_cuda_launch_layer_norm(
                input.as_ptr(),
                weight.as_ptr(),
                bias.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                eps,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_layer_norm")
    }

    pub fn launch_silu_mul(
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "silu_mul len")?;
        launch_status(unsafe {
            hi_cuda_launch_silu_mul(
                gate.as_ptr(),
                up.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_silu_mul")
    }

    /// SwiGLU with a fused f16 copy of the output (see the kernel comment):
    /// `output` is f32 `[len]`, `output_f16` is half `[len]`.
    pub fn launch_silu_mul_f32_f16(
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        output: &DeviceBuffer,
        output_f16: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "silu_mul_f32_f16 len")?;
        launch_status(unsafe {
            hi_cuda_launch_silu_mul_f32_f16(
                gate.as_ptr(),
                up.as_ptr(),
                output.as_mut_ptr(),
                output_f16.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_silu_mul_f32_f16")
    }

    pub fn launch_gelu(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "gelu len")?;
        launch_status(unsafe {
            hi_cuda_launch_gelu(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gelu")
    }

    pub fn launch_gelu_mul(
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "gelu_mul len")?;
        launch_status(unsafe {
            hi_cuda_launch_gelu_mul(
                gate.as_ptr(),
                up.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gelu_mul")
    }

    pub fn launch_softcap(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        cap: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "softcap len")?;
        launch_status(unsafe {
            hi_cuda_launch_softcap(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                cap,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_softcap")
    }

    pub fn launch_add(
        left: &DeviceBuffer,
        right: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "add len")?;
        launch_status(unsafe {
            hi_cuda_launch_add(
                left.as_ptr(),
                right.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_add")
    }

    pub fn launch_add_rowwise(
        input: &DeviceBuffer,
        bias: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "add_rowwise rows")?;
        ensure_len(cols, "add_rowwise cols")?;
        launch_status(unsafe {
            hi_cuda_launch_add_rowwise(
                input.as_ptr(),
                bias.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_add_rowwise")
    }

    pub fn launch_copy_row_f32(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        row: usize,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row, "copy_row row")?;
        ensure_len(rows, "copy_row rows")?;
        ensure_len(cols, "copy_row cols")?;
        launch_status(unsafe {
            hi_cuda_launch_copy_row_f32(
                input.as_ptr(),
                output.as_mut_ptr(),
                row as c_int,
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_copy_row_f32")
    }

    pub fn launch_add_scaled_row_in_place(
        output: &DeviceBuffer,
        row_values: &DeviceBuffer,
        row: usize,
        rows: usize,
        cols: usize,
        scale: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row, "add_scaled_row row")?;
        ensure_len(rows, "add_scaled_row rows")?;
        ensure_len(cols, "add_scaled_row cols")?;
        launch_status(unsafe {
            hi_cuda_launch_add_scaled_row_in_place(
                output.as_mut_ptr(),
                row_values.as_ptr(),
                row as c_int,
                rows as c_int,
                cols as c_int,
                scale,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_add_scaled_row_in_place")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_qwen_ssm_streaming_step(
        qkv: &DeviceBuffer,
        gate: Option<&DeviceBuffer>,
        conv_weight: &DeviceBuffer,
        ba: &DeviceBuffer,
        ba_alpha: Option<&DeviceBuffer>,
        dt_bias: &DeviceBuffer,
        a_log: &DeviceBuffer,
        norm_weight: &DeviceBuffer,
        conv_ring: &DeviceBuffer,
        recurrent_state: &DeviceBuffer,
        output: &DeviceBuffer,
        conv_next: usize,
        conv_len: usize,
        conv_kernel: usize,
        conv_dim: usize,
        state_size: usize,
        time_step_rank: usize,
        group_count: usize,
        head_v_dim: usize,
        packed_qkvz: bool,
        kv_group_round_robin: bool,
        eps: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(conv_next, "qwen_ssm conv_next")?;
        ensure_len(conv_len, "qwen_ssm conv_len")?;
        ensure_len(conv_kernel, "qwen_ssm conv_kernel")?;
        ensure_len(conv_dim, "qwen_ssm conv_dim")?;
        ensure_len(state_size, "qwen_ssm state_size")?;
        ensure_len(time_step_rank, "qwen_ssm time_step_rank")?;
        ensure_len(group_count, "qwen_ssm group_count")?;
        ensure_len(head_v_dim, "qwen_ssm head_v_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_qwen_ssm_streaming_step(
                qkv.as_ptr(),
                gate.map_or(std::ptr::null(), DeviceBuffer::as_ptr),
                conv_weight.as_ptr(),
                ba.as_ptr(),
                ba_alpha.map_or(std::ptr::null(), DeviceBuffer::as_ptr),
                dt_bias.as_ptr(),
                a_log.as_ptr(),
                norm_weight.as_ptr(),
                conv_ring.as_mut_ptr(),
                recurrent_state.as_mut_ptr(),
                output.as_mut_ptr(),
                conv_next as c_int,
                conv_len as c_int,
                conv_kernel as c_int,
                conv_dim as c_int,
                state_size as c_int,
                time_step_rank as c_int,
                group_count as c_int,
                head_v_dim as c_int,
                if packed_qkvz { 1 } else { 0 },
                if kv_group_round_robin { 1 } else { 0 },
                eps,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_qwen_ssm_streaming_step")
    }

    pub fn launch_moe_topk_router(
        scores: &DeviceBuffer,
        output_ids: &DeviceBuffer,
        output_weights: &DeviceBuffer,
        rows: usize,
        experts: usize,
        top_k: usize,
        norm_topk: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "moe_topk rows")?;
        ensure_len(experts, "moe_topk experts")?;
        ensure_len(top_k, "moe_topk top_k")?;
        launch_status(unsafe {
            hi_cuda_launch_moe_topk_router(
                scores.as_ptr(),
                output_ids.as_mut_ptr(),
                output_weights.as_mut_ptr(),
                rows as c_int,
                experts as c_int,
                top_k as c_int,
                if norm_topk { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_moe_topk_router")
    }

    pub fn launch_cast_f32_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "cast_f32_to_f16 len")?;
        launch_status(unsafe {
            hi_cuda_launch_cast_f32_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cast_f32_to_f16")
    }

    /// Dequantize a Q4_0 weight matrix straight to f16 (no f32 intermediate + cast).
    /// `elements` is rows*cols; `output` must hold `elements` f16 values.
    pub fn launch_dequantize_q4_0_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q4_0_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q4_0_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q4_0_to_f16")
    }

    /// Dequantize a Q4_K weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q4_k_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q4_k_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q4_k_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q4_k_to_f16")
    }

    /// Dequantize a Q6_K weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q6_k_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q6_k_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q6_k_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q6_k_to_f16")
    }

    /// Dequantize a Q5_K weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q5_k_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q5_k_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q5_k_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q5_k_to_f16")
    }

    /// Dequantize an IQ4_NL weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_iq4_nl_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_iq4_nl_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_iq4_nl_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_iq4_nl_to_f16")
    }

    /// Dequantize a Q8_0 weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q8_0_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q8_0_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q8_0_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q8_0_to_f16")
    }

    /// Dequantize a Q2_K weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q2_k_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q2_k_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q2_k_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q2_k_to_f16")
    }

    /// Dequantize a Q3_K weight matrix straight to f16 (no f32 intermediate + cast).
    pub fn launch_dequantize_q3_k_to_f16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize_q3_k_to_f16 elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_q3_k_to_f16(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_q3_k_to_f16")
    }

    pub fn launch_cast_f32_to_bf16(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "cast_f32_to_bf16 len")?;
        launch_status(unsafe {
            hi_cuda_launch_cast_f32_to_bf16(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cast_f32_to_bf16")
    }

    pub fn launch_cast_f16_to_f32(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "cast_f16_to_f32 len")?;
        launch_status(unsafe {
            hi_cuda_launch_cast_f16_to_f32(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cast_f16_to_f32")
    }

    pub fn launch_cast_bf16_to_f32(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "cast_bf16_to_f32 len")?;
        launch_status(unsafe {
            hi_cuda_launch_cast_bf16_to_f32(
                input.as_ptr(),
                output.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cast_bf16_to_f32")
    }

    pub fn launch_gather_rows_f16_to_f32(
        matrix: &DeviceBuffer,
        row_ids: &DeviceBuffer,
        output: &DeviceBuffer,
        row_count: usize,
        cols: usize,
        matrix_rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "gather row_count")?;
        ensure_len(cols, "gather cols")?;
        ensure_len(matrix_rows, "gather matrix_rows")?;
        launch_status(unsafe {
            hi_cuda_launch_gather_rows_f16_to_f32(
                matrix.as_ptr(),
                row_ids.as_ptr(),
                output.as_mut_ptr(),
                row_count as c_int,
                cols as c_int,
                matrix_rows as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gather_rows_f16_to_f32")
    }

    pub fn launch_gather_rows_bf16_to_f32(
        matrix: &DeviceBuffer,
        row_ids: &DeviceBuffer,
        output: &DeviceBuffer,
        row_count: usize,
        cols: usize,
        matrix_rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "gather row_count")?;
        ensure_len(cols, "gather cols")?;
        ensure_len(matrix_rows, "gather matrix_rows")?;
        launch_status(unsafe {
            hi_cuda_launch_gather_rows_bf16_to_f32(
                matrix.as_ptr(),
                row_ids.as_ptr(),
                output.as_mut_ptr(),
                row_count as c_int,
                cols as c_int,
                matrix_rows as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gather_rows_bf16_to_f32")
    }

    pub fn launch_gather_rows_f32_to_f32(
        matrix: &DeviceBuffer,
        row_ids: &DeviceBuffer,
        output: &DeviceBuffer,
        row_count: usize,
        cols: usize,
        matrix_rows: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "gather row_count")?;
        ensure_len(cols, "gather cols")?;
        ensure_len(matrix_rows, "gather matrix_rows")?;
        launch_status(unsafe {
            hi_cuda_launch_gather_rows_f32_to_f32(
                matrix.as_ptr(),
                row_ids.as_ptr(),
                output.as_mut_ptr(),
                row_count as c_int,
                cols as c_int,
                matrix_rows as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gather_rows_f32_to_f32")
    }

    /// Gather whole quantized rows (`row_bytes` each) into a compact buffer, so the
    /// caller can dequantize only the gathered rows instead of the full matrix.
    pub fn launch_gather_quant_rows(
        matrix: &DeviceBuffer,
        row_ids: &DeviceBuffer,
        output: &DeviceBuffer,
        row_count: usize,
        row_bytes: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "gather_quant row_count")?;
        ensure_len(row_bytes, "gather_quant row_bytes")?;
        launch_status(unsafe {
            hi_cuda_launch_gather_quant_rows(
                matrix.as_ptr(),
                row_ids.as_ptr(),
                output.as_mut_ptr(),
                row_count as c_int,
                row_bytes as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gather_quant_rows")
    }

    pub fn launch_quantize_q8_row(
        x: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(k, "quantize_q8 k")?;
        launch_status(unsafe {
            hi_cuda_launch_quantize_q8_row(
                x.as_ptr(),
                xq.as_mut_ptr(),
                dx.as_mut_ptr(),
                xsum.as_mut_ptr(),
                k as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_quantize_q8_row")
    }

    /// Batched per-32-block int8 activation quant for M rows (W4A8 prefill GEMM): produces
    /// xq[M,K] int8, dx[M,K/32] block scales, xsum[M,K/32] block int sums.
    pub fn launch_quantize_q8_rows(
        x: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        m: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(m, "quantize_q8_rows m")?;
        ensure_len(k, "quantize_q8_rows k")?;
        launch_status(unsafe {
            hi_cuda_launch_quantize_q8_rows(
                x.as_ptr(),
                xq.as_mut_ptr(),
                dx.as_mut_ptr(),
                xsum.as_mut_ptr(),
                m as c_int,
                k as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_quantize_q8_rows")
    }

    /// Expert weight dtype for the grouped MoE dp4a GEMV.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MoeGroupedGemvDtype {
        Q4K = 0,
        Q6K = 1,
    }

    /// GQA-grouped (+ optional grid-split) paged decode attention. One block per
    /// (kv_head, batch row, split chunk) serves all grouped Q heads, reading each
    /// K/V vector once. Position resolves per-row (`positions`), from the device
    /// counter (`d_position`), or the host scalar. With `split_count` == 1 writes
    /// `out_or_partials` as the attention output; otherwise writes per-chunk
    /// flash partials for `launch_gqa_split_decode_merge`. Returns Ok(false) when
    /// the (kv_repeats, head_dim) combination has no kernel bucket — the caller
    /// falls back to the per-head kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_gqa_paged_decode_attention(
        q8: bool,
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: Option<&DeviceBuffer>,
        v_scales: Option<&DeviceBuffer>,
        page_table: &DeviceBuffer,
        out_or_partials: &DeviceBuffer,
        positions: Option<&DeviceBuffer>,
        d_position: Option<&DeviceBuffer>,
        position: usize,
        batch_count: usize,
        split_count: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        qk_head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<bool> {
        ensure_len(position, "gqa_decode position")?;
        ensure_len(batch_count, "gqa_decode batch_count")?;
        ensure_len(split_count, "gqa_decode split_count")?;
        ensure_len(page_size, "gqa_decode page_size")?;
        ensure_len(page_table_len, "gqa_decode page_table_len")?;
        ensure_len(heads, "gqa_decode heads")?;
        ensure_len(kv_heads, "gqa_decode kv_heads")?;
        ensure_len(qk_head_dim, "gqa_decode qk_head_dim")?;
        ensure_len(v_head_dim, "gqa_decode v_head_dim")?;
        ensure_len(window, "gqa_decode window")?;
        let null = std::ptr::null();
        let status = unsafe {
            hi_cuda_launch_gqa_paged_decode_attention(
                c_int::from(q8),
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.map_or(null, |buffer| buffer.as_ptr()),
                v_scales.map_or(null, |buffer| buffer.as_ptr()),
                page_table.as_ptr(),
                out_or_partials.as_mut_ptr(),
                positions.map_or(null, |buffer| buffer.as_ptr()),
                d_position.map_or(null, |buffer| buffer.as_ptr()),
                position as c_int,
                batch_count as c_int,
                split_count as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                qk_head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        };
        if status == 2 {
            return Ok(false);
        }
        launch_status(status)?;
        check_last_error("hi_cuda_launch_gqa_paged_decode_attention")?;
        Ok(true)
    }

    pub fn launch_gqa_split_decode_merge(
        partials: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        heads: usize,
        split_count: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "gqa_merge batch_count")?;
        ensure_len(heads, "gqa_merge heads")?;
        ensure_len(split_count, "gqa_merge split_count")?;
        ensure_len(v_head_dim, "gqa_merge v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_gqa_split_decode_merge(
                partials.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                heads as c_int,
                split_count as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_gqa_split_decode_merge")
    }

    /// dp4a K-quant GEMM for small M (2..=32): each decoded weight sub-block is
    /// dotted against ALL M activation rows, so quantized weights stream once for
    /// the whole batch. Per output row bit-identical to the M=1 dp4a GEMV.
    /// Activations are the [m, cols] int8 rows from `launch_quantize_q8_rows`;
    /// y is [m, rows].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kquant_dp4a_gemm(
        dtype: MoeGroupedGemvDtype,
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        m: usize,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(m, "kquant_gemm m")?;
        ensure_len(rows, "kquant_gemm rows")?;
        ensure_len(cols, "kquant_gemm cols")?;
        launch_status(unsafe {
            hi_cuda_launch_kquant_dp4a_gemm(
                dtype as c_int,
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                m as c_int,
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_kquant_dp4a_gemm")
    }

    /// One launch for every (token row, routed expert) pair of a MoE projection:
    /// `expert_ptrs` is a device table of expert weight base addresses (u64),
    /// `route_ids[pairs]` selects the expert per pair, activations are the per-32
    /// int8 rows from `launch_quantize_q8_rows` (`act_per_pair`: 0 = gate/up read
    /// token row pair/top_k, 1 = down reads activated row pair). y is [pairs, rows].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_grouped_dp4a_gemv(
        dtype: MoeGroupedGemvDtype,
        expert_ptrs: &DeviceBuffer,
        route_ids: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        pairs: usize,
        top_k: usize,
        act_per_pair: bool,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(pairs, "moe_grouped_gemv pairs")?;
        ensure_len(top_k, "moe_grouped_gemv top_k")?;
        ensure_len(rows, "moe_grouped_gemv rows")?;
        ensure_len(cols, "moe_grouped_gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_moe_grouped_dp4a_gemv(
                dtype as c_int,
                expert_ptrs.as_ptr(),
                route_ids.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                pairs as c_int,
                top_k as c_int,
                c_int::from(act_per_pair),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_moe_grouped_dp4a_gemv")
    }

    /// out[row] += sum over the row's top_k pairs of route_weight * down_row, in
    /// rank order (matching the sequential per-expert accumulation it replaces).
    pub fn launch_moe_scatter_reduce(
        down: &DeviceBuffer,
        route_weights: &DeviceBuffer,
        out: &DeviceBuffer,
        rows: usize,
        top_k: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "moe_scatter_reduce rows")?;
        ensure_len(top_k, "moe_scatter_reduce top_k")?;
        ensure_len(cols, "moe_scatter_reduce cols")?;
        launch_status(unsafe {
            hi_cuda_launch_moe_scatter_reduce(
                down.as_ptr(),
                route_weights.as_ptr(),
                out.as_mut_ptr(),
                rows as c_int,
                top_k as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_moe_scatter_reduce")
    }

    /// out[row] += sigmoid(gates[row]) * values[row] (gates None = scale 1), for
    /// the MoE shared expert; keeps the per-row scalar gate on the device.
    pub fn launch_moe_add_rows_scaled_by_sigmoid(
        values: &DeviceBuffer,
        gates: Option<&DeviceBuffer>,
        out: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "moe_add_sigmoid rows")?;
        ensure_len(cols, "moe_add_sigmoid cols")?;
        launch_status(unsafe {
            hi_cuda_launch_moe_add_rows_scaled_by_sigmoid(
                values.as_ptr(),
                gates.map_or(std::ptr::null(), |buffer| buffer.as_ptr()),
                out.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_moe_add_rows_scaled_by_sigmoid")
    }

    /// W4A8 prefill GEMM: C[M,N] f32 = A[M,K] x W[N,K]^T with A int8 (per-32-block dx/xsum
    /// from `launch_quantize_q8_rows`) and W Q4_K, via int8 tensor cores + per-block rescale.
    /// Requires k % 256 == 0.
    ///
    /// EXPERIMENTAL / NOT WIRED INTO THE MODEL. Parity-validated (see
    /// `native_cuda_q4_k_a8_gemm_matches_dequant_reference`) but currently ~0.26x the speed of
    /// the shipping dequant->f16 + cuBLAS f16 path at the qwen3-8B projection shapes (see
    /// `bench_w4a8_vs_f16_gemm`): a hand-rolled tiled GEMM at ~3.4% of int8 peak can't beat
    /// cuBLAS's ~35% of f16 peak on this card. Kept as a validated foundation. To compete it
    /// needs cutlass/MMQ-class machinery — cp.async double/triple-buffered staging (sm_80+, so
    /// a multi-arch fatbin since hi-cuda ships compute_75 PTX), swizzled bank-conflict-free
    /// shared, warp specialization. Even then the end-to-end ceiling is ~9% of prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_q4_k_a8_gemm(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(m, "q4_k_a8_gemm m")?;
        ensure_len(n, "q4_k_a8_gemm n")?;
        ensure_len(k, "q4_k_a8_gemm k")?;
        launch_status(unsafe {
            hi_cuda_launch_q4_k_a8_gemm(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                out.as_mut_ptr(),
                m as c_int,
                n as c_int,
                k as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q4_k_a8_gemm")
    }

    pub fn launch_q4_0_dp4a_gemv(
        weight: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q4_0 dp4a rows")?;
        ensure_len(cols, "q4_0 dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q4_0_dp4a_gemv(
                weight.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q4_0_dp4a_gemv")
    }

    /// Fused MXFP4 GEMV (M=1 decode): reads the packed fp4 blocks natively against
    /// f32 activations. Requires cols % 32 == 0.
    pub fn launch_mxfp4_gemv(
        weight: &DeviceBuffer,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "mxfp4 gemv rows")?;
        ensure_len(cols, "mxfp4 gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_mxfp4_gemv(
                weight.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_mxfp4_gemv")
    }

    /// Fused NVFP4 GEMV (M=1 decode): reads the packed fp4 blocks natively against
    /// f32 activations. Requires cols % 64 == 0.
    pub fn launch_nvfp4_gemv(
        weight: &DeviceBuffer,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "nvfp4 gemv rows")?;
        ensure_len(cols, "nvfp4 gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_nvfp4_gemv(
                weight.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_nvfp4_gemv")
    }

    /// dp4a Q4_K GEMV (M=1 decode): reads Q4_K weights + int8-quantized activation
    /// (from `launch_quantize_q8_row`, block 32) via `__dp4a`. Requires cols % 256 == 0.
    pub fn launch_q4_k_dp4a_gemv(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q4_k dp4a rows")?;
        ensure_len(cols, "q4_k dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q4_k_dp4a_gemv(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q4_k_dp4a_gemv")
    }

    /// dp4a Q5_K GEMV (M=1 decode): Q5_K weights + int8-quantized activation via
    /// `__dp4a`. Requires cols % 256 == 0.
    pub fn launch_q5_k_dp4a_gemv(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q5_k dp4a rows")?;
        ensure_len(cols, "q5_k dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q5_k_dp4a_gemv(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q5_k_dp4a_gemv")
    }

    /// dp4a Q6_K GEMV (M=1 decode): Q6_K weights + int8-quantized activation via
    /// `__dp4a` (per-16 sums computed in-kernel). Requires cols % 256 == 0.
    pub fn launch_q6_k_dp4a_gemv(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q6_k dp4a rows")?;
        ensure_len(cols, "q6_k dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q6_k_dp4a_gemv(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q6_k_dp4a_gemv")
    }

    /// dp4a Q2_K GEMV (M=1 decode). Requires cols % 256 == 0.
    pub fn launch_q2_k_dp4a_gemv(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q2_k dp4a rows")?;
        ensure_len(cols, "q2_k dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q2_k_dp4a_gemv(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q2_k_dp4a_gemv")
    }

    /// dp4a Q3_K GEMV (M=1 decode). Requires cols % 256 == 0.
    pub fn launch_q3_k_dp4a_gemv(
        weights: &DeviceBuffer,
        xq: &DeviceBuffer,
        dx: &DeviceBuffer,
        xsum: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q3_k dp4a rows")?;
        ensure_len(cols, "q3_k dp4a cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q3_k_dp4a_gemv(
                weights.as_ptr(),
                xq.as_ptr(),
                dx.as_ptr(),
                xsum.as_ptr(),
                y.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q3_k_dp4a_gemv")
    }

    /// Fused Q6_K GEMV (M=1 decode): reads Q6_K weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_q6_k_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q6_k gemv rows")?;
        ensure_len(cols, "q6_k gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q6_k_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q6_k_gemv")
    }

    /// Fused Q4_K GEMV (M=1 decode): reads Q4_K weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_q4_k_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q4_k gemv rows")?;
        ensure_len(cols, "q4_k gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q4_k_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q4_k_gemv")
    }

    /// Fused Q5_K GEMV (M=1 decode): reads Q5_K weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_q5_k_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q5_k gemv rows")?;
        ensure_len(cols, "q5_k gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q5_k_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q5_k_gemv")
    }

    /// Fused Q3_K GEMV (M=1 decode): reads Q3_K weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_q3_k_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q3_k gemv rows")?;
        ensure_len(cols, "q3_k gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q3_k_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q3_k_gemv")
    }

    /// Fused Q2_K GEMV (M=1 decode): reads Q2_K weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_q2_k_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "q2_k gemv rows")?;
        ensure_len(cols, "q2_k gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_q2_k_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_q2_k_gemv")
    }

    /// Fused IQ4_NL GEMV (M=1 decode): reads IQ4_NL weights directly, f32 activation.
    /// Requires cols % 32 == 0.
    pub fn launch_iq4_nl_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq4_nl gemv rows")?;
        ensure_len(cols, "iq4_nl gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq4_nl_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq4_nl_gemv")
    }

    /// Fused IQ4_XS GEMV (M=1 decode): reads IQ4_XS weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq4_xs_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq4_xs gemv rows")?;
        ensure_len(cols, "iq4_xs gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq4_xs_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq4_xs_gemv")
    }

    /// Fused IQ3_S GEMV (M=1 decode): reads IQ3_S weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq3_s_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq3_s gemv rows")?;
        ensure_len(cols, "iq3_s gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq3_s_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq3_s_gemv")
    }

    /// Fused IQ2_XXS GEMV (M=1 decode): reads IQ2_XXS weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq2_xxs_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq2_xxs gemv rows")?;
        ensure_len(cols, "iq2_xxs gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq2_xxs_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq2_xxs_gemv")
    }

    /// Fused IQ2_S GEMV (M=1 decode): reads IQ2_S weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq2_s_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq2_s gemv rows")?;
        ensure_len(cols, "iq2_s gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq2_s_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq2_s_gemv")
    }

    /// Fused IQ2_XS GEMV (M=1 decode): reads IQ2_XS weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq2_xs_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq2_xs gemv rows")?;
        ensure_len(cols, "iq2_xs gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq2_xs_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq2_xs_gemv")
    }

    /// Fused IQ1_S GEMV (M=1 decode): reads IQ1_S weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq1_s_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq1_s gemv rows")?;
        ensure_len(cols, "iq1_s gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq1_s_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq1_s_gemv")
    }

    /// Fused IQ1_M GEMV (M=1 decode): reads IQ1_M weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq1_m_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq1_m gemv rows")?;
        ensure_len(cols, "iq1_m gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq1_m_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq1_m_gemv")
    }

    /// Fused IQ3_XXS GEMV (M=1 decode): reads IQ3_XXS weights directly, f32 activation.
    /// Requires cols % 256 == 0.
    pub fn launch_iq3_xxs_gemv(
        weights: &DeviceBuffer,
        x: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "iq3_xxs gemv rows")?;
        ensure_len(cols, "iq3_xxs gemv cols")?;
        launch_status(unsafe {
            hi_cuda_launch_iq3_xxs_gemv(
                weights.as_ptr(),
                x.as_ptr(),
                output.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_iq3_xxs_gemv")
    }

    pub fn launch_dequantize_matrix(
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        elements: usize,
        quant_type: i32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(elements, "dequantize elements")?;
        launch_status(unsafe {
            hi_cuda_launch_dequantize_matrix(
                input.as_ptr(),
                output.as_mut_ptr(),
                elements as c_int,
                quant_type as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_dequantize_matrix")
    }

    pub fn launch_rope(
        values: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rot_dim: usize,
        base: f32,
        scale: f32,
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "rope seq_len")?;
        ensure_len(heads, "rope heads")?;
        ensure_len(head_dim, "rope head_dim")?;
        ensure_len(rot_dim, "rope rot_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_rope(
                values.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                rot_dim as c_int,
                base,
                scale,
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rope")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_with_offset(
        values: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rot_dim: usize,
        base: f32,
        scale: f32,
        position_offset: usize,
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "rope seq_len")?;
        ensure_len(heads, "rope heads")?;
        ensure_len(head_dim, "rope head_dim")?;
        ensure_len(rot_dim, "rope rot_dim")?;
        ensure_len(position_offset, "rope position_offset")?;
        launch_status(unsafe {
            hi_cuda_launch_rope_with_offset(
                values.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                rot_dim as c_int,
                base,
                scale,
                position_offset as c_int,
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rope_with_offset")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_batched_with_offset(
        values: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rot_dim: usize,
        base: f32,
        scale: f32,
        position_offset: usize,
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "rope batch_count")?;
        ensure_len(seq_len, "rope seq_len")?;
        ensure_len(heads, "rope heads")?;
        ensure_len(head_dim, "rope head_dim")?;
        ensure_len(rot_dim, "rope rot_dim")?;
        ensure_len(position_offset, "rope position_offset")?;
        launch_status(unsafe {
            hi_cuda_launch_rope_batched_with_offset(
                values.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                rot_dim as c_int,
                base,
                scale,
                position_offset as c_int,
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rope_batched_with_offset")
    }

    /// CUDA-graph decode variant: the RoPE position offset is read from a device buffer
    /// (`d_position_offset`, one i32) so a captured graph replays for every token.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_batched_with_offset_devpos(
        values: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rot_dim: usize,
        base: f32,
        scale: f32,
        d_position_offset: &DeviceBuffer,
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "rope batch_count")?;
        ensure_len(seq_len, "rope seq_len")?;
        ensure_len(heads, "rope heads")?;
        ensure_len(head_dim, "rope head_dim")?;
        ensure_len(rot_dim, "rope rot_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_rope_batched_with_offset_devpos(
                values.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                rot_dim as c_int,
                base,
                scale,
                d_position_offset.as_ptr(),
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rope_batched_with_offset_devpos")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_batched_positions(
        values: &DeviceBuffer,
        positions: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        rot_dim: usize,
        base: f32,
        scale: f32,
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "rope positions batch_count")?;
        ensure_len(seq_len, "rope positions seq_len")?;
        ensure_len(heads, "rope positions heads")?;
        ensure_len(head_dim, "rope positions head_dim")?;
        ensure_len(rot_dim, "rope positions rot_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_rope_batched_positions(
                values.as_mut_ptr(),
                positions.as_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                rot_dim as c_int,
                base,
                scale,
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_rope_batched_positions")
    }

    pub fn launch_vision_rope(
        values: &DeviceBuffer,
        pos_h: &DeviceBuffer,
        pos_w: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        base: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "vision_rope seq_len")?;
        ensure_len(heads, "vision_rope heads")?;
        ensure_len(head_dim, "vision_rope head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_vision_rope(
                values.as_mut_ptr(),
                pos_h.as_ptr(),
                pos_w.as_ptr(),
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                base,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_vision_rope")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_mrope(
        values: &DeviceBuffer,
        pos_t: &DeviceBuffer,
        pos_h: &DeviceBuffer,
        pos_w: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        head_dim: usize,
        base: f32,
        scale: f32,
        sections: [usize; 4],
        split_half: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "mrope seq_len")?;
        ensure_len(heads, "mrope heads")?;
        ensure_len(head_dim, "mrope head_dim")?;
        for section in sections {
            ensure_len(section, "mrope section")?;
        }
        launch_status(unsafe {
            hi_cuda_launch_mrope(
                values.as_mut_ptr(),
                pos_t.as_ptr(),
                pos_h.as_ptr(),
                pos_w.as_ptr(),
                seq_len as c_int,
                heads as c_int,
                head_dim as c_int,
                base,
                scale,
                sections[0] as c_int,
                sections[1] as c_int,
                sections[2] as c_int,
                sections[3] as c_int,
                if split_half { 1 } else { 0 },
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_mrope")
    }

    pub fn launch_write_kv_cache(
        values: &DeviceBuffer,
        cache: &DeviceBuffer,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        start_pos: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "kv_cache row_count")?;
        ensure_len(kv_heads, "kv_cache kv_heads")?;
        ensure_len(head_dim, "kv_cache head_dim")?;
        ensure_len(max_seq, "kv_cache max_seq")?;
        ensure_len(start_pos, "kv_cache start_pos")?;
        launch_status(unsafe {
            hi_cuda_launch_write_kv_cache(
                values.as_ptr(),
                cache.as_mut_ptr(),
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                max_seq as c_int,
                start_pos as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_kv_cache")
    }

    pub fn launch_write_kv_cache_batched(
        values: &DeviceBuffer,
        cache: &DeviceBuffer,
        batch_count: usize,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        start_pos: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "kv_cache batch_count")?;
        ensure_len(row_count, "kv_cache row_count")?;
        ensure_len(kv_heads, "kv_cache kv_heads")?;
        ensure_len(head_dim, "kv_cache head_dim")?;
        ensure_len(max_seq, "kv_cache max_seq")?;
        ensure_len(start_pos, "kv_cache start_pos")?;
        launch_status(unsafe {
            hi_cuda_launch_write_kv_cache_batched(
                values.as_ptr(),
                cache.as_mut_ptr(),
                batch_count as c_int,
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                max_seq as c_int,
                start_pos as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_kv_cache_batched")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_paged_kv_cache(
        values: &DeviceBuffer,
        pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        start_pos: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(row_count, "paged_kv_cache row_count")?;
        ensure_len(kv_heads, "paged_kv_cache kv_heads")?;
        ensure_len(head_dim, "paged_kv_cache head_dim")?;
        ensure_len(page_size, "paged_kv_cache page_size")?;
        ensure_len(page_table_len, "paged_kv_cache page_table_len")?;
        ensure_len(start_pos, "paged_kv_cache start_pos")?;
        launch_status(unsafe {
            hi_cuda_launch_write_paged_kv_cache(
                values.as_ptr(),
                pages.as_mut_ptr(),
                page_table.as_ptr(),
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                start_pos as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_paged_kv_cache")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_paged_kv_cache_batched(
        values: &DeviceBuffer,
        pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        batch_count: usize,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        start_pos: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "paged_kv_cache batch_count")?;
        ensure_len(row_count, "paged_kv_cache row_count")?;
        ensure_len(kv_heads, "paged_kv_cache kv_heads")?;
        ensure_len(head_dim, "paged_kv_cache head_dim")?;
        ensure_len(page_size, "paged_kv_cache page_size")?;
        ensure_len(page_table_len, "paged_kv_cache page_table_len")?;
        ensure_len(start_pos, "paged_kv_cache start_pos")?;
        launch_status(unsafe {
            hi_cuda_launch_write_paged_kv_cache_batched(
                values.as_ptr(),
                pages.as_mut_ptr(),
                page_table.as_ptr(),
                batch_count as c_int,
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                start_pos as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_paged_kv_cache_batched")
    }

    /// CUDA-graph decode variant: `start_pos` is read from a device buffer (one i32).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_paged_kv_cache_batched_devpos(
        values: &DeviceBuffer,
        pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        batch_count: usize,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        d_start_pos: &DeviceBuffer,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "paged_kv_cache batch_count")?;
        ensure_len(row_count, "paged_kv_cache row_count")?;
        ensure_len(kv_heads, "paged_kv_cache kv_heads")?;
        ensure_len(head_dim, "paged_kv_cache head_dim")?;
        ensure_len(page_size, "paged_kv_cache page_size")?;
        ensure_len(page_table_len, "paged_kv_cache page_table_len")?;
        launch_status(unsafe {
            hi_cuda_launch_write_paged_kv_cache_batched_devpos(
                values.as_ptr(),
                pages.as_mut_ptr(),
                page_table.as_ptr(),
                batch_count as c_int,
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                d_start_pos.as_ptr(),
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_paged_kv_cache_batched_devpos")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_paged_kv_cache_batched_positions(
        values: &DeviceBuffer,
        pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        positions: &DeviceBuffer,
        batch_count: usize,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "paged_kv_cache positions batch_count")?;
        ensure_len(row_count, "paged_kv_cache positions row_count")?;
        ensure_len(kv_heads, "paged_kv_cache positions kv_heads")?;
        ensure_len(head_dim, "paged_kv_cache positions head_dim")?;
        ensure_len(page_size, "paged_kv_cache positions page_size")?;
        ensure_len(page_table_len, "paged_kv_cache positions page_table_len")?;
        launch_status(unsafe {
            hi_cuda_launch_write_paged_kv_cache_batched_positions(
                values.as_ptr(),
                pages.as_mut_ptr(),
                page_table.as_ptr(),
                positions.as_ptr(),
                batch_count as c_int,
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_paged_kv_cache_batched_positions")
    }

    /// int8/Q8 paged KV write. Quantizes each `(batch,row,kv_head)` head_dim vector to
    /// int8 + one f32 scale. `positions` (per-batch base position) is used when `Some`
    /// (decode); otherwise `start_pos` is the base for all rows (prefill).
    #[allow(clippy::too_many_arguments)]
    /// Position precedence matches the kernel: per-batch `positions`, else the
    /// scalar `device_start_pos` counter (CUDA-graph capture), else `start_pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_paged_kv_cache_q8_batched(
        values: &DeviceBuffer,
        pages: &DeviceBuffer,
        scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        positions: Option<&DeviceBuffer>,
        device_start_pos: Option<&DeviceBuffer>,
        start_pos: usize,
        batch_count: usize,
        row_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "kv_q8 batch_count")?;
        ensure_len(row_count, "kv_q8 row_count")?;
        ensure_len(kv_heads, "kv_q8 kv_heads")?;
        ensure_len(head_dim, "kv_q8 head_dim")?;
        ensure_len(page_size, "kv_q8 page_size")?;
        ensure_len(page_table_len, "kv_q8 page_table_len")?;
        let positions_ptr = positions
            .map(|buffer| buffer.as_ptr())
            .unwrap_or(std::ptr::null());
        let device_start_pos_ptr = device_start_pos
            .map(|buffer| buffer.as_ptr())
            .unwrap_or(std::ptr::null());
        launch_status(unsafe {
            hi_cuda_launch_write_paged_kv_cache_q8_batched(
                values.as_ptr(),
                pages.as_mut_ptr(),
                scales.as_mut_ptr(),
                page_table.as_ptr(),
                positions_ptr,
                device_start_pos_ptr,
                start_pos as c_int,
                batch_count as c_int,
                row_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_write_paged_kv_cache_q8_batched")
    }

    pub fn launch_copy_paged_kv_cache_prefix_batched(
        pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        batch_count: usize,
        token_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "copy_paged_kv_prefix batch_count")?;
        ensure_len(token_count, "copy_paged_kv_prefix token_count")?;
        ensure_len(kv_heads, "copy_paged_kv_prefix kv_heads")?;
        ensure_len(head_dim, "copy_paged_kv_prefix head_dim")?;
        ensure_len(page_size, "copy_paged_kv_prefix page_size")?;
        ensure_len(page_table_len, "copy_paged_kv_prefix page_table_len")?;
        launch_status(unsafe {
            hi_cuda_launch_copy_paged_kv_cache_prefix_batched(
                pages.as_mut_ptr(),
                page_table.as_ptr(),
                batch_count as c_int,
                token_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_copy_paged_kv_cache_prefix_batched")
    }

    /// int8/Q8 prefix copy: copies int8 page data + the parallel per-vector scales from
    /// batch row 0 to the other batch rows. Call once for K (pages+scales) and once for V.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_copy_paged_kv_cache_prefix_batched_q8(
        pages: &DeviceBuffer,
        scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        batch_count: usize,
        token_count: usize,
        kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        page_table_len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "copy_paged_kv_prefix_q8 batch_count")?;
        ensure_len(token_count, "copy_paged_kv_prefix_q8 token_count")?;
        ensure_len(kv_heads, "copy_paged_kv_prefix_q8 kv_heads")?;
        ensure_len(head_dim, "copy_paged_kv_prefix_q8 head_dim")?;
        ensure_len(page_size, "copy_paged_kv_prefix_q8 page_size")?;
        ensure_len(page_table_len, "copy_paged_kv_prefix_q8 page_table_len")?;
        launch_status(unsafe {
            hi_cuda_launch_copy_paged_kv_cache_prefix_batched_q8(
                pages.as_mut_ptr(),
                scales.as_mut_ptr(),
                page_table.as_ptr(),
                batch_count as c_int,
                token_count as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                page_size as c_int,
                page_table_len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_copy_paged_kv_cache_prefix_batched_q8")
    }

    pub fn launch_causal_attention(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "attention seq_len")?;
        ensure_len(heads, "attention heads")?;
        ensure_len(kv_heads, "attention kv_heads")?;
        ensure_len(head_dim, "attention head_dim")?;
        ensure_len(v_head_dim, "attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_causal_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_causal_attention")
    }

    pub fn launch_flash_causal_attention(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "flash_attention seq_len")?;
        ensure_len(heads, "flash_attention heads")?;
        ensure_len(kv_heads, "flash_attention kv_heads")?;
        ensure_len(head_dim, "flash_attention head_dim")?;
        ensure_len(v_head_dim, "flash_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_causal_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_causal_attention")
    }

    pub fn launch_tiled_causal_attention(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "tiled_attention seq_len")?;
        ensure_len(heads, "tiled_attention heads")?;
        ensure_len(kv_heads, "tiled_attention kv_heads")?;
        ensure_len(head_dim, "tiled_attention head_dim")?;
        ensure_len(v_head_dim, "tiled_attention v_head_dim")?;
        ensure_len(window, "tiled_attention window")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_causal_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_causal_attention")
    }

    pub fn launch_causal_attention_batched(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "attention batch_count")?;
        ensure_len(seq_len, "attention seq_len")?;
        ensure_len(heads, "attention heads")?;
        ensure_len(kv_heads, "attention kv_heads")?;
        ensure_len(head_dim, "attention head_dim")?;
        ensure_len(v_head_dim, "attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_causal_attention_batched(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_causal_attention_batched")
    }

    pub fn launch_flash_causal_attention_batched(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "flash_attention batch_count")?;
        ensure_len(seq_len, "flash_attention seq_len")?;
        ensure_len(heads, "flash_attention heads")?;
        ensure_len(kv_heads, "flash_attention kv_heads")?;
        ensure_len(head_dim, "flash_attention head_dim")?;
        ensure_len(v_head_dim, "flash_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_causal_attention_batched(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_causal_attention_batched")
    }

    pub fn launch_tiled_causal_attention_batched(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "tiled_attention batch_count")?;
        ensure_len(seq_len, "tiled_attention seq_len")?;
        ensure_len(heads, "tiled_attention heads")?;
        ensure_len(kv_heads, "tiled_attention kv_heads")?;
        ensure_len(head_dim, "tiled_attention head_dim")?;
        ensure_len(v_head_dim, "tiled_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_causal_attention_batched(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_causal_attention_batched")
    }

    /// Flash-attention (shared-memory K/V tiling) causal batched prefill. Causal only
    /// (no sliding window); head_dim capped at HI_CUDA_FLASH_TILE_MAX_HEAD_DIM.
    pub fn launch_flashtile_causal_attention_batched(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "flash_attention batch_count")?;
        ensure_len(seq_len, "flash_attention seq_len")?;
        ensure_len(heads, "flash_attention heads")?;
        ensure_len(kv_heads, "flash_attention kv_heads")?;
        ensure_len(head_dim, "flash_attention head_dim")?;
        ensure_len(v_head_dim, "flash_attention v_head_dim")?;
        ensure_len(window, "flash_attention window")?;
        launch_status(unsafe {
            hi_cuda_launch_flashtile_causal_attention_batched(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flashtile_causal_attention_batched")
    }

    /// Chunked-prefill causal attention: a `chunk_len`-token query chunk attends to the f16
    /// paged KV cache (which already holds `[0, query_offset+chunk_len)`), query row r at
    /// absolute position `query_offset+r`. Shares the KV load across the query tile.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_paged_prefill_causal_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "paged_prefill query_offset")?;
        ensure_len(batch_count, "paged_prefill batch_count")?;
        ensure_len(chunk_len, "paged_prefill chunk_len")?;
        ensure_len(page_size, "paged_prefill page_size")?;
        ensure_len(page_table_len, "paged_prefill page_table_len")?;
        ensure_len(heads, "paged_prefill heads")?;
        ensure_len(kv_heads, "paged_prefill kv_heads")?;
        ensure_len(head_dim, "paged_prefill head_dim")?;
        ensure_len(v_head_dim, "paged_prefill v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_paged_prefill_causal_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_paged_prefill_causal_attention_batched")
    }

    /// int8/Q8 chunked-prefill causal attention (dequantizes int8 paged K/V via the scale
    /// buffers). Mirrors `launch_paged_prefill_causal_attention_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_paged_prefill_causal_attention_batched_q8(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: &DeviceBuffer,
        v_scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "paged_prefill_q8 query_offset")?;
        ensure_len(batch_count, "paged_prefill_q8 batch_count")?;
        ensure_len(chunk_len, "paged_prefill_q8 chunk_len")?;
        ensure_len(page_size, "paged_prefill_q8 page_size")?;
        ensure_len(page_table_len, "paged_prefill_q8 page_table_len")?;
        ensure_len(heads, "paged_prefill_q8 heads")?;
        ensure_len(kv_heads, "paged_prefill_q8 kv_heads")?;
        ensure_len(head_dim, "paged_prefill_q8 head_dim")?;
        ensure_len(v_head_dim, "paged_prefill_q8 v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_paged_prefill_causal_attention_batched_q8(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.as_ptr(),
                v_scales.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_paged_prefill_causal_attention_batched_q8")
    }

    /// Tensor-core (WMMA) paged chunked-prefill causal attention (f16 cache). Same signature as
    /// `launch_paged_prefill_causal_attention_batched` minus v_head_dim (requires qk==v head dim).
    /// q/output f32; head_dim multiple of 16, <=128.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_wmma_paged_prefill_causal_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "wmma_paged_prefill query_offset")?;
        ensure_len(batch_count, "wmma_paged_prefill batch_count")?;
        ensure_len(chunk_len, "wmma_paged_prefill chunk_len")?;
        ensure_len(page_size, "wmma_paged_prefill page_size")?;
        ensure_len(page_table_len, "wmma_paged_prefill page_table_len")?;
        ensure_len(heads, "wmma_paged_prefill heads")?;
        ensure_len(kv_heads, "wmma_paged_prefill kv_heads")?;
        ensure_len(head_dim, "wmma_paged_prefill head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_wmma_paged_prefill_causal_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_wmma_paged_prefill_causal_attention_batched")
    }

    /// int8/Q8 variant of `launch_wmma_paged_prefill_causal_attention_batched` (dequantizes the
    /// int8 paged K/V via the scale buffers during the shared gather).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_wmma_paged_prefill_causal_attention_batched_q8(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: &DeviceBuffer,
        v_scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "wmma_paged_prefill_q8 query_offset")?;
        ensure_len(batch_count, "wmma_paged_prefill_q8 batch_count")?;
        ensure_len(chunk_len, "wmma_paged_prefill_q8 chunk_len")?;
        ensure_len(page_size, "wmma_paged_prefill_q8 page_size")?;
        ensure_len(page_table_len, "wmma_paged_prefill_q8 page_table_len")?;
        ensure_len(heads, "wmma_paged_prefill_q8 heads")?;
        ensure_len(kv_heads, "wmma_paged_prefill_q8 kv_heads")?;
        ensure_len(head_dim, "wmma_paged_prefill_q8 head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_wmma_paged_prefill_causal_attention_batched_q8(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.as_ptr(),
                v_scales.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_wmma_paged_prefill_causal_attention_batched_q8")
    }

    /// FA2 multi-warp paged prefill attention (4 warps per 64-query block tile,
    /// shared 32-key K/V staging): same interface and shape constraints as
    /// `launch_wmma_paged_prefill_causal_attention_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fa2_paged_prefill_causal_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "fa2_paged_prefill query_offset")?;
        ensure_len(batch_count, "fa2_paged_prefill batch_count")?;
        ensure_len(chunk_len, "fa2_paged_prefill chunk_len")?;
        ensure_len(page_size, "fa2_paged_prefill page_size")?;
        ensure_len(page_table_len, "fa2_paged_prefill page_table_len")?;
        ensure_len(heads, "fa2_paged_prefill heads")?;
        ensure_len(kv_heads, "fa2_paged_prefill kv_heads")?;
        ensure_len(head_dim, "fa2_paged_prefill head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_fa2_paged_prefill_causal_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_fa2_paged_prefill_causal_attention_batched")
    }

    /// int8/Q8 variant of `launch_fa2_paged_prefill_causal_attention_batched`
    /// (dequantizes via the scale buffers during the shared gather).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fa2_paged_prefill_causal_attention_batched_q8(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: &DeviceBuffer,
        v_scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        query_offset: usize,
        batch_count: usize,
        chunk_len: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(query_offset, "fa2_paged_prefill_q8 query_offset")?;
        ensure_len(batch_count, "fa2_paged_prefill_q8 batch_count")?;
        ensure_len(chunk_len, "fa2_paged_prefill_q8 chunk_len")?;
        ensure_len(page_size, "fa2_paged_prefill_q8 page_size")?;
        ensure_len(page_table_len, "fa2_paged_prefill_q8 page_table_len")?;
        ensure_len(heads, "fa2_paged_prefill_q8 heads")?;
        ensure_len(kv_heads, "fa2_paged_prefill_q8 kv_heads")?;
        ensure_len(head_dim, "fa2_paged_prefill_q8 head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_fa2_paged_prefill_causal_attention_batched_q8(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.as_ptr(),
                v_scales.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                query_offset as c_int,
                batch_count as c_int,
                chunk_len as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_fa2_paged_prefill_causal_attention_batched_q8")
    }

    /// Tensor-core (WMMA) flash-attention, causal batched. q/k/v are f16; output f32.
    /// head_dim multiple of 16, <=128, v_head_dim==head_dim, no window.
    pub fn launch_wmma_causal_attention_batched(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "wmma_attention batch_count")?;
        ensure_len(seq_len, "wmma_attention seq_len")?;
        ensure_len(heads, "wmma_attention heads")?;
        ensure_len(kv_heads, "wmma_attention kv_heads")?;
        ensure_len(head_dim, "wmma_attention head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_wmma_causal_attention_batched(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_wmma_causal_attention_batched")
    }

    pub fn launch_full_attention(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        output: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "full_attention seq_len")?;
        ensure_len(heads, "full_attention heads")?;
        ensure_len(kv_heads, "full_attention kv_heads")?;
        ensure_len(head_dim, "full_attention head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_full_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_full_attention")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_window_attention(
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        window_start: &DeviceBuffer,
        window_end: &DeviceBuffer,
        output: &DeviceBuffer,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(seq_len, "window_attention seq_len")?;
        ensure_len(heads, "window_attention heads")?;
        ensure_len(kv_heads, "window_attention kv_heads")?;
        ensure_len(head_dim, "window_attention head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_window_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                window_start.as_ptr(),
                window_end.as_ptr(),
                output.as_mut_ptr(),
                seq_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_window_attention")
    }

    pub fn launch_cached_decode_attention(
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        max_seq: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "cached_attention position")?;
        ensure_len(max_seq, "cached_attention max_seq")?;
        ensure_len(heads, "cached_attention heads")?;
        ensure_len(kv_heads, "cached_attention kv_heads")?;
        ensure_len(head_dim, "cached_attention head_dim")?;
        ensure_len(v_head_dim, "cached_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_cached_decode_attention(
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                max_seq as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cached_decode_attention")
    }

    pub fn launch_flash_cached_decode_attention(
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        max_seq: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "flash_cached_attention position")?;
        ensure_len(max_seq, "flash_cached_attention max_seq")?;
        ensure_len(heads, "flash_cached_attention heads")?;
        ensure_len(kv_heads, "flash_cached_attention kv_heads")?;
        ensure_len(head_dim, "flash_cached_attention head_dim")?;
        ensure_len(v_head_dim, "flash_cached_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_cached_decode_attention(
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                max_seq as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_cached_decode_attention")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_paged_decode_attention(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "paged_attention position")?;
        ensure_len(page_size, "paged_attention page_size")?;
        ensure_len(page_table_len, "paged_attention page_table_len")?;
        ensure_len(heads, "paged_attention heads")?;
        ensure_len(kv_heads, "paged_attention kv_heads")?;
        ensure_len(head_dim, "paged_attention head_dim")?;
        ensure_len(v_head_dim, "paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_paged_decode_attention(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_paged_decode_attention")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_flash_paged_decode_attention(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "flash_paged_attention position")?;
        ensure_len(page_size, "flash_paged_attention page_size")?;
        ensure_len(page_table_len, "flash_paged_attention page_table_len")?;
        ensure_len(heads, "flash_paged_attention heads")?;
        ensure_len(kv_heads, "flash_paged_attention kv_heads")?;
        ensure_len(head_dim, "flash_paged_attention head_dim")?;
        ensure_len(v_head_dim, "flash_paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_paged_decode_attention(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_paged_decode_attention")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "tiled_paged_attention position")?;
        ensure_len(page_size, "tiled_paged_attention page_size")?;
        ensure_len(page_table_len, "tiled_paged_attention page_table_len")?;
        ensure_len(heads, "tiled_paged_attention heads")?;
        ensure_len(kv_heads, "tiled_paged_attention kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention v_head_dim")?;
        ensure_len(window, "tiled_paged_attention window")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention")
    }

    /// int8/Q8 tiled paged decode attention: K/V pages are int8, dequantized per-vector
    /// via the parallel `k_scales`/`v_scales` buffers (one f32 scale per cache vector).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention_q8(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: &DeviceBuffer,
        v_scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(position, "tiled_paged_attention_q8 position")?;
        ensure_len(page_size, "tiled_paged_attention_q8 page_size")?;
        ensure_len(page_table_len, "tiled_paged_attention_q8 page_table_len")?;
        ensure_len(heads, "tiled_paged_attention_q8 heads")?;
        ensure_len(kv_heads, "tiled_paged_attention_q8 kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention_q8 head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention_q8 v_head_dim")?;
        ensure_len(window, "tiled_paged_attention_q8 window")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention_q8(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.as_ptr(),
                v_scales.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention_q8")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_paged_decode_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "paged_attention batch_count")?;
        ensure_len(position, "paged_attention position")?;
        ensure_len(page_size, "paged_attention page_size")?;
        ensure_len(page_table_len, "paged_attention page_table_len")?;
        ensure_len(heads, "paged_attention heads")?;
        ensure_len(kv_heads, "paged_attention kv_heads")?;
        ensure_len(head_dim, "paged_attention head_dim")?;
        ensure_len(v_head_dim, "paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_paged_decode_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_paged_decode_attention_batched")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_flash_paged_decode_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "flash_paged_attention batch_count")?;
        ensure_len(position, "flash_paged_attention position")?;
        ensure_len(page_size, "flash_paged_attention page_size")?;
        ensure_len(page_table_len, "flash_paged_attention page_table_len")?;
        ensure_len(heads, "flash_paged_attention heads")?;
        ensure_len(kv_heads, "flash_paged_attention kv_heads")?;
        ensure_len(head_dim, "flash_paged_attention head_dim")?;
        ensure_len(v_head_dim, "flash_paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_paged_decode_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_paged_decode_attention_batched")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention_batched(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        position: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "tiled_paged_attention batch_count")?;
        ensure_len(position, "tiled_paged_attention position")?;
        ensure_len(page_size, "tiled_paged_attention page_size")?;
        ensure_len(page_table_len, "tiled_paged_attention page_table_len")?;
        ensure_len(heads, "tiled_paged_attention heads")?;
        ensure_len(kv_heads, "tiled_paged_attention kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention_batched(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                position as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention_batched")
    }

    /// CUDA-graph decode variant: `position` (last attended KV index) is read from a device
    /// buffer (one i32). The device position must stay within `page_size * page_table_len`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention_batched_devpos(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        d_position: &DeviceBuffer,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "tiled_paged_attention batch_count")?;
        ensure_len(page_size, "tiled_paged_attention page_size")?;
        ensure_len(page_table_len, "tiled_paged_attention page_table_len")?;
        ensure_len(heads, "tiled_paged_attention heads")?;
        ensure_len(kv_heads, "tiled_paged_attention kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention_batched_devpos(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                d_position.as_ptr(),
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention_batched_devpos")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_paged_decode_attention_batched_positions(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        positions: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "paged_attention positions batch_count")?;
        ensure_len(page_size, "paged_attention positions page_size")?;
        ensure_len(page_table_len, "paged_attention positions page_table_len")?;
        ensure_len(heads, "paged_attention positions heads")?;
        ensure_len(kv_heads, "paged_attention positions kv_heads")?;
        ensure_len(head_dim, "paged_attention positions head_dim")?;
        ensure_len(v_head_dim, "paged_attention positions v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_paged_decode_attention_batched_positions(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                positions.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_paged_decode_attention_batched_positions")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention_batched_positions(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        page_table: &DeviceBuffer,
        positions: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "tiled_paged_attention positions batch_count")?;
        ensure_len(page_size, "tiled_paged_attention positions page_size")?;
        ensure_len(
            page_table_len,
            "tiled_paged_attention positions page_table_len",
        )?;
        ensure_len(heads, "tiled_paged_attention positions heads")?;
        ensure_len(kv_heads, "tiled_paged_attention positions kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention positions head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention positions v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention_batched_positions(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                page_table.as_ptr(),
                positions.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention_batched_positions")
    }

    /// int8/Q8 batched-positions tiled paged decode attention (dequantizes int8 K/V pages
    /// via the parallel scale buffers). Mirrors the f16 batched-positions variant.
    #[allow(clippy::too_many_arguments)]
    /// Position precedence matches the kernel: per-batch `positions`, else the
    /// scalar `device_start_pos` counter (CUDA-graph capture), else `start_pos`
    /// (plain eager — no per-call positions upload).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_tiled_paged_decode_attention_batched_positions_q8(
        q: &DeviceBuffer,
        k_pages: &DeviceBuffer,
        v_pages: &DeviceBuffer,
        k_scales: &DeviceBuffer,
        v_scales: &DeviceBuffer,
        page_table: &DeviceBuffer,
        positions: Option<&DeviceBuffer>,
        device_start_pos: Option<&DeviceBuffer>,
        start_pos: usize,
        output: &DeviceBuffer,
        batch_count: usize,
        page_size: usize,
        page_table_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        window: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(
            batch_count,
            "tiled_paged_attention_q8 positions batch_count",
        )?;
        ensure_len(page_size, "tiled_paged_attention_q8 positions page_size")?;
        ensure_len(
            page_table_len,
            "tiled_paged_attention_q8 positions page_table_len",
        )?;
        ensure_len(heads, "tiled_paged_attention_q8 positions heads")?;
        ensure_len(kv_heads, "tiled_paged_attention_q8 positions kv_heads")?;
        ensure_len(head_dim, "tiled_paged_attention_q8 positions head_dim")?;
        ensure_len(v_head_dim, "tiled_paged_attention_q8 positions v_head_dim")?;
        ensure_len(start_pos, "tiled_paged_attention_q8 start_pos")?;
        let positions_ptr = positions
            .map(|buffer| buffer.as_ptr())
            .unwrap_or(std::ptr::null());
        let device_start_pos_ptr = device_start_pos
            .map(|buffer| buffer.as_ptr())
            .unwrap_or(std::ptr::null());
        launch_status(unsafe {
            hi_cuda_launch_tiled_paged_decode_attention_batched_positions_q8(
                q.as_ptr(),
                k_pages.as_ptr(),
                v_pages.as_ptr(),
                k_scales.as_ptr(),
                v_scales.as_ptr(),
                page_table.as_ptr(),
                positions_ptr,
                device_start_pos_ptr,
                start_pos as c_int,
                output.as_mut_ptr(),
                batch_count as c_int,
                page_size as c_int,
                page_table_len as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                window as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_tiled_paged_decode_attention_batched_positions_q8")
    }

    pub fn launch_cached_decode_attention_batched(
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        position: usize,
        max_seq: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "cached_attention batch_count")?;
        ensure_len(position, "cached_attention position")?;
        ensure_len(max_seq, "cached_attention max_seq")?;
        ensure_len(heads, "cached_attention heads")?;
        ensure_len(kv_heads, "cached_attention kv_heads")?;
        ensure_len(head_dim, "cached_attention head_dim")?;
        ensure_len(v_head_dim, "cached_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_cached_decode_attention_batched(
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                position as c_int,
                max_seq as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_cached_decode_attention_batched")
    }

    pub fn launch_flash_cached_decode_attention_batched(
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        output: &DeviceBuffer,
        batch_count: usize,
        position: usize,
        max_seq: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "flash_cached_attention batch_count")?;
        ensure_len(position, "flash_cached_attention position")?;
        ensure_len(max_seq, "flash_cached_attention max_seq")?;
        ensure_len(heads, "flash_cached_attention heads")?;
        ensure_len(kv_heads, "flash_cached_attention kv_heads")?;
        ensure_len(head_dim, "flash_cached_attention head_dim")?;
        ensure_len(v_head_dim, "flash_cached_attention v_head_dim")?;
        launch_status(unsafe {
            hi_cuda_launch_flash_cached_decode_attention_batched(
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                output.as_mut_ptr(),
                batch_count as c_int,
                position as c_int,
                max_seq as c_int,
                heads as c_int,
                kv_heads as c_int,
                head_dim as c_int,
                v_head_dim as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_flash_cached_decode_attention_batched")
    }

    pub fn launch_argmax(
        logits: &DeviceBuffer,
        output_token: &DeviceBuffer,
        len: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(len, "argmax len")?;
        launch_status(unsafe {
            hi_cuda_launch_argmax(
                logits.as_ptr(),
                output_token.as_mut_ptr(),
                len as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_argmax")
    }

    pub fn launch_argmax_last_row(
        logits: &DeviceBuffer,
        output_token: &DeviceBuffer,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "argmax_last_row rows")?;
        ensure_len(cols, "argmax_last_row cols")?;
        launch_status(unsafe {
            hi_cuda_launch_argmax_last_row(
                logits.as_ptr(),
                output_token.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_argmax_last_row")
    }

    pub fn launch_argmax_batched_last_token(
        logits: &DeviceBuffer,
        output_tokens: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "argmax_batched batch_count")?;
        ensure_len(seq_len, "argmax_batched seq_len")?;
        ensure_len(cols, "argmax_batched cols")?;
        launch_status(unsafe {
            hi_cuda_launch_argmax_batched_last_token(
                logits.as_ptr(),
                output_tokens.as_mut_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                cols as c_int,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_argmax_batched_last_token")
    }

    pub fn launch_sample_last_row(
        logits: &DeviceBuffer,
        output_token: &DeviceBuffer,
        rows: usize,
        cols: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        sample: f32,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(rows, "sample_last_row rows")?;
        ensure_len(cols, "sample_last_row cols")?;
        let top_k = match top_k {
            Some(value) => {
                if value > c_int::MAX as u32 {
                    bail!("sample_last_row top_k {value} exceeds CUDA launch i32 limit");
                }
                value as c_int
            }
            None => 0,
        };
        launch_status(unsafe {
            hi_cuda_launch_sample_last_row(
                logits.as_ptr(),
                output_token.as_mut_ptr(),
                rows as c_int,
                cols as c_int,
                temperature,
                top_p,
                top_k,
                sample,
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_sample_last_row")
    }

    /// Token id emitted by the ranked GPU sampler when a row cannot be sampled
    /// on-device (top-p nucleus larger than the survivor buffer, or a top_k
    /// beyond it): the caller re-samples that row on the host.
    pub const RANKED_SAMPLER_OVERFLOW: u32 = u32::MAX;

    /// Capacity of the ranked GPU sampler's shared-memory survivor buffer
    /// (must match `HI_CUDA_RANKED_SURVIVORS` in kernels.cu): top_k configs up
    /// to this bound sample fully on-device and can be graph-captured.
    pub const RANKED_SAMPLER_SURVIVORS: usize = 1024;

    pub fn launch_sample_batched_last_token(
        logits: &DeviceBuffer,
        output_tokens: &DeviceBuffer,
        samples: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        cols: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
        gpu_ranked: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "sample_batched batch_count")?;
        ensure_len(seq_len, "sample_batched seq_len")?;
        ensure_len(cols, "sample_batched cols")?;
        let top_k = match top_k {
            Some(value) => {
                if value > c_int::MAX as u32 {
                    bail!("sample_batched top_k {value} exceeds CUDA launch i32 limit");
                }
                value as c_int
            }
            None => 0,
        };
        launch_status(unsafe {
            hi_cuda_launch_sample_batched_last_token(
                logits.as_ptr(),
                output_tokens.as_mut_ptr(),
                samples.as_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                cols as c_int,
                temperature,
                top_p,
                top_k,
                c_int::from(gpu_ranked),
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_sample_batched_last_token")
    }

    /// Per-row sampling configs for a heterogeneous decode batch: `temperatures`,
    /// `top_ps`, `top_ks` (i32; 0 = unset), and `samples` are `[batch_count]` device
    /// arrays. Greedy rows (temperature <= 0) bit-match the batched argmax kernel;
    /// ranked rows either sample on-device via the radix sampler (`gpu_ranked`,
    /// overflow rows come back as [`RANKED_SAMPLER_OVERFLOW`]) or write the argmax
    /// as a placeholder for the host ranked sampler.
    pub fn launch_select_batched_last_token_per_row(
        logits: &DeviceBuffer,
        output_tokens: &DeviceBuffer,
        samples: &DeviceBuffer,
        temperatures: &DeviceBuffer,
        top_ps: &DeviceBuffer,
        top_ks: &DeviceBuffer,
        batch_count: usize,
        seq_len: usize,
        cols: usize,
        gpu_ranked: bool,
        stream: &Stream,
    ) -> Result<()> {
        ensure_len(batch_count, "select_batched_per_row batch_count")?;
        ensure_len(seq_len, "select_batched_per_row seq_len")?;
        ensure_len(cols, "select_batched_per_row cols")?;
        launch_status(unsafe {
            hi_cuda_launch_select_batched_last_token_per_row(
                logits.as_ptr(),
                output_tokens.as_mut_ptr(),
                samples.as_ptr(),
                temperatures.as_ptr(),
                top_ps.as_ptr(),
                top_ks.as_ptr(),
                batch_count as c_int,
                seq_len as c_int,
                cols as c_int,
                c_int::from(gpu_ranked),
                stream.as_raw(),
            )
        })?;
        check_last_error("hi_cuda_launch_select_batched_last_token_per_row")
    }

    fn ensure_len(value: usize, label: &str) -> Result<()> {
        if value > c_int::MAX as usize {
            bail!("{label} {value} exceeds CUDA launch i32 limit");
        }
        Ok(())
    }

    fn launch_status(status: c_int) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            bail!("hi-cuda kernel launcher rejected arguments with status {status}");
        }
    }
}

#[cfg(feature = "native-cuda")]
pub use native::*;

pub fn native_cuda_kernels_enabled() -> bool {
    cfg!(feature = "native-cuda")
}
