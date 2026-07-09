#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>

namespace {

// Paged KV cache element type. Phase 1 keeps it `float` (byte-identical to the
// original f32 pages, validating the mechanical refactor); flipping it to
// `__half` halves KV memory and decode attention bandwidth. All paged-KV
// kernels read/write pages through kv_to_float / kv_from_float, overloaded so
// the same source compiles for either element type.
typedef __half kv_t;

__device__ __forceinline__ float kv_to_float(float v) { return v; }
__device__ __forceinline__ float kv_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ void kv_from_float(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_from_float(__half* p, float v) { *p = __float2half(v); }

__device__ __forceinline__ float hi_sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float hi_siluf(float x) {
  return x * hi_sigmoidf(x);
}

__device__ __forceinline__ float hi_softplusf(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return expf(x);
  }
  return log1pf(expf(x));
}

// One block per row: all threads cooperate on the sum-of-squares (coalesced
// strided loads + block reduction), then write the output in parallel. The old
// one-thread-per-row layout left a single thread doing ~2*cols sequential
// unhidden global loads at M=1 (decode), which nsys measured at ~163us/call and
// ~27% of decode time on a 3B model. Reduction order differs from the sequential
// sum, so results are not bit-identical but the relative FP error is ~1e-6.
__global__ void rms_norm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int cols,
    float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* in = input + static_cast<size_t>(row) * cols;
  float* out = output + static_cast<size_t>(row) * cols;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  float local = 0.0f;
  for (int col = tid; col < cols; col += nthreads) {
    float v = in[col];
    local += v * v;
  }
  __shared__ float sdata[256];
  sdata[tid] = local;
  __syncthreads();
  for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / static_cast<float>(cols) + eps);
  for (int col = tid; col < cols; col += nthreads) {
    out[col] = in[col] * inv * weight[col];
  }
}

__global__ void layer_norm_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int cols,
    float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const float* in = input + row * cols;
  float* out = output + row * cols;
  float mean = 0.0f;
  for (int col = 0; col < cols; ++col) {
    mean += in[col];
  }
  mean /= static_cast<float>(cols);
  float variance = 0.0f;
  for (int col = 0; col < cols; ++col) {
    float centered = in[col] - mean;
    variance += centered * centered;
  }
  float inv = rsqrtf(variance / static_cast<float>(cols) + eps);
  for (int col = 0; col < cols; ++col) {
    out[col] = (in[col] - mean) * inv * weight[col] + bias[col];
  }
}

__global__ void silu_mul_kernel(
    const float* gate,
    const float* up,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  float value = gate[idx];
  output[idx] = hi_siluf(value) * up[idx];
}

__global__ void cast_f16_to_f32_kernel(
    const uint16_t* input,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  const __half* halves = reinterpret_cast<const __half*>(input);
  output[idx] = __half2float(halves[idx]);
}

__global__ void cast_bf16_to_f32_kernel(
    const uint16_t* input,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(input);
  output[idx] = __bfloat162float(values[idx]);
}

__device__ float qwen_ssm_mixed_qkv_value(
    const float* qkv,
    int channel,
    int packed_qkvz,
    int state_size,
    int time_step_rank,
    int group_count,
    int head_v_dim,
    int key_dim,
    int value_dim) {
  if (!packed_qkvz) {
    return qkv[channel];
  }
  int repeat = time_step_rank / group_count;
  int value_group_dim = repeat * head_v_dim;
  int source_group_dim = state_size * 2 + value_group_dim * 2;
  if (channel < key_dim) {
    int group = channel / state_size;
    int local = channel % state_size;
    return qkv[group * source_group_dim + local];
  }
  if (channel < 2 * key_dim) {
    int k_channel = channel - key_dim;
    int group = k_channel / state_size;
    int local = k_channel % state_size;
    return qkv[group * source_group_dim + state_size + local];
  }
  int value_channel = channel - 2 * key_dim;
  int group = value_channel / value_group_dim;
  int local = value_channel % value_group_dim;
  return qkv[group * source_group_dim + 2 * state_size + local];
}

__device__ float qwen_ssm_gate_value(
    const float* qkv,
    const float* gate,
    int value_channel,
    int packed_qkvz,
    int state_size,
    int time_step_rank,
    int group_count,
    int head_v_dim) {
  if (!packed_qkvz) {
    return gate[value_channel];
  }
  int repeat = time_step_rank / group_count;
  int value_group_dim = repeat * head_v_dim;
  int source_group_dim = state_size * 2 + value_group_dim * 2;
  int group = value_channel / value_group_dim;
  int local = value_channel % value_group_dim;
  return qkv[group * source_group_dim + 2 * state_size + value_group_dim + local];
}

__global__ void qwen_ssm_streaming_step_kernel(
    const float* qkv,
    const float* gate,
    const float* conv_weight,
    const float* ba,
    const float* dt_bias,
    const float* a_log,
    const float* norm_weight,
    float* conv_ring,
    float* recurrent_state,
    float* scratch,
    float* output,
    int conv_next,
    int conv_len,
    int conv_kernel,
    int conv_dim,
    int state_size,
    int time_step_rank,
    int group_count,
    int head_v_dim,
    int packed_qkvz,
    float eps) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int key_dim = group_count * state_size;
  int value_dim = time_step_rank * head_v_dim;
  int repeat = time_step_rank / group_count;
  int group_ba_dim = repeat * 2;
  int current_slot = conv_next;
  int new_conv_len = conv_len + 1;
  if (new_conv_len > conv_kernel) {
    new_conv_len = conv_kernel;
  }

  float* conv = scratch;
  float* query = scratch + conv_dim;
  float* key = scratch + conv_dim + key_dim;

  for (int channel = 0; channel < conv_dim; ++channel) {
    conv_ring[current_slot * conv_dim + channel] =
        qwen_ssm_mixed_qkv_value(
            qkv,
            channel,
            packed_qkvz,
            state_size,
            time_step_rank,
            group_count,
            head_v_dim,
            key_dim,
            value_dim);
  }

  for (int channel = 0; channel < conv_dim; ++channel) {
    float sum = 0.0f;
    for (int kernel = 0; kernel < conv_kernel; ++kernel) {
      int relative = conv_kernel - 1 - kernel;
      if (relative >= new_conv_len) {
        continue;
      }
      int slot = (current_slot + conv_kernel - relative) % conv_kernel;
      sum += conv_weight[channel * conv_kernel + kernel]
          * conv_ring[slot * conv_dim + channel];
    }
    conv[channel] = hi_siluf(sum);
  }

  for (int group = 0; group < group_count; ++group) {
    float query_norm = 0.0f;
    float key_norm = 0.0f;
    int start = group * state_size;
    for (int state_dim = 0; state_dim < state_size; ++state_dim) {
      float q = conv[start + state_dim];
      float k = conv[key_dim + start + state_dim];
      query_norm += q * q;
      key_norm += k * k;
    }
    float inv_query = rsqrtf(query_norm + 1.0e-6f);
    float inv_key = rsqrtf(key_norm + 1.0e-6f);
    for (int state_dim = 0; state_dim < state_size; ++state_dim) {
      query[start + state_dim] = conv[start + state_dim] * inv_query;
      key[start + state_dim] = conv[key_dim + start + state_dim] * inv_key;
    }
  }

  float q_scale = rsqrtf(static_cast<float>(state_size));
  for (int head = 0; head < time_step_rank; ++head) {
    int group = head / repeat;
    int local_head = head % repeat;
    int q_start = group * state_size;
    int k_start = group * state_size;
    int v_start = head * head_v_dim;
    int ba_group = group * group_ba_dim;
    float beta = hi_sigmoidf(ba[ba_group + local_head]);
    float alpha = ba[ba_group + repeat + local_head];
    float decay = expf(-expf(a_log[head]) * hi_softplusf(alpha + dt_bias[head]));
    int state_start = head * state_size * head_v_dim;

    for (int state_dim = 0; state_dim < state_size; ++state_dim) {
      for (int value_dim = 0; value_dim < head_v_dim; ++value_dim) {
        recurrent_state[state_start + state_dim * head_v_dim + value_dim] *= decay;
      }
    }

    for (int value_dim = 0; value_dim < head_v_dim; ++value_dim) {
      float kv_mem = 0.0f;
      for (int state_dim = 0; state_dim < state_size; ++state_dim) {
        kv_mem += recurrent_state[state_start + state_dim * head_v_dim + value_dim]
            * key[k_start + state_dim];
      }
      float value = conv[2 * key_dim + v_start + value_dim];
      float delta = (value - kv_mem) * beta;
      for (int state_dim = 0; state_dim < state_size; ++state_dim) {
        recurrent_state[state_start + state_dim * head_v_dim + value_dim] +=
            key[k_start + state_dim] * delta;
      }
      float core = 0.0f;
      for (int state_dim = 0; state_dim < state_size; ++state_dim) {
        core += recurrent_state[state_start + state_dim * head_v_dim + value_dim]
            * query[q_start + state_dim]
            * q_scale;
      }
      output[v_start + value_dim] = core;
    }

    float variance = 0.0f;
    for (int value_dim = 0; value_dim < head_v_dim; ++value_dim) {
      float value = output[v_start + value_dim];
      variance += value * value;
    }
    float scale = rsqrtf(variance / static_cast<float>(head_v_dim) + eps);
    for (int value_dim = 0; value_dim < head_v_dim; ++value_dim) {
      int value_channel = v_start + value_dim;
      float z = qwen_ssm_gate_value(
          qkv,
          gate,
          value_channel,
          packed_qkvz,
          state_size,
          time_step_rank,
          group_count,
          head_v_dim);
      output[value_channel] =
          output[value_channel] * scale * norm_weight[value_dim] * hi_siluf(z);
    }
  }
}

__global__ void gelu_kernel(
    const float* input,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  float x = input[idx];
  float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

// GeGLU: gelu(gate) * up, using the tanh gelu approximation (gelu_pytorch_tanh)
// that Gemma uses. Mirrors silu_mul_kernel for SwiGLU models.
__global__ void gelu_mul_kernel(
    const float* gate,
    const float* up,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  float x = gate[idx];
  float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  float gelu = 0.5f * x * (1.0f + tanhf(inner));
  output[idx] = gelu * up[idx];
}

// Gemma logit soft-capping: cap * tanh(x / cap). Monotonic in x, so it never
// changes a greedy argmax; it compresses the distribution for sampling.
__global__ void softcap_kernel(
    const float* input,
    float* output,
    int len,
    float cap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  output[idx] = cap * tanhf(input[idx] / cap);
}

__global__ void add_kernel(
    const float* left,
    const float* right,
    float* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  output[idx] = left[idx] + right[idx];
}

__global__ void add_rowwise_kernel(
    const float* input,
    const float* bias,
    float* output,
    int rows,
    int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) {
    return;
  }
  int col = idx % cols;
  output[idx] = input[idx] + bias[col];
}

__global__ void copy_row_f32_kernel(
    const float* input,
    float* output,
    int row,
    int rows,
    int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  if (row < 0 || row >= rows) {
    output[col] = 0.0f;
    return;
  }
  output[col] = input[row * cols + col];
}

__global__ void add_scaled_row_in_place_kernel(
    float* output,
    const float* row_values,
    int row,
    int rows,
    int cols,
    float scale) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || row < 0 || row >= rows) {
    return;
  }
  output[row * cols + col] += row_values[col] * scale;
}

__global__ void moe_topk_router_kernel(
    const float* scores,
    uint32_t* output_ids,
    float* output_weights,
    int rows,
    int experts,
    int top_k,
    int norm_topk) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_scores = scores + row * experts;
  uint32_t* row_ids = output_ids + row * top_k;
  float* row_weights = output_weights + row * top_k;

  float max_score = -INFINITY;
  for (int expert = 0; expert < experts; ++expert) {
    float value = row_scores[expert];
    if (value > max_score) {
      max_score = value;
    }
  }

  if (!isfinite(max_score)) {
    float weight = (norm_topk && top_k > 1)
        ? 1.0f / static_cast<float>(top_k)
        : 1.0f / static_cast<float>(experts);
    for (int rank = 0; rank < top_k; ++rank) {
      row_ids[rank] = static_cast<uint32_t>(rank);
      row_weights[rank] = weight;
    }
    return;
  }

  float denom = 0.0f;
  for (int expert = 0; expert < experts; ++expert) {
    denom += expf(row_scores[expert] - max_score);
  }
  if (denom <= 0.0f || !isfinite(denom)) {
    float weight = (norm_topk && top_k > 1)
        ? 1.0f / static_cast<float>(top_k)
        : 1.0f / static_cast<float>(experts);
    for (int rank = 0; rank < top_k; ++rank) {
      row_ids[rank] = static_cast<uint32_t>(rank);
      row_weights[rank] = weight;
    }
    return;
  }

  float previous_weight = INFINITY;
  int previous_id = -1;
  float selected_sum = 0.0f;
  for (int rank = 0; rank < top_k; ++rank) {
    int best_id = -1;
    float best_weight = -1.0f;
    for (int expert = 0; expert < experts; ++expert) {
      float weight = expf(row_scores[expert] - max_score) / denom;
      if (!isfinite(weight)) {
        weight = 0.0f;
      }
      bool eligible = previous_id < 0 || weight < previous_weight ||
                      (weight == previous_weight && expert > previous_id);
      if (!eligible) {
        continue;
      }
      if (best_id < 0 || weight > best_weight ||
          (weight == best_weight && expert < best_id)) {
        best_id = expert;
        best_weight = weight;
      }
    }
    if (best_id < 0) {
      best_id = rank < experts ? rank : experts - 1;
      best_weight = 0.0f;
    }
    row_ids[rank] = static_cast<uint32_t>(best_id);
    row_weights[rank] = best_weight;
    selected_sum += best_weight;
    previous_id = best_id;
    previous_weight = best_weight;
  }

  if (norm_topk && top_k > 1 && selected_sum > 1.0e-7f &&
      isfinite(selected_sum)) {
    for (int rank = 0; rank < top_k; ++rank) {
      row_weights[rank] /= selected_sum;
    }
  }
}

__global__ void cast_f32_to_f16_kernel(
    const float* input,
    __half* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  output[idx] = __float2half(input[idx]);
}

__global__ void cast_f32_to_bf16_kernel(
    const float* input,
    __nv_bfloat16* output,
    int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= len) {
    return;
  }
  output[idx] = __float2bfloat16(input[idx]);
}

__global__ void gather_rows_f16_to_f32_kernel(
    const __half* matrix,
    const uint32_t* row_ids,
    float* output,
    int row_count,
    int cols,
    int matrix_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = row_count * cols;
  if (idx >= total) {
    return;
  }
  int out_row = idx / cols;
  int col = idx % cols;
  uint32_t matrix_row = row_ids[out_row];
  if (matrix_row >= static_cast<uint32_t>(matrix_rows)) {
    output[idx] = 0.0f;
    return;
  }
  output[idx] = __half2float(matrix[matrix_row * cols + col]);
}

__global__ void gather_rows_bf16_to_f32_kernel(
    const __nv_bfloat16* matrix,
    const uint32_t* row_ids,
    float* output,
    int row_count,
    int cols,
    int matrix_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = row_count * cols;
  if (idx >= total) {
    return;
  }
  int out_row = idx / cols;
  int col = idx % cols;
  uint32_t matrix_row = row_ids[out_row];
  if (matrix_row >= static_cast<uint32_t>(matrix_rows)) {
    output[idx] = 0.0f;
    return;
  }
  output[idx] = __bfloat162float(matrix[matrix_row * cols + col]);
}

__global__ void gather_rows_f32_to_f32_kernel(
    const float* matrix,
    const uint32_t* row_ids,
    float* output,
    int row_count,
    int cols,
    int matrix_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = row_count * cols;
  if (idx >= total) {
    return;
  }
  int out_row = idx / cols;
  int col = idx % cols;
  uint32_t matrix_row = row_ids[out_row];
  if (matrix_row >= static_cast<uint32_t>(matrix_rows)) {
    output[idx] = 0.0f;
    return;
  }
  output[idx] = matrix[matrix_row * cols + col];
}

__device__ void q4_k_scale_min(int index, const uint8_t* scales, uint8_t* scale, uint8_t* min) {
  if (index < 4) {
    *scale = scales[index] & 0x3f;
    *min = scales[index + 4] & 0x3f;
  } else {
    *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    *min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
  }
}

__device__ int q3_k_scale(int index, const uint8_t* scales) {
  uint8_t low = index < 8 ? (scales[index] & 0x0f) : (scales[index - 8] >> 4);
  uint8_t high = (scales[8 + (index % 4)] >> (2 * (index / 4))) & 0x03;
  return static_cast<int>(low | (high << 4)) - 32;
}

__device__ __constant__ int8_t IQ4_NL_VALUES[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __constant__ uint16_t IQ2_XXS_GRID[256] = {
    0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130, 138,
    162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642, 1025, 1028, 1040,
    1057, 1060, 1088, 1090, 1096, 1120, 1153, 1156, 1168, 1188, 1280, 1282, 1288,
    1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056,
    2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097, 4100,
    4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480,
    4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145,
    6148, 6160, 6208, 6273, 6400, 6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289,
    8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537,
    9572, 9872, 10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417,
    16420, 16448, 16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672, 16737,
    16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536,
    17561, 17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776,
    18785, 18818, 19013, 19088, 20480, 20488, 20497, 20505, 20512, 20608, 20616,
    20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112,
    25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770, 32778, 32833,
    32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888,
    34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880,
    36900, 36928, 37025, 37142, 37248, 37445, 37888, 37922, 37956, 38225, 39041,
    39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268};

__device__ __constant__ uint16_t IQ2_XS_GRID[512] = {
    0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85,
    88, 97, 100, 128, 130, 133, 136, 145, 148, 153, 160, 257, 260, 262, 265, 272,
    274, 277, 280, 282, 289, 292, 320, 322, 325, 328, 337, 340, 352, 360, 385,
    388, 400, 512, 514, 517, 520, 529, 532, 544, 577, 580, 592, 597, 640, 650,
    1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1088, 1090, 1093,
    1096, 1105, 1108, 1110, 1120, 1153, 1156, 1168, 1280, 1282, 1285, 1288, 1297,
    1300, 1312, 1345, 1348, 1360, 1377, 1408, 1537, 1540, 1552, 1574, 1600, 1602,
    1668, 2048, 2050, 2053, 2056, 2058, 2065, 2068, 2080, 2085, 2113, 2116, 2128,
    2136, 2176, 2208, 2218, 2305, 2308, 2320, 2368, 2433, 2441, 2560, 2592, 2600,
    2710, 2720, 4097, 4100, 4102, 4105, 4112, 4114, 4117, 4120, 4129, 4132, 4160,
    4162, 4165, 4168, 4177, 4180, 4192, 4202, 4225, 4228, 4240, 4352, 4354, 4357,
    4360, 4369, 4372, 4384, 4417, 4420, 4432, 4480, 4500, 4502, 4609, 4612, 4614,
    4624, 4672, 4704, 5120, 5122, 5125, 5128, 5137, 5140, 5152, 5185, 5188, 5193,
    5200, 5220, 5248, 5377, 5380, 5392, 5440, 5632, 5652, 5705, 6145, 6148, 6160,
    6162, 6208, 6228, 6278, 6400, 6405, 6502, 6737, 6825, 8192, 8194, 8197, 8200,
    8202, 8209, 8212, 8224, 8257, 8260, 8272, 8320, 8352, 8449, 8452, 8464, 8512,
    8520, 8549, 8704, 8738, 8832, 8872, 9217, 9220, 9232, 9257, 9280, 9472, 9537,
    9554, 9625, 9729, 9754, 9894, 10240, 10248, 10250, 10272, 10325, 10376, 10402,
    10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388, 16390, 16393,
    16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458,
    16465, 16468, 16480, 16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648,
    16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773, 16802, 16897, 16900,
    16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433,
    17440, 17473, 17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728,
    17818, 17920, 17930, 17988, 18000, 18433, 18436, 18448, 18496, 18501, 18516,
    18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488,
    20497, 20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752,
    20757, 20800, 20802, 20992, 21060, 21162, 21505, 21508, 21520, 21537, 21568,
    21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
    22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592,
    24640, 24660, 24674, 24710, 24745, 24832, 25124, 25162, 25234, 25600, 25622,
    25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234, 32768,
    32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896,
    32898, 32936, 32938, 33025, 33028, 33030, 33040, 33088, 33105, 33113, 33280,
    33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810, 33813, 33856,
    33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906,
    34944, 34946, 34984, 35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868,
    36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462, 37513, 37888, 37893,
    37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445,
    39520, 40960, 40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382,
    41472, 41474, 41480, 41514, 41600, 41632, 42048, 42133, 42597, 42648, 43018,
    43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665,
    43690};

__device__ __constant__ uint8_t IQ2_XXS_VALUES[4] = {8, 25, 43, 0};

__device__ __constant__ uint32_t IQ3_XXS_GRID[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404,
    0x04041414, 0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c,
    0x040c0c04, 0x040c0c14, 0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c,
    0x040c2c24, 0x040c3e04, 0x04140404, 0x04140414, 0x04140424, 0x04140c0c, 0x04141404,
    0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e, 0x04142c0c, 0x04142c3e, 0x04143e2c,
    0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c, 0x041c3e04, 0x04240c1c,
    0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c, 0x042c043e,
    0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c,
    0x0c04141c, 0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404,
    0x0c0c0414, 0x0c0c0c0c, 0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04,
    0x0c140c14, 0x0c14140c, 0x0c141c04, 0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404,
    0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c, 0x0c24042c, 0x0c242c04, 0x0c2c1404,
    0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414, 0x0c3e2404, 0x14040404,
    0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434, 0x14041c0c,
    0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c,
    0x14140c3e, 0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c,
    0x141c0c04, 0x141c0c24, 0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c,
    0x142c143e, 0x142c240c, 0x142c3e24, 0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c,
    0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c, 0x1c04141c, 0x1c042c04, 0x1c04342c,
    0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c, 0x1c0c2424, 0x1c0c2434,
    0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14, 0x1c1c0c0c,
    0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424,
    0x24040c3e, 0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404,
    0x24141c3e, 0x24142404, 0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424,
    0x24242c0c, 0x24243424, 0x242c142c, 0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04,
    0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c, 0x2c043e04, 0x2c0c0404, 0x2c0c0434,
    0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14, 0x2c1c0414, 0x2c1c2c1c,
    0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c, 0x2c342c04,
    0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c,
    0x342c2c14, 0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e,
    0x3e040c04, 0x3e041c14, 0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c,
    0x3e142c14, 0x3e1c0404, 0x3e1c0c2c, 0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c,
    0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04};

__device__ __constant__ char IQ1_S_GRID_HEX[] =
    "00000200050008000a00110015002000220028002a00450051005400560065008000820088008a009500a000a200a800"
    "aa000401050111011401160119011a012501410146014901520155015a0161016401660168018501910194019601a501"
    "0002020208020a0215022002220228022a02450251025902640269028002820288028a02910295029902a002a202a802"
    "aa0211041404160425044104490455045a046404650491049904a5040105040505050605150518051a05290540054505"
    "4a0550055105540555055605590560056205650568056a0581059105950598059a05a105a405a505a605a90514061906"
    "410644065006520655065806600661066606690685069106940699060008020808080a0815082008220828082a084508"
    "5108560865088008820888088a089508a008a208a808aa08050911091409190924092509410950095109550961096409"
    "69099109940996099909a509000a020a080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
    "8a0a950aa00aa20aa80aaa0a101011101410191024102510411044105010551058106110641065106910911094109610"
    "a110a5100111041106110911101112111511181121112411291145114a11501151115211541155115611591160116511"
    "841192119511a111a41111121412161225124012461249125212551258125a12641266128512911294129612a5120114"
    "0614091414141514181419142114261441144514461448144a1451145414551456145914621465146814841489149014"
    "94149514981499149a14a114a414a514a914021505150a151115141515151615191520152215251528152a1541154415"
    "451546155115521554155515561559155a1561156415651566156915801582158415851588158a159015911594159515"
    "961599159a15a015a215a51501160416051606161516161618161a1621162616401642164416451648164a1651165516"
    "561658165916611664166516681669166a1686168a1692169516a416a916111816182518411844184618491850185518"
    "58185a1860186118641866186918851891189418a5181019121915191a19211925194219441945194819511954195519"
    "561959195a19601965196a1989199119921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
    "581a611a661a691a851a911a961a9a1a0020022008200a20152020202220252028202a20452051205920612065208020"
    "822088208a209520a020a220a520a820aa2005211121142119212521422144214921552158215a216121642165216621"
    "8521902196219921a521012208220a22112215222022222228222a2245225122562259226522812288228a2291229522"
    "a022a222a822aa220524142416241924252444244524462449245224552458245a2466248524912494249924a124a524"
    "09251525212529254025452548255125542555255925622565256825892590259425952598259a25a125a425a625a925"
    "052610261226192625264126492655266026612669268426862690269a260028022808280a2815282028222828282a28"
    "45285128542865288028822888288a28a028a228a828aa28092911291429192925294629492952295529612964296629"
    "69298529902996299929a429a529002a022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
    "952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440664094409940a140a640004101410441"
    "0641094112411541164118411a41214126412941454148414a41514154415541564159415a41654168416a4181418441"
    "8641904192419541a041a141a241054211421442164225424142524255425a426442694289429442a542014415441944"
    "2944454448444a44514454445544564461446244654468446a44814486448944904492449544a044a144a94401450245"
    "05450a4511451445154516451945204525452a4541454445454546454945504551455445554556455845594561456445"
    "6545664569458245844585458845914594459545964599459a45a545a845aa450146054609461446154618461a462146"
    "244629464046424645464846504651465246554656465946624665466846814685468a4694469546a146a446a6460548"
    "114815481a48254842484948504855485848614864486648694885489148944896489948a5480149054906490a491049"
    "144915491849214924492649404945494a4951495249544955495649594960496249654966496a498649894992499549"
    "96499849a149a449a649a949164a444a464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
    "1a5021502450295040504550485051505450555056505950655068508650895095509850a050a150a650a95005510851"
    "09510a5111511451155116511851195120512551265128512a5141514451455146514951505151515251545155515651"
    "585159515a51615164516551665169518251855191519451955196519951a051a551aa5101520652125215521a522152"
    "2452425245524a525152545255525652595262526552855290529252955299529a52a452045405541154145415541654"
    "185419542154255428542a54415444544554465449544a5450545154545455545654585459545a546154625464546554"
    "66546954805488548a5491549454955496549954a154a454a554aa540155025504550555065509551055115512551455"
    "1555165519551a5521552455255526552955405541554255445545554655485549555055515552555455555556555855"
    "59555a5560556155645565556655685569556a5581558455855589558a559055915594559555965598559955a155a455"
    "a555a655a955005601560256045606560856095611561456155618561956205621562256245625562656285629564156"
    "45564656485649564a56505651565256545655565656585659565a566156645665566956825685568656885689568a56"
    "915695569a56a256a556a656a856a956045805580658095810581558185821582a58455848584a585158545855585658"
    "585859586058625864586558825889589058925895589858a158a9580159025905590a59115914591559165919592559"
    "41594459455946594959505951595259545955595659585959595a596159645965596659695981598559895991599459"
    "9559965998599959a559045a085a155a1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
    "6a5a815a8a5a925a955a965a985a9a5aa15a05601460166019602560446050605560566058605a606160646066606960"
    "81609660a5600161046106610961126115612161226126612961456149615161556156615961656166616a6184618a61"
    "92619561a161a661a9611162166219624062416246625562566258626062856291629662a56211641264156416641a64"
    "21642664296440644264456448644a64516454645564566459645a646064626465648464856489649064926494649564"
    "966498649a64a164a464a964056508650a65116515651665196544654565466549655065516554655565566559656165"
    "6465656566656965866589658a6591659565966599659a65a265a565a665a86502660966156620662666286629664066"
    "456648664a66516654665566566658665a666066656668668066826685668a669466966698669966a066a466a666aa66"
    "1668196825684168526855685a6861686968856891689868a66801690469106915692169246926692969406941694569"
    "4669486951695469556956695969606965696a69826984698a699569a169a469a569a969116a166a186a416a446a496a"
    "506a556a586a5a6a646a656a696a866a946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
    "5680598065808080828088808a809580a080a280a880aa80058111811481168119812581418144814981508152815581"
    "56815881598164816681698185818981948196819981a5810082028208820a8215822082228228822a82518254825982"
    "65828082828288828a829582a082a282a882aa821484198441844484518455845a846184648469849484998401850985"
    "128515851a85268529854085418545854885518554855585568559855a856585668568856a8581858485868589859085"
    "928595859885a68511861686198625864186448649864a865086558659865a86618666866a86858691869a86a4860088"
    "028808880a8815882088228828882a8841884588518854885988658869888088828888888a889588a088a288a888aa88"
    "05890689118914891689258941894489468949895089528955895a8961896489858996899989a589008a028a088a0a8a"
    "158a208a228a288a2a8a458a518a548a568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
    "419046904990559058905a9069906a9085909190949096909990a59001910491069109911091159118911a9121912491"
    "26912991409145915091519154915591569159916291659184918691929195919891a191a491a691a991059211921492"
    "19922592449246924992509252925592589266926992859294929692a992019404940694109415941894269440944a94"
    "5194549455945694589459946094619462946594849486949294949495949894a194a9940095059508950a9510951195"
    "14951595169519952195259529952a9541954495459546954995509551955295549555955695589559955a9561956495"
    "6595669569958195859588959195929594959595969599959a95a095a295a595a895aa95019604961096159619962096"
    "2696299645964896499651965296559656965996659668968296849689968a96929694969596a496a696a99605981698"
    "199825984198469850985298559856985a98649865988598919896989998a59804990699099910991299159918991a99"
    "209921992499269940994299459948994a99519954995599569959996299659966996a99819984999099929995999a99"
    "a199a699059a159a259a449a469a499a509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
    "28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa005a109a111a114a116a119a11aa146a1"
    "49a151a155a158a15aa161a164a185a190a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
    "65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a455a458a45aa461a465a466a468a469a4"
    "85a406a509a510a512a515a518a526a529a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
    "95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a662a686a690a695a696a699a6a1a6a4a6"
    "a6a600a802a808a80aa820a822a828a82aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
    "19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa20aa22aa28aa2aaa51aa54aa56aa80aa"
    "82aa88aa8aaa95aaa0aaa2aaa8aaaaaa";

__device__ __constant__ char IQ3_S_GRID_HEX[] =
    "000001000200050007001000110012001400160020002100250033004000420045004700510053006000620071007400"
    "770000010101020104011001110115012001230127013101350144016101650172010002010205020702100213021602"
    "210225023002340242024502470251025302700273020303110315032003220331033303360344035003520367037103"
    "750300041304170421042404320440044304510470040205040520052205260533054105450547056605730506061106"
    "130631065206710600070207040720072207260733075007540700100110021004101010111013101510171020102210"
    "311034103610541056106110721000110111031106111011141121113011331141115011521170117611001212121512"
    "171220122412321240124312551260127212011304130713101313132113271330133413411362137013031405141214"
    "141431143314421446145014541401151015131521153015321551152016241627164416461601170317101712172117"
    "351741176217701700200120032005200720102012201420162021202320272030203220412043204520502052206720"
    "702073207520002102211021132117212221252131213421422151210122042207222122232230223722412253225722"
    "712274220023022305231123222324233123332342235023662301240724202423243224352441247224752404251125"
    "222537254025532570250026022607262126552661260527112726273027432750270230113013301530173022303130"
    "333035304230443047305130633071300131033105311431213123314031603172317631003212322032323234325032"
    "013310331433213323332733303341334333473355337333033411341634223431345234603464340135103512352535"
    "323544355635733516364136013703372037223735370040044012402040244027403240414050407040024107411141"
    "134122413041354143415141554101420342104215422142334240425742624270420443114313432043224331433543"
    "004402442444374440447144054507452145624513463446604610471547304743475147025010501450225040504450"
    "475052506650745001510351055112512151325172510052115223523052365253520253075310532753445351536553"
    "735301540454205432544654125526555155535542560257045722571160136015603160336060600061206127616461"
    "126234624262556262627062006314632163406325644364626400650365346560650566406611671367007004700770"
    "207022703670407054706270027111712471437145710172047210721672217230725172027332733573537301740574"
    "13742074507422754275027631760077";

__device__ __constant__ char IQ2_S_GRID_HEX[] =
    "00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100640066006900800082008500880091009400a000"
    "a500aa0001010401060109011001120115011801210124014001420145014801510154015601590160016501680181018401900192019501a101a40100020202"
    "050208021102140220022a02410244024602490250025502800285028a029402a202010404040604090410041204150418042104240426042904400442044504"
    "48044a0451045404560459046004620465048104840486048904900495049804a104a40400050205050508050a05110514051605190520052505280541054405"
    "46054905500552055505580561056405800582058505880591059405a00501060406060609061006150640064506480651065406600681068406900600080208"
    "050808081108140816081908200825082a084108440846084908500852085508580861086408800885089408aa08010904091009120915091809210940094509"
    "480951095409600981099009000a110a140a220a280a2a0a500a990a011004100610091010101210151018102110241026104010421045104810511054105610"
    "59106010621065106810811084108610901095109810a110a41000110211051108110a1111111411161119112011221125112811411144114611491150115211"
    "55115811611164118011821185118811911194110112041209121012151221122412401245125112541281128412901200140214051408141114141416141914"
    "2014251428144114441446144914501452145514581461146414801482148514881491149414a014011504150615091510151215151518152115241540154215"
    "451548155115541560158115841590150016051608161116141620164116441650168016aa160118041806180918101815181818211840184218451848185118"
    "541860188118841800190219051908191119141920194119441950196919a219041a101a401a561a00200220052008201120142016201920202025202a204120"
    "4420502052205520642080208a209420aa2001210421102112211521212140214221452151215421602181218421902100220a22222228222a22442250228822"
    "8a22a822012404240624092410241524182421242424402442244524482451245424602481248424902400250525082511251425202541254425502566258025"
    "0126042610264026592600280528112814284128442850288a28aa2801290429102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
    "21402440264040404240454048404a40514054405640594060406240654081408440904095409840a140a4400041024105410841114114411641194120412241"
    "25414141444146414941504152415541584161416441804182418541884191419441a04101420442104212421542184224424042454248425142544260428142"
    "844200440244054408440a44114414441644194420442244254428444144444446444944504452445544584461446444804482448544884491449444a0440145"
    "04450645094510451245154518452145244540454245454548455145544560456a4581458445904500460246054608461146144620464146444650468046a546"
    "014804480948104812481548184821482448404842484548484851485448604884489048004902490549084911491449204941494449504980499649014a044a"
    "104a404a005002500550085011501450165019502050225025502850415044504650495050505250555058506150645080508250855088509150945001510451"
    "06510951105112511551185121512451405142514551485151515451605181518451905100520552085211521452205241524452505269528052015404540654"
    "09541054125415541854215424544054425445544854515454546054815484549054005502550555085511551455205541554455505580550156045610562656"
    "405600580258055808581158145820584158445850585a5880580159045910594059005a195a855aa85a01600460066010601260156018602160246040604560"
    "4860516054606060846090600061026105610861116114612061416144615061806199610462106240625662a162006405640864116414642064416444645064"
    "806401650465106540654a6568659265006694660168046810686568986800692a69426aa16a0080028005800880118014801980208025804180448050805280"
    "5580588061808080858091809480018104810981108112811581188121812481408142814581488151815481818184819081a981008205820a82118214824182"
    "44825082018404840684098410841284158418842184408442844584488451845484608481848484908400850285058508851185148520854185448550858085"
    "8a85018604861086298640860088058811881488418844885088a2880189048940896589228a588a5a8a828aa28a019004900990109012901590189024904090"
    "42904590489051905490609081908490909000910591119114914191449150915a910192049210924092a6920094029405940894119414942094419444945094"
    "8094969401950495109540959895a19500964696649601980498109826984098a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
    "40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa";

__device__ __constant__ uint8_t TQ_POW3[6] = {1, 3, 9, 27, 81, 243};

__device__ __constant__ int8_t MXFP4_VALUES[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float e8m0_to_float_half(uint8_t value) {
  uint32_t bits = value < 2 ? (0x00200000u << value) : (static_cast<uint32_t>(value - 1) << 23);
  return __uint_as_float(bits);
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t value) {
  if (value == 0 || value == 0x7f) {
    return 0.0f;
  }
  int exponent = (value >> 3) & 0x0f;
  int mantissa = value & 0x07;
  float raw = exponent == 0
                  ? ldexpf(static_cast<float>(mantissa), -9)
                  : ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
  return raw * 0.5f;
}

__global__ void dequantize_q8_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 34;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  int8_t q = static_cast<int8_t>(block[2 + within]);
  output[idx] = d * static_cast<float>(q);
}

__global__ void dequantize_q8_1_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 36;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  int8_t q = static_cast<int8_t>(block[4 + within]);
  output[idx] = d * static_cast<float>(q);
}

__global__ void dequantize_q4_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 18;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint8_t packed = block[2 + (within % 16)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(static_cast<int>(quant) - 8);
}

// Dequantize Q4_0 straight to f16 (no f32 intermediate + separate cast). The
// prefill f16-GEMM path used dequant->f32 then a cast_f32_to_f16 pass, which for
// short prefills is ~40% of GPU time (write 4B/weight then re-read+write 2B) plus
// a >4MB f32 scratch per weight whose synchronizing cudaFree dominates host time.
// Same value as dequantize_q4_0_kernel then f32->f16, so bit-identical f16 output.
__global__ void dequantize_q4_0_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 18;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint8_t packed = block[2 + (within % 16)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = __float2half(d * static_cast<float>(static_cast<int>(quant) - 8));
}

extern "C" int hi_cuda_launch_dequantize_q4_0_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q4_0_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

__global__ void dequantize_q4_1_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 20;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float m = __half2float(*reinterpret_cast<const __half*>(block + 2));
  uint8_t packed = block[4 + (within % 16)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(quant) + m;
}

__global__ void dequantize_q1_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 128;
  int within = idx % 128;
  const uint8_t* block = input + block_id * 18;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint8_t quant = static_cast<uint8_t>((block[2 + within / 8] >> (within & 7)) & 1);
  output[idx] = quant != 0 ? d : -d;
}

__global__ void dequantize_mxfp4_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 17;
  float d = e8m0_to_float_half(block[0]);
  uint8_t packed = block[1 + (within & 15)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(MXFP4_VALUES[quant]);
}

__global__ void dequantize_nvfp4_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 64;
  int within = idx % 64;
  int sub_block = within / 16;
  int sub_offset = within % 16;
  const uint8_t* block = input + block_id * 36;
  const uint8_t* qs = block + 4;
  float d = ue4m3_to_float(block[sub_block]);
  uint8_t packed = qs[sub_block * 8 + (sub_offset & 7)];
  uint8_t quant = sub_offset < 8 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(MXFP4_VALUES[quant]);
}

__global__ void dequantize_iq4_nl_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 18;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint8_t packed = block[2 + (within % 16)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(IQ4_NL_VALUES[quant]);
}

__global__ void dequantize_iq4_xs_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  const uint8_t* block = input + block_id * 136;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint16_t scales_h = static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8);
  const uint8_t* scales_l = block + 4;
  const uint8_t* qs = block + 8;

  uint8_t scale_low = (scales_l[group32 / 2] >> (4 * (group32 % 2))) & 0x0f;
  uint8_t scale_high = static_cast<uint8_t>((scales_h >> (2 * group32)) & 0x03);
  float dl = d * static_cast<float>(static_cast<int>(scale_low | (scale_high << 4)) - 32);

  uint8_t packed = qs[group32 * 16 + (offset32 % 16)];
  uint8_t quant = offset32 < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = dl * static_cast<float>(IQ4_NL_VALUES[quant]);
}

__device__ __forceinline__ uint8_t iq2_xxs_signs(uint8_t index) {
  return index | ((__popc(static_cast<unsigned int>(index)) & 1) ? 0x80 : 0x00);
}

__global__ void dequantize_iq2_xxs_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 66;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* q = block + 2 + group32 * 8;
  uint32_t aux32 = static_cast<uint32_t>(q[4]) |
                   (static_cast<uint32_t>(q[5]) << 8) |
                   (static_cast<uint32_t>(q[6]) << 16) |
                   (static_cast<uint32_t>(q[7]) << 24);
  float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
  uint16_t grid = IQ2_XXS_GRID[q[lane]];
  uint8_t value = IQ2_XXS_VALUES[(grid >> (2 * j)) & 0x03];
  uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((aux32 >> (7 * lane)) & 0x7f));
  float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
  output[idx] = db * static_cast<float>(value) * sign;
}

__global__ void dequantize_iq2_xs_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 74;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* qs = block + 2;
  const uint8_t* scales = block + 66;
  int q_index = group32 * 4 + lane;
  uint16_t q = static_cast<uint16_t>(qs[2 * q_index]) |
               (static_cast<uint16_t>(qs[2 * q_index + 1]) << 8);
  uint8_t scale = scales[group32];
  float db = d *
             (0.5f + static_cast<float>(lane < 2 ? (scale & 0x0f) : (scale >> 4))) *
             0.25f;
  uint16_t grid = IQ2_XS_GRID[q & 0x01ff];
  uint8_t value = IQ2_XXS_VALUES[(grid >> (2 * j)) & 0x03];
  uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((q >> 9) & 0x7f));
  float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
  output[idx] = db * static_cast<float>(value) * sign;
}

__global__ void dequantize_iq3_xxs_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 98;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* qs = block + 2;
  const uint8_t* scales_and_signs = block + 66;
  const uint8_t* aux = scales_and_signs + 4 * group32;
  uint32_t aux32 = static_cast<uint32_t>(aux[0]) |
                   (static_cast<uint32_t>(aux[1]) << 8) |
                   (static_cast<uint32_t>(aux[2]) << 16) |
                   (static_cast<uint32_t>(aux[3]) << 24);
  float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
  uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((aux32 >> (7 * lane)) & 0x7f));
  uint8_t q = qs[8 * group32 + 2 * lane + (j >= 4 ? 1 : 0)];
  uint32_t grid = IQ3_XXS_GRID[q];
  uint8_t value = static_cast<uint8_t>((grid >> (8 * (j & 3))) & 0xff);
  float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
  output[idx] = db * static_cast<float>(value) * sign;
}

__device__ __forceinline__ uint8_t iq1_s_hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint8_t>(value - 'a' + 10);
  }
  return static_cast<uint8_t>(value - 'A' + 10);
}

__device__ __forceinline__ uint8_t iq1_s_grid_code(int grid_index, int j) {
  int packed_value = grid_index * 8 + j;
  int byte_offset = packed_value >> 2;
  uint8_t byte = static_cast<uint8_t>((iq1_s_hex_nibble(IQ1_S_GRID_HEX[2 * byte_offset]) << 4) |
                                      iq1_s_hex_nibble(IQ1_S_GRID_HEX[2 * byte_offset + 1]));
  return static_cast<uint8_t>((byte >> (2 * (packed_value & 3))) & 0x03);
}

__device__ __forceinline__ uint8_t iq3_s_grid_value(int grid_index, int j) {
  int packed_value = grid_index * 4 + j;
  int byte_offset = packed_value >> 1;
  uint8_t byte = static_cast<uint8_t>((iq1_s_hex_nibble(IQ3_S_GRID_HEX[2 * byte_offset]) << 4) |
                                      iq1_s_hex_nibble(IQ3_S_GRID_HEX[2 * byte_offset + 1]));
  return static_cast<uint8_t>(1 + 2 * ((byte >> (4 * (packed_value & 1))) & 0x07));
}

__device__ __forceinline__ uint8_t iq2_s_grid_value(int grid_index, int j) {
  int packed_value = grid_index * 8 + j;
  int byte_offset = packed_value >> 2;
  uint8_t byte = static_cast<uint8_t>((iq1_s_hex_nibble(IQ2_S_GRID_HEX[2 * byte_offset]) << 4) |
                                      iq1_s_hex_nibble(IQ2_S_GRID_HEX[2 * byte_offset + 1]));
  uint8_t code = static_cast<uint8_t>((byte >> (2 * (packed_value & 3))) & 0x03);
  return code == 0 ? 0x08 : (code == 1 ? 0x19 : 0x2b);
}

__device__ __forceinline__ float f16_bits_to_float(uint16_t bits) {
  __half value = *reinterpret_cast<const __half*>(&bits);
  return __half2float(value);
}

__global__ void dequantize_iq1_s_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 50;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* qs = block + 2;
  const uint8_t* qh = block + 34;
  uint16_t qh_word = static_cast<uint16_t>(qh[2 * group32]) |
                     (static_cast<uint16_t>(qh[2 * group32 + 1]) << 8);
  float dl = d * static_cast<float>(2 * ((qh_word >> 12) & 7) + 1);
  float delta = (qh_word & 0x8000) != 0 ? -1.125f : -0.875f;
  int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                   (static_cast<int>((qh_word >> (3 * lane)) & 0x07) << 8);
  uint8_t code = iq1_s_grid_code(grid_index, j);
  output[idx] = dl * (static_cast<float>(code) + delta);
}

__global__ void dequantize_iq1_m_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 56;
  const uint8_t* qs = block;
  const uint8_t* qh = block + 32;
  const uint8_t* scales = block + 48;
  uint16_t sc[4] = {
      static_cast<uint16_t>(static_cast<uint16_t>(scales[0]) |
                            (static_cast<uint16_t>(scales[1]) << 8)),
      static_cast<uint16_t>(static_cast<uint16_t>(scales[2]) |
                            (static_cast<uint16_t>(scales[3]) << 8)),
      static_cast<uint16_t>(static_cast<uint16_t>(scales[4]) |
                            (static_cast<uint16_t>(scales[5]) << 8)),
      static_cast<uint16_t>(static_cast<uint16_t>(scales[6]) |
                            (static_cast<uint16_t>(scales[7]) << 8)),
  };
  uint16_t scale_bits = static_cast<uint16_t>((sc[0] >> 12) |
                                              ((sc[1] >> 8) & 0x00f0) |
                                              ((sc[2] >> 4) & 0x0f00) |
                                              (sc[3] & 0xf000));
  float d = f16_bits_to_float(scale_bits);
  int scale_index = 2 * group32 + lane / 2;
  uint16_t scale_word = sc[scale_index / 4];
  float dl = d * static_cast<float>(2 * ((scale_word >> (3 * (scale_index & 3))) & 0x07) + 1);
  uint8_t qh_byte = qh[2 * group32 + lane / 2];
  int qh_shift = 4 * (lane & 1);
  float delta = (qh_byte & (0x08u << qh_shift)) != 0 ? -1.125f : -0.875f;
  int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                   (static_cast<int>((qh_byte >> qh_shift) & 0x07) << 8);
  uint8_t code = iq1_s_grid_code(grid_index, j);
  output[idx] = dl * (static_cast<float>(code) + delta);
}

__global__ void dequantize_iq2_s_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 82;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* qs = block + 2;
  const uint8_t* signs = block + 34;
  const uint8_t* qh = block + 66;
  const uint8_t* scales = block + 74;
  float db = d * (0.5f + static_cast<float>(scales[group32] >> 4)) * 0.25f;
  int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                   (static_cast<int>((qh[group32] >> (2 * lane)) & 0x03) << 8);
  uint8_t value = iq2_s_grid_value(grid_index, j);
  uint8_t signs_byte = signs[4 * group32 + lane];
  float sign = (signs_byte & (1u << j)) != 0 ? -1.0f : 1.0f;
  output[idx] = db * static_cast<float>(value) * sign;
}

__global__ void dequantize_iq3_s_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  int group32 = within / 32;
  int offset32 = within % 32;
  int lane = offset32 / 8;
  int j = offset32 % 8;

  const uint8_t* block = input + block_id * 110;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* qs = block + 2;
  const uint8_t* qh = block + 66;
  const uint8_t* signs = block + 74;
  const uint8_t* scales = block + 106;
  uint8_t scale_byte = scales[group32 / 2];
  uint8_t scale = (scale_byte >> (4 * (group32 & 1))) & 0x0f;
  float db = d * static_cast<float>(1 + 2 * scale);
  int q_slot = 2 * lane + (j >= 4 ? 1 : 0);
  int grid_index = static_cast<int>(qs[8 * group32 + q_slot]) |
                   (static_cast<int>((qh[group32] >> q_slot) & 0x01) << 8);
  uint8_t value = iq3_s_grid_value(grid_index, j & 3);
  uint8_t signs_byte = signs[4 * group32 + lane];
  float sign = (signs_byte & (1u << j)) != 0 ? -1.0f : 1.0f;
  output[idx] = db * static_cast<float>(value) * sign;
}

__device__ __forceinline__ float dequantize_tq_value(uint8_t value, uint8_t pow, float d) {
  uint8_t q = static_cast<uint8_t>(value * pow);
  int xi = (static_cast<int>(q) * 3) >> 8;
  return static_cast<float>(xi - 1) * d;
}

__global__ void dequantize_tq1_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 54;
  const uint8_t* qs = block;
  const uint8_t* qh = block + 48;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 52));

  uint8_t value;
  uint8_t pow;
  if (within < 160) {
    int n = within / 32;
    int m = within % 32;
    value = qs[m];
    pow = TQ_POW3[n];
  } else if (within < 240) {
    int tail = within - 160;
    int n = tail / 16;
    int m = tail % 16;
    value = qs[32 + m];
    pow = TQ_POW3[n];
  } else {
    int tail = within - 240;
    int n = tail / 4;
    int m = tail % 4;
    value = qh[m];
    pow = TQ_POW3[n];
  }
  output[idx] = dequantize_tq_value(value, pow, d);
}

__global__ void dequantize_tq2_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 66;
  const uint8_t* qs = block;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 64));

  int half = within / 128;
  int rem = within % 128;
  int lane = rem / 32;
  int m = rem % 32;
  int quant = (qs[half * 32 + m] >> (lane * 2)) & 0x03;
  output[idx] = static_cast<float>(quant - 1) * d;
}

__global__ void dequantize_q5_0_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 22;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint32_t qh = static_cast<uint32_t>(block[2]) |
                (static_cast<uint32_t>(block[3]) << 8) |
                (static_cast<uint32_t>(block[4]) << 16) |
                (static_cast<uint32_t>(block[5]) << 24);
  uint8_t packed = block[6 + (within % 16)];
  uint8_t low = within < 16 ? (packed & 0x0f) : (packed >> 4);
  uint8_t high = static_cast<uint8_t>((qh >> within) & 1);
  uint8_t quant = low | (high << 4);
  output[idx] = d * static_cast<float>(static_cast<int>(quant) - 16);
}

__global__ void dequantize_q5_1_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 24;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float m = __half2float(*reinterpret_cast<const __half*>(block + 2));
  uint32_t qh = static_cast<uint32_t>(block[4]) |
                (static_cast<uint32_t>(block[5]) << 8) |
                (static_cast<uint32_t>(block[6]) << 16) |
                (static_cast<uint32_t>(block[7]) << 24);
  uint8_t packed = block[8 + (within % 16)];
  uint8_t low = within < 16 ? (packed & 0x0f) : (packed >> 4);
  uint8_t high = static_cast<uint8_t>((qh >> within) & 1);
  uint8_t quant = low | (high << 4);
  output[idx] = d * static_cast<float>(quant) + m;
}

__global__ void dequantize_q2_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 84;
  const uint8_t* scales = block;
  const uint8_t* qs = block + 16;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 80));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 82));

  int group16 = within / 16;
  int offset16 = within % 16;
  int half128 = group16 / 8;
  int group_in_half = group16 % 8;
  int pair = group_in_half / 2;
  bool upper16 = (group_in_half % 2) != 0;
  int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
  int shift = 2 * pair;
  uint8_t sc = scales[group16];
  uint8_t quant = (qs[q_index] >> shift) & 0x03;
  output[idx] = d * static_cast<float>(sc & 0x0f) * static_cast<float>(quant) -
                dmin * static_cast<float>(sc >> 4);
}

__global__ void dequantize_q3_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 110;
  const uint8_t* hmask = block;
  const uint8_t* qs = block + 32;
  const uint8_t* scales = block + 96;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 108));

  int group16 = within / 16;
  int offset16 = within % 16;
  int half128 = group16 / 8;
  int group_in_half = group16 % 8;
  int pair = group_in_half / 2;
  bool upper16 = (group_in_half % 2) != 0;
  int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
  int h_index = (upper16 ? 16 : 0) + offset16;
  int shift = 2 * pair;
  uint8_t mask = static_cast<uint8_t>(1u << (4 * half128 + pair));

  int low = static_cast<int>((qs[q_index] >> shift) & 0x03);
  int quant = low - ((hmask[h_index] & mask) != 0 ? 0 : 4);
  output[idx] = d * static_cast<float>(q3_k_scale(group16, scales)) * static_cast<float>(quant);
}

__global__ void dequantize_q4_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 144;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;

  int group64 = within / 64;
  int offset64 = within % 64;
  int scale_index = group64 * 2 + (offset64 >= 32 ? 1 : 0);
  uint8_t scale;
  uint8_t min;
  q4_k_scale_min(scale_index, scales, &scale, &min);

  uint8_t packed = qs[group64 * 32 + (offset64 % 32)];
  uint8_t quant = offset64 < 32 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = d * static_cast<float>(scale) * static_cast<float>(quant) -
                dmin * static_cast<float>(min);
}

__global__ void dequantize_q5_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 176;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;

  int group32 = within / 32;
  int offset32 = within % 32;
  int group64 = group32 / 2;
  uint8_t scale;
  uint8_t min;
  q4_k_scale_min(group32, scales, &scale, &min);

  uint8_t packed = qs[group64 * 32 + offset32];
  uint8_t low = (group32 % 2) == 0 ? (packed & 0x0f) : (packed >> 4);
  uint8_t high = (qh[offset32] & (1u << group32)) != 0 ? 16 : 0;
  uint8_t quant = low + high;
  output[idx] = d * static_cast<float>(scale) * static_cast<float>(quant) -
                dmin * static_cast<float>(min);
}

__global__ void dequantize_q6_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 210;
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const uint8_t* scales = block + 192;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

  int half = within / 128;
  int pos = within % 128;
  int l = pos % 32;
  int group = pos / 32;
  int ql_base = half * 64;
  int qh_base = half * 32;
  int scale_base = half * 8;
  int is = l / 16;
  uint8_t high = qh[qh_base + l];

  int quant = 0;
  int scale_index = scale_base + is;
  switch (group) {
    case 0:
      quant = (ql[ql_base + l] & 0x0f) | (((high >> 0) & 0x03) << 4);
      scale_index += 0;
      break;
    case 1:
      quant = (ql[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4);
      scale_index += 2;
      break;
    case 2:
      quant = (ql[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4);
      scale_index += 4;
      break;
    default:
      quant = (ql[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4);
      scale_index += 6;
      break;
  }
  int8_t scale = static_cast<int8_t>(scales[scale_index]);
  output[idx] = d * static_cast<float>(scale) * static_cast<float>(quant - 32);
}

// f16-output twins of dequantize_q4_k_kernel / dequantize_q6_k_kernel (exact same
// value, wrapped in __float2half) for the fused prefill dequant->f16 path.
__global__ void dequantize_q4_k_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 144;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qs = block + 16;

  int group64 = within / 64;
  int offset64 = within % 64;
  int scale_index = group64 * 2 + (offset64 >= 32 ? 1 : 0);
  uint8_t scale;
  uint8_t min;
  q4_k_scale_min(scale_index, scales, &scale, &min);

  uint8_t packed = qs[group64 * 32 + (offset64 % 32)];
  uint8_t quant = offset64 < 32 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = __float2half(d * static_cast<float>(scale) * static_cast<float>(quant) -
                             dmin * static_cast<float>(min));
}

__global__ void dequantize_q6_k_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 210;
  const uint8_t* ql = block;
  const uint8_t* qh = block + 128;
  const uint8_t* scales = block + 192;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 208));

  int half = within / 128;
  int pos = within % 128;
  int l = pos % 32;
  int group = pos / 32;
  int ql_base = half * 64;
  int qh_base = half * 32;
  int scale_base = half * 8;
  int is = l / 16;
  uint8_t high = qh[qh_base + l];

  int quant = 0;
  int scale_index = scale_base + is;
  switch (group) {
    case 0:
      quant = (ql[ql_base + l] & 0x0f) | (((high >> 0) & 0x03) << 4);
      scale_index += 0;
      break;
    case 1:
      quant = (ql[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4);
      scale_index += 2;
      break;
    case 2:
      quant = (ql[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4);
      scale_index += 4;
      break;
    default:
      quant = (ql[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4);
      scale_index += 6;
      break;
  }
  int8_t scale = static_cast<int8_t>(scales[scale_index]);
  output[idx] = __float2half(d * static_cast<float>(scale) * static_cast<float>(quant - 32));
}

extern "C" int hi_cuda_launch_dequantize_q4_k_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q4_k_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

extern "C" int hi_cuda_launch_dequantize_q6_k_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q6_k_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

// f16-output twins of dequantize_q5_k_kernel / dequantize_iq4_nl_kernel /
// dequantize_q8_0_kernel (exact same value, wrapped in __float2half) for the
// fused prefill dequant->f16 path.
__global__ void dequantize_q5_k_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 176;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
  const uint8_t* scales = block + 4;
  const uint8_t* qh = block + 16;
  const uint8_t* qs = block + 48;

  int group32 = within / 32;
  int offset32 = within % 32;
  int group64 = group32 / 2;
  uint8_t scale;
  uint8_t min;
  q4_k_scale_min(group32, scales, &scale, &min);

  uint8_t packed = qs[group64 * 32 + offset32];
  uint8_t low = (group32 % 2) == 0 ? (packed & 0x0f) : (packed >> 4);
  uint8_t high = (qh[offset32] & (1u << group32)) != 0 ? 16 : 0;
  uint8_t quant = low + high;
  output[idx] = __float2half(d * static_cast<float>(scale) * static_cast<float>(quant) -
                             dmin * static_cast<float>(min));
}

__global__ void dequantize_iq4_nl_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 18;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  uint8_t packed = block[2 + (within % 16)];
  uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
  output[idx] = __float2half(d * static_cast<float>(IQ4_NL_VALUES[quant]));
}

__global__ void dequantize_q8_0_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 32;
  int within = idx % 32;
  const uint8_t* block = input + block_id * 34;
  float d = __half2float(*reinterpret_cast<const __half*>(block));
  int8_t q = static_cast<int8_t>(block[2 + within]);
  output[idx] = __float2half(d * static_cast<float>(q));
}

extern "C" int hi_cuda_launch_dequantize_q5_k_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q5_k_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

extern "C" int hi_cuda_launch_dequantize_iq4_nl_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_iq4_nl_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

extern "C" int hi_cuda_launch_dequantize_q8_0_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q8_0_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

// f16-output twins of dequantize_q2_k_kernel / dequantize_q3_k_kernel (exact same
// value, wrapped in __float2half) for the fused prefill dequant->f16 path.
__global__ void dequantize_q2_k_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 84;
  const uint8_t* scales = block;
  const uint8_t* qs = block + 16;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 80));
  float dmin = __half2float(*reinterpret_cast<const __half*>(block + 82));

  int group16 = within / 16;
  int offset16 = within % 16;
  int half128 = group16 / 8;
  int group_in_half = group16 % 8;
  int pair = group_in_half / 2;
  bool upper16 = (group_in_half % 2) != 0;
  int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
  int shift = 2 * pair;
  uint8_t sc = scales[group16];
  uint8_t quant = (qs[q_index] >> shift) & 0x03;
  output[idx] = __float2half(d * static_cast<float>(sc & 0x0f) * static_cast<float>(quant) -
                             dmin * static_cast<float>(sc >> 4));
}

__global__ void dequantize_q3_k_to_f16_kernel(
    const uint8_t* input, __half* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 110;
  const uint8_t* hmask = block;
  const uint8_t* qs = block + 32;
  const uint8_t* scales = block + 96;
  float d = __half2float(*reinterpret_cast<const __half*>(block + 108));

  int group16 = within / 16;
  int offset16 = within % 16;
  int half128 = group16 / 8;
  int group_in_half = group16 % 8;
  int pair = group_in_half / 2;
  bool upper16 = (group_in_half % 2) != 0;
  int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
  int h_index = (upper16 ? 16 : 0) + offset16;
  int shift = 2 * pair;
  uint8_t mask = static_cast<uint8_t>(1u << (4 * half128 + pair));

  int low = static_cast<int>((qs[q_index] >> shift) & 0x03);
  int quant = low - ((hmask[h_index] & mask) != 0 ? 0 : 4);
  output[idx] = __float2half(
      d * static_cast<float>(q3_k_scale(group16, scales)) * static_cast<float>(quant));
}

extern "C" int hi_cuda_launch_dequantize_q2_k_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q2_k_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

extern "C" int hi_cuda_launch_dequantize_q3_k_to_f16(
    const void* input, void* output, int elements, void* stream) {
  if (input == nullptr || output == nullptr || elements <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  dequantize_q3_k_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), static_cast<__half*>(output), elements);
  return 0;
}

__global__ void dequantize_q8_k_kernel(const uint8_t* input, float* output, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int block_id = idx / 256;
  int within = idx % 256;
  const uint8_t* block = input + block_id * 292;
  float d = *reinterpret_cast<const float*>(block);
  int8_t quant = static_cast<int8_t>(block[4 + within]);
  output[idx] = d * static_cast<float>(quant);
}

__global__ void rope_kernel(
    float* values,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int position_offset,
    int split_half) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_token = heads * (head_dim / 2);
  int total_pairs = seq_len * pairs_per_token;
  if (pair >= total_pairs) {
    return;
  }

  int position = pair / pairs_per_token + position_offset;
  int row = pair / pairs_per_token;
  int within = pair % pairs_per_token;
  int head = within / (head_dim / 2);
  int pair_in_head = within % (head_dim / 2);
  int base_idx = (row * heads + head) * head_dim;
  int left_idx;
  int right_idx;
  if (split_half) {
    left_idx = base_idx + pair_in_head;
    right_idx = base_idx + pair_in_head + head_dim / 2;
  } else {
    left_idx = base_idx + pair_in_head * 2;
    right_idx = left_idx + 1;
  }

  float freq = powf(base, -(static_cast<float>(pair_in_head) * 2.0f) /
                              static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * scale * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float left = values[left_idx];
  float right = values[right_idx];
  values[left_idx] = left * c - right * s;
  values[right_idx] = right * c + left * s;
}

__global__ void rope_batched_kernel(
    float* values,
    int batch_count,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int position_offset,
    int split_half) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_token = heads * (head_dim / 2);
  int total_pairs = batch_count * seq_len * pairs_per_token;
  if (pair >= total_pairs) {
    return;
  }

  int row = pair / pairs_per_token;
  int position = (row % seq_len) + position_offset;
  int within = pair % pairs_per_token;
  int head = within / (head_dim / 2);
  int pair_in_head = within % (head_dim / 2);
  int base_idx = (row * heads + head) * head_dim;
  int left_idx;
  int right_idx;
  if (split_half) {
    left_idx = base_idx + pair_in_head;
    right_idx = base_idx + pair_in_head + head_dim / 2;
  } else {
    left_idx = base_idx + pair_in_head * 2;
    right_idx = left_idx + 1;
  }

  float freq = powf(base, -(static_cast<float>(pair_in_head) * 2.0f) /
                              static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * scale * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float left = values[left_idx];
  float right = values[right_idx];
  values[left_idx] = left * c - right * s;
  values[right_idx] = right * c + left * s;
}

__global__ void rope_batched_positions_kernel(
    float* values,
    const uint32_t* positions,
    int batch_count,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int split_half) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_token = heads * (head_dim / 2);
  int total_pairs = batch_count * seq_len * pairs_per_token;
  if (pair >= total_pairs) {
    return;
  }

  int row = pair / pairs_per_token;
  int batch = row / seq_len;
  int position = static_cast<int>(positions[batch]) + (row % seq_len);
  int within = pair % pairs_per_token;
  int head = within / (head_dim / 2);
  int pair_in_head = within % (head_dim / 2);
  int base_idx = (row * heads + head) * head_dim;
  int left_idx;
  int right_idx;
  if (split_half) {
    left_idx = base_idx + pair_in_head;
    right_idx = base_idx + pair_in_head + head_dim / 2;
  } else {
    left_idx = base_idx + pair_in_head * 2;
    right_idx = left_idx + 1;
  }

  float freq = powf(base, -(static_cast<float>(pair_in_head) * 2.0f) /
                              static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * scale * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float left = values[left_idx];
  float right = values[right_idx];
  values[left_idx] = left * c - right * s;
  values[right_idx] = right * c + left * s;
}

__global__ void mrope_kernel(
    float* values,
    const uint32_t* pos_t,
    const uint32_t* pos_h,
    const uint32_t* pos_w,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int section_t,
    int section_h,
    int section_w,
    int section_e,
    int split_half) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int pairs_per_token = heads * half_dim;
  int total_pairs = seq_len * pairs_per_token;
  if (pair >= total_pairs) {
    return;
  }

  int section_sum = section_t + section_h + section_w + section_e;
  if (section_sum <= 0) {
    return;
  }

  int row = pair / pairs_per_token;
  int within = pair % pairs_per_token;
  int head = within / half_dim;
  int pair_in_head = within % half_dim;
  int sector = pair_in_head % section_sum;
  uint32_t position = pos_t[row];
  int h_start = section_t;
  int w_start = h_start + section_h;
  int e_start = w_start + section_w;
  if (sector >= h_start && sector < w_start) {
    position = pos_h[row];
  } else if (sector >= w_start && sector < e_start) {
    position = pos_w[row];
  }

  int base_idx = (row * heads + head) * head_dim;
  int left_idx;
  int right_idx;
  if (split_half) {
    left_idx = base_idx + pair_in_head;
    right_idx = base_idx + pair_in_head + half_dim;
  } else {
    left_idx = base_idx + pair_in_head * 2;
    right_idx = left_idx + 1;
  }

  float freq = powf(base, -(static_cast<float>(pair_in_head) * 2.0f) /
                              static_cast<float>(head_dim));
  float angle = static_cast<float>(position) * scale * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float left = values[left_idx];
  float right = values[right_idx];
  values[left_idx] = left * c - right * s;
  values[right_idx] = right * c + left * s;
}

__global__ void vision_rope_kernel(
    float* values,
    const uint32_t* pos_h,
    const uint32_t* pos_w,
    int seq_len,
    int heads,
    int head_dim,
    float base) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int axis_dim = half_dim / 2;
  int pairs_per_token = heads * half_dim;
  int total_pairs = seq_len * pairs_per_token;
  if (pair >= total_pairs) {
    return;
  }

  int row = pair / pairs_per_token;
  int within = pair % pairs_per_token;
  int head = within / half_dim;
  int pair_in_head = within % half_dim;
  int axis_pair = pair_in_head < axis_dim ? pair_in_head : pair_in_head - axis_dim;
  uint32_t position = pair_in_head < axis_dim ? pos_h[row] : pos_w[row];
  int base_idx = (row * heads + head) * head_dim;
  int left_idx = base_idx + pair_in_head;
  int right_idx = left_idx + half_dim;

  float freq = powf(base, -(static_cast<float>(axis_pair) * 2.0f) /
                              static_cast<float>(half_dim));
  float angle = static_cast<float>(position) * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float left = values[left_idx];
  float right = values[right_idx];
  values[left_idx] = left * c - right * s;
  values[right_idx] = right * c + left * s;
}


__global__ void write_kv_cache_kernel(
    const float* values,
    float* cache,
    int row_count,
    int kv_heads,
    int head_dim,
    int max_seq,
    int start_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = row_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = idx / (head_dim * kv_heads);
  int cache_pos = start_pos + row;
  cache[(head * max_seq + cache_pos) * head_dim + dim] =
      values[(row * kv_heads + head) * head_dim + dim];
}

__global__ void write_kv_cache_batched_kernel(
    const float* values,
    float* cache,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int max_seq,
    int start_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * row_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = (idx / (head_dim * kv_heads)) % row_count;
  int batch = idx / (row_count * kv_heads * head_dim);
  int cache_pos = start_pos + row;
  int cache_idx = ((batch * kv_heads + head) * max_seq + cache_pos) * head_dim + dim;
  int value_idx = ((batch * row_count + row) * kv_heads + head) * head_dim + dim;
  cache[cache_idx] = values[value_idx];
}

__global__ void write_paged_kv_cache_kernel(
    const float* values,
    kv_t* pages,
    const uint32_t* page_table,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    int start_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = row_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = idx / (head_dim * kv_heads);
  int logical_pos = start_pos + row;
  int logical_page = logical_pos / page_size;
  int page_offset = logical_pos - logical_page * page_size;
  if (logical_page >= page_table_len) {
    return;
  }
  uint32_t physical_page = page_table[logical_page];
  int page_idx =
      ((static_cast<int>(physical_page) * kv_heads + head) * page_size + page_offset) *
          head_dim +
      dim;
  kv_from_float(&pages[page_idx], values[(row * kv_heads + head) * head_dim + dim]);
}

__global__ void write_paged_kv_cache_batched_kernel(
    const float* values,
    kv_t* pages,
    const uint32_t* page_table,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    int start_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * row_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = (idx / (head_dim * kv_heads)) % row_count;
  int batch = idx / (row_count * kv_heads * head_dim);
  int logical_pos = start_pos + row;
  int logical_page = logical_pos / page_size;
  int page_offset = logical_pos - logical_page * page_size;
  if (logical_page >= page_table_len) {
    return;
  }
  uint32_t physical_page = page_table[batch * page_table_len + logical_page];
  int page_idx =
      ((static_cast<int>(physical_page) * kv_heads + head) * page_size + page_offset) *
          head_dim +
      dim;
  int value_idx = ((batch * row_count + row) * kv_heads + head) * head_dim + dim;
  kv_from_float(&pages[page_idx], values[value_idx]);
}

__global__ void write_paged_kv_cache_batched_positions_kernel(
    const float* values,
    kv_t* pages,
    const uint32_t* page_table,
    const uint32_t* positions,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * row_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = (idx / (head_dim * kv_heads)) % row_count;
  int batch = idx / (row_count * kv_heads * head_dim);
  int logical_pos = static_cast<int>(positions[batch]) + row;
  int logical_page = logical_pos / page_size;
  int page_offset = logical_pos - logical_page * page_size;
  if (logical_pos < 0 || logical_page >= page_table_len) {
    return;
  }
  uint32_t physical_page = page_table[batch * page_table_len + logical_page];
  int page_idx =
      ((static_cast<int>(physical_page) * kv_heads + head) * page_size + page_offset) *
          head_dim +
      dim;
  int value_idx = ((batch * row_count + row) * kv_heads + head) * head_dim + dim;
  kv_from_float(&pages[page_idx], values[value_idx]);
}

__global__ void copy_paged_kv_cache_prefix_batched_kernel(
    kv_t* pages,
    const uint32_t* page_table,
    int batch_count,
    int token_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int dst_batches = batch_count - 1;
  int total = dst_batches * token_count * kv_heads * head_dim;
  if (idx >= total) {
    return;
  }
  int dim = idx % head_dim;
  int head = (idx / head_dim) % kv_heads;
  int row = (idx / (head_dim * kv_heads)) % token_count;
  int dst_batch = 1 + idx / (token_count * kv_heads * head_dim);
  int logical_page = row / page_size;
  int page_offset = row - logical_page * page_size;
  if (logical_page >= page_table_len) {
    return;
  }
  uint32_t src_page = page_table[logical_page];
  uint32_t dst_page = page_table[dst_batch * page_table_len + logical_page];
  int src_idx =
      ((static_cast<int>(src_page) * kv_heads + head) * page_size + page_offset) *
          head_dim +
      dim;
  int dst_idx =
      ((static_cast<int>(dst_page) * kv_heads + head) * page_size + page_offset) *
          head_dim +
      dim;
  pages[dst_idx] = pages[src_idx];
}

__global__ void causal_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * heads;
  if (idx >= total) {
    return;
  }

  int target = idx / heads;
  int head = idx % heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (target * heads + head) * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float score = dot * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + (target * heads + head) * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v + (source * kv_heads + kv_head) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

__global__ void causal_attention_batched_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * seq_len * heads;
  if (idx >= total) {
    return;
  }

  int head = idx % heads;
  int target_global = idx / heads;
  int target = target_global % seq_len;
  int batch = target_global / seq_len;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + ((batch * seq_len + target) * heads + head) * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + ((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float score = dot * scale;
    max_score = fmaxf(max_score, score);
  }

  float denom = 0.0f;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + ((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + ((batch * seq_len + target) * heads + head) * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + ((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v + ((batch * seq_len + source) * kv_heads + kv_head) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

__global__ void full_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int seq_len,
    int heads,
    int kv_heads,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * heads;
  if (idx >= total) {
    return;
  }

  int target = idx / heads;
  int head = idx % heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float* q_vec = q + (target * heads + head) * head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source < seq_len; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    max_score = fmaxf(max_score, dot * scale);
  }

  float denom = 0.0f;
  for (int source = 0; source < seq_len; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + (target * heads + head) * head_dim;
  for (int dim = 0; dim < head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = 0; source < seq_len; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v + (source * kv_heads + kv_head) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

__global__ void window_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    const uint32_t* window_start,
    const uint32_t* window_end,
    float* output,
    int seq_len,
    int heads,
    int kv_heads,
    int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * heads;
  if (idx >= total) {
    return;
  }

  int target = idx / heads;
  int head = idx % heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  int start = static_cast<int>(window_start[target]);
  int end = static_cast<int>(window_end[target]);
  start = max(0, min(start, seq_len));
  end = max(start, min(end, seq_len));
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float* q_vec = q + (target * heads + head) * head_dim;

  float max_score = -INFINITY;
  for (int source = start; source < end; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    max_score = fmaxf(max_score, dot * scale);
  }

  float denom = 0.0f;
  for (int source = start; source < end; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + (target * heads + head) * head_dim;
  for (int dim = 0; dim < head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = start; source < end; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v + (source * kv_heads + kv_head) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

__global__ void cached_decode_attention_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* output,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head >= heads) {
    return;
  }

  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + head * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + (kv_head * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    max_score = fmaxf(max_score, dot * scale);
  }

  float denom = 0.0f;
  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + (kv_head * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + head * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + (kv_head * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v_cache + (kv_head * max_seq + source) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

__global__ void cached_decode_attention_batched_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* output,
    int batch_count,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * heads;
  if (idx >= total) {
    return;
  }

  int head = idx % heads;
  int batch = idx / heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + ((batch * kv_heads + kv_head) * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    max_score = fmaxf(max_score, dot * scale);
  }

  float denom = 0.0f;
  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + ((batch * kv_heads + kv_head) * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    denom += expf(dot * scale - max_score);
  }

  float* out_vec = output + (batch * heads + head) * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    return;
  }

  for (int source = 0; source <= position; ++source) {
    const float* k_vec = k_cache + ((batch * kv_heads + kv_head) * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float weight = expf(dot * scale - max_score) / denom;
    const float* v_vec = v_cache + ((batch * kv_heads + kv_head) * max_seq + source) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * v_vec[dim];
    }
  }
}

constexpr int HI_CUDA_FLASH_MAX_HEAD_DIM = 512;

// Flash-attention (shared-memory K/V tiling) prefill: QWARPS queries per block share
// each loaded K/V tile, cutting global K/V reads by QWARPS vs the per-query kernel.
// Capped to this head_dim so a KTILE fits in <48 KB shared (KTILE*(qk+v)*4 bytes).
constexpr int HI_CUDA_FLASH_QWARPS = 8;
constexpr int HI_CUDA_FLASH_KTILE = 32;
constexpr int HI_CUDA_FLASH_TILE_MAX_HEAD_DIM = 128;

// Warps per (head) block in the split-K flash-decode kernels. Decode has one
// query per head, so a single-warp serial loop over the KV cache is latency-bound
// on the paged reads; splitting the key range across this many warps issues that
// many concurrent K/V loads to hide the latency, then a shared-memory flash
// rescale-merge combines the per-warp partial softmaxes.
constexpr int HI_CUDA_DECODE_WARPS = 16;

__global__ void flash_causal_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  int target = idx / heads;
  int head = idx % heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (target * heads + head) * qk_head_dim;
  float* out_vec = output + (target * heads + head) * v_head_dim;

  float acc[HI_CUDA_FLASH_MAX_HEAD_DIM];
  for (int dim = 0; dim < v_head_dim; ++dim) {
    acc[dim] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    float weight = expf(score - next_max);
    const float* v_vec = v + (source * kv_heads + kv_head) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      acc[dim] = acc[dim] * old_scale + weight * v_vec[dim];
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = acc[dim] / denom;
  }
}

__global__ void tiled_causal_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window) {
  // One warp (32 threads) per (query, head). The q*k dot is reduced with warp
  // shuffles (no shared memory / __syncthreads), q is cached in registers, and
  // the online-softmax output is kept in a per-lane register array. This replaces
  // a 512-thread block that ran a log2-depth shared-memory tree reduction *per
  // key*, i.e. ~O(seq^2) __syncthreads that dominated prefill.
  const int idx = blockIdx.x;
  const int lane = threadIdx.x;  // 0..31
  const int total = seq_len * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  const int target = idx / heads;
  const int head = idx % heads;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (target * heads + head) * qk_head_dim;
  float* out_vec = output + (target * heads + head) * v_head_dim;

  // Each lane owns dims lane, lane+32, ... of the head. Cache q once.
  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;

  // Sliding-window (Gemma-3 local layers): only attend to the last `window` keys.
  // window <= 0 means unlimited causal attention.
  const int source_start = (window > 0 && target >= window) ? target - window + 1 : 0;
  for (int source = source_start; source <= target; ++source) {
    const float* k_vec = k + (source * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < qk_head_dim) {
        dot += q_reg[i] * k_vec[dim];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_max = fmaxf(max_score, score);
    const float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    const float weight = expf(score - next_max);
    const float* v_vec = v + (source * kv_heads + kv_head) * v_head_dim;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < v_head_dim) {
        acc[i] = acc[i] * old_scale + weight * v_vec[dim];
      }
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }

  const float inv = (denom == 0.0f || !isfinite(denom)) ? 0.0f : 1.0f / denom;
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      out_vec[dim] = acc[i] * inv;
    }
  }
}

__global__ void flash_causal_attention_batched_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * seq_len * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  int head = idx % heads;
  int target_global = idx / heads;
  int target = target_global % seq_len;
  int batch = target_global / seq_len;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + ((batch * seq_len + target) * heads + head) * qk_head_dim;
  float* out_vec = output + ((batch * seq_len + target) * heads + head) * v_head_dim;

  float acc[HI_CUDA_FLASH_MAX_HEAD_DIM];
  for (int dim = 0; dim < v_head_dim; ++dim) {
    acc[dim] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int source = 0; source <= target; ++source) {
    const float* k_vec = k + ((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * k_vec[dim];
    }
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    float weight = expf(score - next_max);
    const float* v_vec = v + ((batch * seq_len + source) * kv_heads + kv_head) * v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      acc[dim] = acc[dim] * old_scale + weight * v_vec[dim];
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = acc[dim] / denom;
  }
}

__global__ void tiled_causal_attention_batched_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window) {
  // One warp (32 threads) per (batch, query, head); warp-shuffle dot reduction,
  // q cached in registers, register-array online softmax — no shared memory or
  // per-key __syncthreads. See tiled_causal_attention_kernel for the rationale.
  const int idx = blockIdx.x;
  const int lane = threadIdx.x;  // 0..31
  const int total = batch_count * seq_len * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  const int head = idx % heads;
  const int target_global = idx / heads;
  const int target = target_global % seq_len;
  const int batch = target_global / seq_len;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + ((batch * seq_len + target) * heads + head) * qk_head_dim;
  float* out_vec = output + ((batch * seq_len + target) * heads + head) * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;

  const int source_start = (window > 0 && target >= window) ? target - window + 1 : 0;
  for (int source = source_start; source <= target; ++source) {
    const float* k_vec = k + ((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < qk_head_dim) {
        dot += q_reg[i] * k_vec[dim];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_max = fmaxf(max_score, score);
    const float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    const float weight = expf(score - next_max);
    const float* v_vec = v + ((batch * seq_len + source) * kv_heads + kv_head) * v_head_dim;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < v_head_dim) {
        acc[i] = acc[i] * old_scale + weight * v_vec[dim];
      }
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }

  const float inv = (denom == 0.0f || !isfinite(denom)) ? 0.0f : 1.0f / denom;
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      out_vec[dim] = acc[i] * inv;
    }
  }
}

__global__ void flash_cached_decode_attention_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* output,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  // Split-K flash decode over the contiguous (legacy) KV cache. HI_CUDA_DECODE_WARPS
  // warps per head each fold a strided slice of the cache into a local online
  // softmax (warp-shuffle dot reduction, q cached in registers), then a shared-memory
  // flash rescale-merge. The concurrent per-warp reads hide the cache read latency a
  // single serial thread cannot. Mirrors tiled_paged_decode_attention_kernel (F32
  // contiguous cache, so no page table / kv_to_float / sliding window).
  const int head = blockIdx.x;
  if (head >= heads || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + head * qk_head_dim;
  float* out_vec = output + head * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  for (int source = warp_id; source <= position; source += HI_CUDA_DECODE_WARPS) {
    const float* k_vec = k_cache + (kv_head * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < qk_head_dim) {
        dot += q_reg[i] * k_vec[dim];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_m = fmaxf(m, score);
    const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
    const float weight = expf(score - next_m);
    const float* v_vec = v_cache + (kv_head * max_seq + source) * v_head_dim;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < v_head_dim) {
        acc[i] = acc[i] * rescale + weight * v_vec[dim];
      }
    }
    l = l * rescale + weight;
    m = next_m;
  }

  extern __shared__ float smem[];
  float* sh_m = smem;
  float* sh_l = smem + HI_CUDA_DECODE_WARPS;
  float* sh_acc = smem + 2 * HI_CUDA_DECODE_WARPS;
  if (lane == 0) {
    sh_m[warp_id] = m;
    sh_l[warp_id] = l;
  }
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      sh_acc[warp_id * v_head_dim + dim] = acc[i];
    }
  }
  __syncthreads();

  float global_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    global_max = fmaxf(global_max, sh_m[w]);
  }
  float total_denom = 0.0f;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    total_denom += sh_l[w] * expf(sh_m[w] - global_max);
  }
  const float inv =
      (total_denom == 0.0f || !isfinite(total_denom)) ? 0.0f : 1.0f / total_denom;
  for (int dim = threadIdx.x; dim < v_head_dim; dim += blockDim.x) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
      a += sh_acc[w * v_head_dim + dim] * expf(sh_m[w] - global_max);
    }
    out_vec[dim] = a * inv;
  }
}

__global__ void paged_decode_attention_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads) {
    return;
  }

  int head = idx;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + head * qk_head_dim;
  float* out_vec = output + head * v_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    max_score = fmaxf(max_score, dot * scale);
  }
  float denom = 0.0f;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    float weight = expf(dot * scale - max_score);
    const kv_t* v_vec =
        v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * kv_to_float(v_vec[dim]);
    }
    denom += weight;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] /= denom;
  }
}

__global__ void flash_paged_decode_attention_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  int head = idx;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + head * qk_head_dim;
  float* out_vec = output + head * v_head_dim;

  float acc[HI_CUDA_FLASH_MAX_HEAD_DIM];
  for (int dim = 0; dim < v_head_dim; ++dim) {
    acc[dim] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    float weight = expf(score - next_max);
    const kv_t* v_vec =
        v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      acc[dim] = acc[dim] * old_scale + weight * kv_to_float(v_vec[dim]);
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = acc[dim] / denom;
  }
}

__global__ void tiled_paged_decode_attention_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window) {
  // Split-K flash decode: HI_CUDA_DECODE_WARPS warps per head, each folding a
  // strided slice of the KV cache into a local online softmax (warp-shuffle dot
  // reduction, q cached in registers), then a shared-memory flash rescale-merge.
  // The concurrent per-warp K/V reads hide the paged-read latency that a single
  // serial warp cannot.
  const int head = blockIdx.x;
  if (head >= heads || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + head * qk_head_dim;
  float* out_vec = output + head * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;  // running max for this warp's key slice
  float l = 0.0f;       // running denom

  // Sliding-window (Gemma-3 local layers): only attend to the last `window`
  // positions. window <= 0 means unlimited causal attention.
  const int source_start = (window > 0 && position >= window) ? position - window + 1 : 0;
  for (int source = source_start + warp_id; source <= position;
       source += HI_CUDA_DECODE_WARPS) {
    const int logical_page = source / page_size;
    const int page_offset = source - logical_page * page_size;
    float dot = 0.0f;
    const kv_t* v_vec = nullptr;
    if (logical_page < page_table_len) {
      const uint32_t physical_page = page_table[logical_page];
      const kv_t* k_vec =
          k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * qk_head_dim;
      v_vec =
          v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * v_head_dim;
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < qk_head_dim) {
          dot += q_reg[i] * kv_to_float(k_vec[dim]);
        }
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_m = fmaxf(m, score);
    const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
    const float weight = expf(score - next_m);
    if (v_vec != nullptr) {
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < v_head_dim) {
          acc[i] = acc[i] * rescale + weight * kv_to_float(v_vec[dim]);
        }
      }
    }
    l = l * rescale + weight;
    m = next_m;
  }

  // Flash rescale-merge across the warps' partial softmaxes.
  // Shared layout: [DECODE_WARPS m][DECODE_WARPS l][DECODE_WARPS * v_head_dim acc].
  extern __shared__ float smem[];
  float* sh_m = smem;
  float* sh_l = smem + HI_CUDA_DECODE_WARPS;
  float* sh_acc = smem + 2 * HI_CUDA_DECODE_WARPS;
  if (lane == 0) {
    sh_m[warp_id] = m;
    sh_l[warp_id] = l;
  }
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      sh_acc[warp_id * v_head_dim + dim] = acc[i];
    }
  }
  __syncthreads();

  float global_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    global_max = fmaxf(global_max, sh_m[w]);
  }
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    total += sh_l[w] * expf(sh_m[w] - global_max);
  }
  const float inv = (total == 0.0f || !isfinite(total)) ? 0.0f : 1.0f / total;
  for (int dim = threadIdx.x; dim < v_head_dim; dim += blockDim.x) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
      a += sh_acc[w * v_head_dim + dim] * expf(sh_m[w] - global_max);
    }
    out_vec[dim] = a * inv;
  }
}

__global__ void paged_decode_attention_batched_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * heads;
  if (idx >= total) {
    return;
  }

  int head = idx % heads;
  int batch = idx / heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[batch * page_table_len + logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    max_score = fmaxf(max_score, dot * scale);
  }
  float denom = 0.0f;
  float* out_vec = output + (batch * heads + head) * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[batch * page_table_len + logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    float weight = expf(dot * scale - max_score);
    const kv_t* v_vec =
        v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * kv_to_float(v_vec[dim]);
    }
    denom += weight;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] /= denom;
  }
}

__global__ void paged_decode_attention_batched_positions_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    const uint32_t* positions,
    float* output,
    int batch_count,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * heads;
  if (idx >= total) {
    return;
  }

  int head = idx % heads;
  int batch = idx / heads;
  int position = static_cast<int>(positions[batch]);
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;

  float max_score = -INFINITY;
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[batch * page_table_len + logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    max_score = fmaxf(max_score, dot * scale);
  }
  float denom = 0.0f;
  float* out_vec = output + (batch * heads + head) * v_head_dim;
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = 0.0f;
  }
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[batch * page_table_len + logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    float weight = expf(dot * scale - max_score);
    const kv_t* v_vec =
        v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] += weight * kv_to_float(v_vec[dim]);
    }
    denom += weight;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] /= denom;
  }
}

__global__ void flash_paged_decode_attention_batched_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_count * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }

  int head = idx % heads;
  int batch = idx / heads;
  int kv_repeats = heads / kv_heads;
  int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;
  float* out_vec = output + (batch * heads + head) * v_head_dim;

  float acc[HI_CUDA_FLASH_MAX_HEAD_DIM];
  for (int dim = 0; dim < v_head_dim; ++dim) {
    acc[dim] = 0.0f;
  }
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int source = 0; source <= position; ++source) {
    int logical_page = source / page_size;
    int page_offset = source - logical_page * page_size;
    if (logical_page >= page_table_len) {
      continue;
    }
    uint32_t physical_page = page_table[batch * page_table_len + logical_page];
    const kv_t* k_vec =
        k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      qk_head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < qk_head_dim; ++dim) {
      dot += q_vec[dim] * kv_to_float(k_vec[dim]);
    }
    float score = dot * scale;
    float next_max = fmaxf(max_score, score);
    float old_scale = isfinite(max_score) ? expf(max_score - next_max) : 0.0f;
    float weight = expf(score - next_max);
    const kv_t* v_vec =
        v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                   page_offset) *
                      v_head_dim;
    for (int dim = 0; dim < v_head_dim; ++dim) {
      acc[dim] = acc[dim] * old_scale + weight * kv_to_float(v_vec[dim]);
    }
    denom = denom * old_scale + weight;
    max_score = next_max;
  }
  if (denom == 0.0f || !isfinite(denom)) {
    for (int dim = 0; dim < v_head_dim; ++dim) {
      out_vec[dim] = 0.0f;
    }
    return;
  }
  for (int dim = 0; dim < v_head_dim; ++dim) {
    out_vec[dim] = acc[dim] / denom;
  }
}

__global__ void tiled_paged_decode_attention_batched_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    float* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window) {
  // Split-K flash decode (batched). See tiled_paged_decode_attention_kernel.
  const int idx = blockIdx.x;
  const int total = batch_count * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = idx % heads;
  const int batch = idx / heads;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;
  float* out_vec = output + (batch * heads + head) * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  const int source_start = (window > 0 && position >= window) ? position - window + 1 : 0;
  for (int source = source_start + warp_id; source <= position;
       source += HI_CUDA_DECODE_WARPS) {
    const int logical_page = source / page_size;
    const int page_offset = source - logical_page * page_size;
    float dot = 0.0f;
    const kv_t* v_vec = nullptr;
    if (logical_page < page_table_len) {
      const uint32_t physical_page = page_table[batch * page_table_len + logical_page];
      const kv_t* k_vec =
          k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * qk_head_dim;
      v_vec =
          v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * v_head_dim;
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < qk_head_dim) {
          dot += q_reg[i] * kv_to_float(k_vec[dim]);
        }
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_m = fmaxf(m, score);
    const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
    const float weight = expf(score - next_m);
    if (v_vec != nullptr) {
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < v_head_dim) {
          acc[i] = acc[i] * rescale + weight * kv_to_float(v_vec[dim]);
        }
      }
    }
    l = l * rescale + weight;
    m = next_m;
  }

  extern __shared__ float smem[];
  float* sh_m = smem;
  float* sh_l = smem + HI_CUDA_DECODE_WARPS;
  float* sh_acc = smem + 2 * HI_CUDA_DECODE_WARPS;
  if (lane == 0) {
    sh_m[warp_id] = m;
    sh_l[warp_id] = l;
  }
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      sh_acc[warp_id * v_head_dim + dim] = acc[i];
    }
  }
  __syncthreads();

  float global_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    global_max = fmaxf(global_max, sh_m[w]);
  }
  float total_denom = 0.0f;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    total_denom += sh_l[w] * expf(sh_m[w] - global_max);
  }
  const float inv =
      (total_denom == 0.0f || !isfinite(total_denom)) ? 0.0f : 1.0f / total_denom;
  for (int dim = threadIdx.x; dim < v_head_dim; dim += blockDim.x) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
      a += sh_acc[w * v_head_dim + dim] * expf(sh_m[w] - global_max);
    }
    out_vec[dim] = a * inv;
  }
}

__global__ void tiled_paged_decode_attention_batched_positions_kernel(
    const float* q,
    const kv_t* k_pages,
    const kv_t* v_pages,
    const uint32_t* page_table,
    const uint32_t* positions,
    float* output,
    int batch_count,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window) {
  // Split-K flash decode with per-batch position. See tiled_paged_decode_attention_kernel.
  const int idx = blockIdx.x;
  const int total = batch_count * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = idx % heads;
  const int batch = idx / heads;
  const int position = static_cast<int>(positions[batch]);
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;
  float* out_vec = output + (batch * heads + head) * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  const int source_start = (window > 0 && position >= window) ? position - window + 1 : 0;
  for (int source = source_start + warp_id; source <= position;
       source += HI_CUDA_DECODE_WARPS) {
    const int logical_page = source / page_size;
    const int page_offset = source - logical_page * page_size;
    float dot = 0.0f;
    const kv_t* v_vec = nullptr;
    if (logical_page < page_table_len) {
      const uint32_t physical_page = page_table[batch * page_table_len + logical_page];
      const kv_t* k_vec =
          k_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * qk_head_dim;
      v_vec =
          v_pages + ((static_cast<int>(physical_page) * kv_heads + kv_head) * page_size +
                     page_offset) * v_head_dim;
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < qk_head_dim) {
          dot += q_reg[i] * kv_to_float(k_vec[dim]);
        }
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_m = fmaxf(m, score);
    const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
    const float weight = expf(score - next_m);
    if (v_vec != nullptr) {
#pragma unroll
      for (int i = 0; i < MAX_PER_LANE; ++i) {
        const int dim = lane + i * 32;
        if (dim < v_head_dim) {
          acc[i] = acc[i] * rescale + weight * kv_to_float(v_vec[dim]);
        }
      }
    }
    l = l * rescale + weight;
    m = next_m;
  }

  extern __shared__ float smem[];
  float* sh_m = smem;
  float* sh_l = smem + HI_CUDA_DECODE_WARPS;
  float* sh_acc = smem + 2 * HI_CUDA_DECODE_WARPS;
  if (lane == 0) {
    sh_m[warp_id] = m;
    sh_l[warp_id] = l;
  }
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      sh_acc[warp_id * v_head_dim + dim] = acc[i];
    }
  }
  __syncthreads();

  float global_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    global_max = fmaxf(global_max, sh_m[w]);
  }
  float total_denom = 0.0f;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    total_denom += sh_l[w] * expf(sh_m[w] - global_max);
  }
  const float inv =
      (total_denom == 0.0f || !isfinite(total_denom)) ? 0.0f : 1.0f / total_denom;
  for (int dim = threadIdx.x; dim < v_head_dim; dim += blockDim.x) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
      a += sh_acc[w * v_head_dim + dim] * expf(sh_m[w] - global_max);
    }
    out_vec[dim] = a * inv;
  }
}

__global__ void flash_cached_decode_attention_batched_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* output,
    int batch_count,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim) {
  // Split-K flash decode (batched, legacy contiguous cache).
  // See flash_cached_decode_attention_kernel.
  const int idx = blockIdx.x;
  const int total = batch_count * heads;
  if (idx >= total || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM) {
    return;
  }
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = idx % heads;
  const int batch = idx / heads;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const float* q_vec = q + (batch * heads + head) * qk_head_dim;
  float* out_vec = output + (batch * heads + head) * v_head_dim;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    q_reg[i] = (dim < qk_head_dim) ? q_vec[dim] : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  for (int source = warp_id; source <= position; source += HI_CUDA_DECODE_WARPS) {
    const float* k_vec =
        k_cache + ((batch * kv_heads + kv_head) * max_seq + source) * qk_head_dim;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < qk_head_dim) {
        dot += q_reg[i] * k_vec[dim];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;

    const float next_m = fmaxf(m, score);
    const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
    const float weight = expf(score - next_m);
    const float* v_vec =
        v_cache + ((batch * kv_heads + kv_head) * max_seq + source) * v_head_dim;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int dim = lane + i * 32;
      if (dim < v_head_dim) {
        acc[i] = acc[i] * rescale + weight * v_vec[dim];
      }
    }
    l = l * rescale + weight;
    m = next_m;
  }

  extern __shared__ float smem[];
  float* sh_m = smem;
  float* sh_l = smem + HI_CUDA_DECODE_WARPS;
  float* sh_acc = smem + 2 * HI_CUDA_DECODE_WARPS;
  if (lane == 0) {
    sh_m[warp_id] = m;
    sh_l[warp_id] = l;
  }
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int dim = lane + i * 32;
    if (dim < v_head_dim) {
      sh_acc[warp_id * v_head_dim + dim] = acc[i];
    }
  }
  __syncthreads();

  float global_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    global_max = fmaxf(global_max, sh_m[w]);
  }
  float total_denom = 0.0f;
#pragma unroll
  for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
    total_denom += sh_l[w] * expf(sh_m[w] - global_max);
  }
  const float inv =
      (total_denom == 0.0f || !isfinite(total_denom)) ? 0.0f : 1.0f / total_denom;
  for (int dim = threadIdx.x; dim < v_head_dim; dim += blockDim.x) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < HI_CUDA_DECODE_WARPS; ++w) {
      a += sh_acc[w * v_head_dim + dim] * expf(sh_m[w] - global_max);
    }
    out_vec[dim] = a * inv;
  }
}

// Parallel last-token argmax: one 256-thread block cooperatively scans `n` logits
// with a shared-memory value+index reduction (ties resolve to the lowest index,
// matching the serial reference). Replaces a single-thread scan over the vocab.
__device__ void block_argmax_last(const float* row, int n, uint32_t* out) {
  __shared__ float sh_val[256];
  __shared__ int sh_idx[256];
  const int t = threadIdx.x;
  float best = -INFINITY;
  int best_idx = 0;
  for (int i = t; i < n; i += blockDim.x) {
    const float v = row[i];
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }
  sh_val[t] = best;
  sh_idx[t] = best_idx;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const float ov = sh_val[t + stride];
      const int oi = sh_idx[t + stride];
      if (ov > sh_val[t] || (ov == sh_val[t] && oi < sh_idx[t])) {
        sh_val[t] = ov;
        sh_idx[t] = oi;
      }
    }
    __syncthreads();
  }
  if (t == 0) {
    *out = static_cast<uint32_t>(sh_idx[0]);
  }
}

__global__ void argmax_kernel(
    const float* logits,
    uint32_t* output_token,
    int len) {
  if (len <= 0) {
    return;
  }
  block_argmax_last(logits, len, output_token);
}

__global__ void argmax_last_row_kernel(
    const float* logits,
    uint32_t* output_token,
    int rows,
    int cols) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  block_argmax_last(logits + (rows - 1) * cols, cols, output_token);
}

__global__ void argmax_batched_last_token_kernel(
    const float* logits,
    uint32_t* output_tokens,
    int batch_count,
    int seq_len,
    int cols) {
  const int batch = blockIdx.x;
  if (batch >= batch_count || seq_len <= 0 || cols <= 0) {
    return;
  }
  block_argmax_last(logits + (batch * seq_len + (seq_len - 1)) * cols, cols,
                    &output_tokens[batch]);
}

__device__ uint32_t argmax_row_device(const float* row, int cols) {
  int best_idx = 0;
  float best = row[0];
  for (int idx = 1; idx < cols; ++idx) {
    float value = row[idx];
    if (value > best) {
      best = value;
      best_idx = idx;
    }
  }
  return static_cast<uint32_t>(best_idx);
}

__device__ float sampling_weight(
    const float* row,
    int idx,
    float temperature,
    float max_scaled) {
  float logit = row[idx];
  if (!isfinite(logit)) {
    return 0.0f;
  }
  float scaled = logit / temperature;
  if (!isfinite(scaled)) {
    return 0.0f;
  }
  return expf(scaled - max_scaled);
}

__device__ int next_ranked_token(
    const float* row,
    int cols,
    float temperature,
    float max_scaled,
    float previous_weight,
    int previous_id,
    float* selected_weight) {
  int best_id = -1;
  float best_weight = -1.0f;
  for (int idx = 0; idx < cols; ++idx) {
    float weight = sampling_weight(row, idx, temperature, max_scaled);
    bool eligible = previous_id < 0 || weight < previous_weight ||
                    (weight == previous_weight && idx > previous_id);
    if (!eligible) {
      continue;
    }
    if (best_id < 0 || weight > best_weight ||
        (weight == best_weight && idx < best_id)) {
      best_id = idx;
      best_weight = weight;
    }
  }
  *selected_weight = best_weight;
  return best_id;
}

__global__ void sample_last_row_kernel(
    const float* logits,
    uint32_t* output_token,
    int rows,
    int cols,
    float temperature,
    float top_p,
    int top_k,
    float sample) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || rows <= 0 || cols <= 0) {
    return;
  }
  const float* row = logits + (rows - 1) * cols;
  if (!isfinite(temperature) || temperature <= 0.0f) {
    *output_token = argmax_row_device(row, cols);
    return;
  }

  float max_scaled = -INFINITY;
  for (int idx = 0; idx < cols; ++idx) {
    float logit = row[idx];
    if (!isfinite(logit)) {
      continue;
    }
    float scaled = logit / temperature;
    if (isfinite(scaled) && scaled > max_scaled) {
      max_scaled = scaled;
    }
  }
  if (!isfinite(max_scaled)) {
    *output_token = argmax_row_device(row, cols);
    return;
  }

  float cutoff = isfinite(top_p) ? fminf(fmaxf(top_p, 0.0f), 1.0f) : 1.0f;
  float uniform = isfinite(sample) ? fminf(fmaxf(sample, 0.0f), 0.99999994f) : 0.0f;
  int effective_top_k = top_k > 0 && top_k < cols ? top_k : cols;

  if (effective_top_k == cols && cutoff >= 1.0f) {
    float total = 0.0f;
    for (int idx = 0; idx < cols; ++idx) {
      total += sampling_weight(row, idx, temperature, max_scaled);
    }
    if (total <= 0.0f || !isfinite(total)) {
      *output_token = argmax_row_device(row, cols);
      return;
    }
    float target = uniform * total;
    float cumulative = 0.0f;
    int last_positive = -1;
    for (int idx = 0; idx < cols; ++idx) {
      float weight = sampling_weight(row, idx, temperature, max_scaled);
      if (weight <= 0.0f) {
        continue;
      }
      last_positive = idx;
      cumulative += weight;
      if (target < cumulative) {
        *output_token = static_cast<uint32_t>(idx);
        return;
      }
    }
    *output_token = static_cast<uint32_t>(last_positive >= 0 ? last_positive : 0);
    return;
  }

  float total = 0.0f;
  float previous_weight = INFINITY;
  int previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    total += weight;
    previous_weight = weight;
    previous_id = token;
  }
  if (total <= 0.0f || !isfinite(total)) {
    *output_token = argmax_row_device(row, cols);
    return;
  }

  float candidate_weight_total = 0.0f;
  float candidate_probability = 0.0f;
  int candidate_count = 0;
  previous_weight = INFINITY;
  previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    previous_weight = weight;
    previous_id = token;
    if (weight <= 0.0f) {
      continue;
    }
    candidate_weight_total += weight;
    candidate_probability += weight / total;
    ++candidate_count;
    if (cutoff < 1.0f && candidate_probability >= cutoff) {
      break;
    }
  }
  if (candidate_count <= 0 || candidate_weight_total <= 0.0f ||
      !isfinite(candidate_weight_total)) {
    *output_token = argmax_row_device(row, cols);
    return;
  }

  float target = uniform * candidate_weight_total;
  float cumulative = 0.0f;
  int emitted = 0;
  previous_weight = INFINITY;
  previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    previous_weight = weight;
    previous_id = token;
    if (weight <= 0.0f) {
      continue;
    }
    ++emitted;
    cumulative += weight;
    if (target < cumulative || emitted == candidate_count) {
      *output_token = static_cast<uint32_t>(token);
      return;
    }
    if (emitted >= candidate_count) {
      break;
    }
  }

  *output_token = argmax_row_device(row, cols);
}

__global__ void sample_batched_last_token_kernel(
    const float* logits,
    uint32_t* output_tokens,
    const float* samples,
    int batch_count,
    int seq_len,
    int cols,
    float temperature,
    float top_p,
    int top_k) {
  int batch = blockIdx.x;
  if (threadIdx.x != 0 || batch >= batch_count || seq_len <= 0 || cols <= 0) {
    return;
  }
  const float* row = logits + (batch * seq_len + (seq_len - 1)) * cols;
  float sample = samples != nullptr ? samples[batch] : 0.0f;
  if (!isfinite(temperature) || temperature <= 0.0f) {
    output_tokens[batch] = argmax_row_device(row, cols);
    return;
  }

  float max_scaled = -INFINITY;
  for (int idx = 0; idx < cols; ++idx) {
    float logit = row[idx];
    if (!isfinite(logit)) {
      continue;
    }
    float scaled = logit / temperature;
    if (isfinite(scaled) && scaled > max_scaled) {
      max_scaled = scaled;
    }
  }
  if (!isfinite(max_scaled)) {
    output_tokens[batch] = argmax_row_device(row, cols);
    return;
  }

  float cutoff = isfinite(top_p) ? fminf(fmaxf(top_p, 0.0f), 1.0f) : 1.0f;
  float uniform = isfinite(sample) ? fminf(fmaxf(sample, 0.0f), 0.99999994f) : 0.0f;
  int effective_top_k = top_k > 0 && top_k < cols ? top_k : cols;

  if (effective_top_k == cols && cutoff >= 1.0f) {
    float total = 0.0f;
    for (int idx = 0; idx < cols; ++idx) {
      total += sampling_weight(row, idx, temperature, max_scaled);
    }
    if (total <= 0.0f || !isfinite(total)) {
      output_tokens[batch] = argmax_row_device(row, cols);
      return;
    }
    float target = uniform * total;
    float cumulative = 0.0f;
    int last_positive = -1;
    for (int idx = 0; idx < cols; ++idx) {
      float weight = sampling_weight(row, idx, temperature, max_scaled);
      if (weight <= 0.0f) {
        continue;
      }
      last_positive = idx;
      cumulative += weight;
      if (target < cumulative) {
        output_tokens[batch] = static_cast<uint32_t>(idx);
        return;
      }
    }
    output_tokens[batch] = static_cast<uint32_t>(last_positive >= 0 ? last_positive : 0);
    return;
  }

  float total = 0.0f;
  float previous_weight = INFINITY;
  int previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    total += weight;
    previous_weight = weight;
    previous_id = token;
  }
  if (total <= 0.0f || !isfinite(total)) {
    output_tokens[batch] = argmax_row_device(row, cols);
    return;
  }

  float candidate_weight_total = 0.0f;
  float candidate_probability = 0.0f;
  int candidate_count = 0;
  previous_weight = INFINITY;
  previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    previous_weight = weight;
    previous_id = token;
    if (weight <= 0.0f) {
      continue;
    }
    candidate_weight_total += weight;
    candidate_probability += weight / total;
    ++candidate_count;
    if (cutoff < 1.0f && candidate_probability >= cutoff) {
      break;
    }
  }
  if (candidate_count <= 0 || candidate_weight_total <= 0.0f ||
      !isfinite(candidate_weight_total)) {
    output_tokens[batch] = argmax_row_device(row, cols);
    return;
  }

  float target = uniform * candidate_weight_total;
  float cumulative = 0.0f;
  int emitted = 0;
  previous_weight = INFINITY;
  previous_id = -1;
  for (int rank = 0; rank < effective_top_k; ++rank) {
    float weight = 0.0f;
    int token = next_ranked_token(
        row, cols, temperature, max_scaled, previous_weight, previous_id, &weight);
    if (token < 0) {
      break;
    }
    previous_weight = weight;
    previous_id = token;
    if (weight <= 0.0f) {
      continue;
    }
    ++emitted;
    cumulative += weight;
    if (target < cumulative || emitted == candidate_count) {
      output_tokens[batch] = static_cast<uint32_t>(token);
      return;
    }
    if (emitted >= candidate_count) {
      break;
    }
  }

  output_tokens[batch] = argmax_row_device(row, cols);
}

int valid_common(void* ptr, int len) {
  return ptr != nullptr && len >= 0;
}

}  // namespace

extern "C" int hi_cuda_launch_rms_norm(
    const void* input,
    const void* weight,
    void* output,
    int rows,
    int cols,
    float eps,
    void* stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows < 0 ||
      cols <= 0 || stream == nullptr) {
    return 1;
  }
  if (rows == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid(rows);
  rms_norm_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<const float*>(weight),
      static_cast<float*>(output),
      rows,
      cols,
      eps);
  return 0;
}

extern "C" int hi_cuda_launch_layer_norm(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int rows,
    int cols,
    float eps,
    void* stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows < 0 || cols <= 0 || stream == nullptr) {
    return 1;
  }
  if (rows == 0) {
    return 0;
  }
  dim3 block(128);
  dim3 grid((rows + block.x - 1) / block.x);
  layer_norm_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<const float*>(weight),
      static_cast<const float*>(bias),
      static_cast<float*>(output),
      rows,
      cols,
      eps);
  return 0;
}

extern "C" int hi_cuda_launch_silu_mul(
    const void* gate,
    const void* up,
    void* output,
    int len,
    void* stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      !valid_common(output, len) || stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  silu_mul_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(gate),
      static_cast<const float*>(up),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_gelu(
    const void* input,
    void* output,
    int len,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  gelu_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_gelu_mul(
    const void* gate,
    const void* up,
    void* output,
    int len,
    void* stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      !valid_common(output, len) || stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  gelu_mul_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(gate),
      static_cast<const float*>(up),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_softcap(
    const void* input,
    void* output,
    int len,
    float cap,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr || !(cap > 0.0f)) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  softcap_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<float*>(output),
      len,
      cap);
  return 0;
}

extern "C" int hi_cuda_launch_cast_f16_to_f32(
    const void* input,
    void* output,
    int len,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  cast_f16_to_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(input),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_cast_bf16_to_f32(
    const void* input,
    void* output,
    int len,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  cast_bf16_to_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint16_t*>(input),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_add(
    const void* left,
    const void* right,
    void* output,
    int len,
    void* stream) {
  if (left == nullptr || right == nullptr || output == nullptr ||
      !valid_common(output, len) || stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  add_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(left),
      static_cast<const float*>(right),
      static_cast<float*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_add_rowwise(
    const void* input,
    const void* bias,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows < 0 ||
      cols <= 0 || stream == nullptr) {
    return 1;
  }
  int total = rows * cols;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  add_rowwise_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<const float*>(bias),
      static_cast<float*>(output),
      rows,
      cols);
  return 0;
}

extern "C" int hi_cuda_launch_qwen_ssm_streaming_step(
    const void* qkv,
    const void* gate,
    const void* conv_weight,
    const void* ba,
    const void* dt_bias,
    const void* a_log,
    const void* norm_weight,
    void* conv_ring,
    void* recurrent_state,
    void* scratch,
    void* output,
    int conv_next,
    int conv_len,
    int conv_kernel,
    int conv_dim,
    int state_size,
    int time_step_rank,
    int group_count,
    int head_v_dim,
    int packed_qkvz,
    float eps,
    void* stream) {
  if (qkv == nullptr || conv_weight == nullptr || ba == nullptr ||
      dt_bias == nullptr || a_log == nullptr || norm_weight == nullptr ||
      conv_ring == nullptr || recurrent_state == nullptr || scratch == nullptr ||
      output == nullptr || stream == nullptr) {
    return 1;
  }
  if (!packed_qkvz && gate == nullptr) {
    return 1;
  }
  if (conv_next < 0 || conv_len < 0 || conv_kernel <= 0 || conv_dim <= 0 ||
      state_size <= 0 || time_step_rank <= 0 || group_count <= 0 ||
      head_v_dim <= 0 || conv_next >= conv_kernel || conv_len > conv_kernel ||
      time_step_rank % group_count != 0) {
    return 2;
  }
  int key_dim = group_count * state_size;
  int value_dim = time_step_rank * head_v_dim;
  if (conv_dim != 2 * key_dim + value_dim) {
    return 3;
  }
  qwen_ssm_streaming_step_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(qkv),
      static_cast<const float*>(gate),
      static_cast<const float*>(conv_weight),
      static_cast<const float*>(ba),
      static_cast<const float*>(dt_bias),
      static_cast<const float*>(a_log),
      static_cast<const float*>(norm_weight),
      static_cast<float*>(conv_ring),
      static_cast<float*>(recurrent_state),
      static_cast<float*>(scratch),
      static_cast<float*>(output),
      conv_next,
      conv_len,
      conv_kernel,
      conv_dim,
      state_size,
      time_step_rank,
      group_count,
      head_v_dim,
      packed_qkvz,
      eps);
  return 0;
}

extern "C" int hi_cuda_launch_copy_row_f32(
    const void* input,
    void* output,
    int row,
    int rows,
    int cols,
    void* stream) {
  if (input == nullptr || output == nullptr || row < 0 || rows <= 0 ||
      row >= rows || cols <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((cols + block.x - 1) / block.x);
  copy_row_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<float*>(output),
      row,
      rows,
      cols);
  return 0;
}

extern "C" int hi_cuda_launch_add_scaled_row_in_place(
    void* output,
    const void* row_values,
    int row,
    int rows,
    int cols,
    float scale,
    void* stream) {
  if (output == nullptr || row_values == nullptr || row < 0 || rows <= 0 ||
      row >= rows || cols <= 0 || !isfinite(scale) || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((cols + block.x - 1) / block.x);
  add_scaled_row_in_place_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(output),
      static_cast<const float*>(row_values),
      row,
      rows,
      cols,
      scale);
  return 0;
}

extern "C" int hi_cuda_launch_moe_topk_router(
    const void* scores,
    void* output_ids,
    void* output_weights,
    int rows,
    int experts,
    int top_k,
    int norm_topk,
    void* stream) {
  if (scores == nullptr || output_ids == nullptr || output_weights == nullptr ||
      rows <= 0 || experts <= 0 || top_k <= 0 || top_k > experts ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((rows + block.x - 1) / block.x);
  moe_topk_router_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(scores),
      static_cast<uint32_t*>(output_ids),
      static_cast<float*>(output_weights),
      rows,
      experts,
      top_k,
      norm_topk);
  return 0;
}

extern "C" int hi_cuda_launch_cast_f32_to_f16(
    const void* input,
    void* output,
    int len,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  cast_f32_to_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<__half*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_cast_f32_to_bf16(
    const void* input,
    void* output,
    int len,
    void* stream) {
  if (input == nullptr || output == nullptr || !valid_common(output, len) ||
      stream == nullptr) {
    return 1;
  }
  if (len == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((len + block.x - 1) / block.x);
  cast_f32_to_bf16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(input),
      static_cast<__nv_bfloat16*>(output),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_gather_rows_f16_to_f32(
    const void* matrix,
    const void* row_ids,
    void* output,
    int row_count,
    int cols,
    int matrix_rows,
    void* stream) {
  if (matrix == nullptr || row_ids == nullptr || output == nullptr ||
      row_count < 0 || cols <= 0 || matrix_rows <= 0 || stream == nullptr) {
    return 1;
  }
  int total = row_count * cols;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  gather_rows_f16_to_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __half*>(matrix),
      static_cast<const uint32_t*>(row_ids),
      static_cast<float*>(output),
      row_count,
      cols,
      matrix_rows);
  return 0;
}

extern "C" int hi_cuda_launch_gather_rows_bf16_to_f32(
    const void* matrix,
    const void* row_ids,
    void* output,
    int row_count,
    int cols,
    int matrix_rows,
    void* stream) {
  if (matrix == nullptr || row_ids == nullptr || output == nullptr ||
      row_count < 0 || cols <= 0 || matrix_rows <= 0 || stream == nullptr) {
    return 1;
  }
  int total = row_count * cols;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  gather_rows_bf16_to_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(matrix),
      static_cast<const uint32_t*>(row_ids),
      static_cast<float*>(output),
      row_count,
      cols,
      matrix_rows);
  return 0;
}

extern "C" int hi_cuda_launch_gather_rows_f32_to_f32(
    const void* matrix,
    const void* row_ids,
    void* output,
    int row_count,
    int cols,
    int matrix_rows,
    void* stream) {
  if (matrix == nullptr || row_ids == nullptr || output == nullptr ||
      row_count < 0 || cols <= 0 || matrix_rows <= 0 || stream == nullptr) {
    return 1;
  }
  int total = row_count * cols;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  gather_rows_f32_to_f32_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(matrix),
      static_cast<const uint32_t*>(row_ids),
      static_cast<float*>(output),
      row_count,
      cols,
      matrix_rows);
  return 0;
}

// Portable signed int8x4 dot-product accumulate (hardware dp4a where available).
__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
  return c + static_cast<int>(pa[0]) * static_cast<int>(pb[0]) +
         static_cast<int>(pa[1]) * static_cast<int>(pb[1]) +
         static_cast<int>(pa[2]) * static_cast<int>(pb[2]) +
         static_cast<int>(pa[3]) * static_cast<int>(pb[3]);
#endif
}

// Quantize an activation row x[k] to signed int8 blocks of 32: per block store the
// int8 quants (xq), f32 scale (dx = max|x|/127) and int sum of quants (xsum, to
// correct the Q4_0 -8 offset in the dp4a dot product).
__global__ void quantize_q8_row_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    float* __restrict__ dx,
    int* __restrict__ xsum,
    int k) {
  const int block = blockIdx.x;
  const int lane = threadIdx.x;
  const int idx = block * 32 + lane;
  const float v = (idx < k) ? x[idx] : 0.0f;
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  }
  const float d = amax / 127.0f;
  const float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
  int q = static_cast<int>(rintf(v * inv));
  q = max(-127, min(127, q));
  if (idx < k) {
    xq[idx] = static_cast<int8_t>(q);
  }
  int s = q;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
  }
  if (lane == 0) {
    dx[block] = d;
    xsum[block] = s;
  }
}

// Q4_0 x Q8 mat-vec via dp4a: y[row] = sum over 32-blocks of
//   d_w * d_x * (dp4a(q, xq) - 8 * sum(xq)).
// One warp per row; reads 4-bit weights + int8 activation (both quantized). Weight
// block layout matches dequantize_q4_0_kernel: byte j = weight j (low nibble) and
// weight j+16 (high nibble).
// Fused Q2_K GEMV (M=1 decode). Reads Q2_K weights directly, dequantizing each 2-bit
// weight on the fly into the dot product. Q2_K block = 84 bytes / 256 weights (16-byte
// packed 4-bit sub-block scales+mins + 64-byte 2-bit quants + f16 d + f16 dmin); the
// per-weight unpack mirrors dequantize_q2_k_kernel. One block per output row; f32
// activation. Requires cols % 256 == 0.
__global__ void q2_k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 84;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 84;
    const uint8_t* scales = blk;
    const uint8_t* qs = blk + 16;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 80));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 82));
    const int group16 = within >> 4;
    const int offset16 = within & 15;
    const int half128 = group16 >> 3;
    const int group_in_half = group16 & 7;
    const int pair = group_in_half >> 1;
    const bool upper16 = (group_in_half & 1) != 0;
    const int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
    const int shift = 2 * pair;
    const uint8_t sc = scales[group16];
    const uint8_t quant = (qs[q_index] >> shift) & 0x03;
    acc += (d * static_cast<float>(sc & 0x0f) * static_cast<float>(quant) -
            dmin * static_cast<float>(sc >> 4)) *
           x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_q2_k_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  q2_k_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused Q3_K GEMV (M=1 decode). Reads Q3_K weights directly, dequantizing each 3-bit
// weight on the fly into the dot product. Q3_K block = 110 bytes / 256 weights (32-byte
// high-bit hmask + 64-byte 2-bit low quants + 12-byte packed 6-bit scales + f16 d); the
// per-weight unpack mirrors dequantize_q3_k_kernel and reuses q3_k_scale. One block per
// output row; f32 activation. Requires cols % 256 == 0.
__global__ void q3_k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 110;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 110;
    const uint8_t* hmask = blk;
    const uint8_t* qs = blk + 32;
    const uint8_t* scales = blk + 96;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 108));
    const int group16 = within >> 4;
    const int offset16 = within & 15;
    const int half128 = group16 >> 3;
    const int group_in_half = group16 & 7;
    const int pair = group_in_half >> 1;
    const bool upper16 = (group_in_half & 1) != 0;
    const int q_index = half128 * 32 + (upper16 ? 16 : 0) + offset16;
    const int h_index = (upper16 ? 16 : 0) + offset16;
    const int shift = 2 * pair;
    const uint8_t mask = static_cast<uint8_t>(1u << (4 * half128 + pair));
    const int low = (qs[q_index] >> shift) & 0x03;
    const int quant = low - ((hmask[h_index] & mask) != 0 ? 0 : 4);
    acc += d * static_cast<float>(q3_k_scale(group16, scales)) *
           static_cast<float>(quant) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_q3_k_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  q3_k_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ3_XXS GEMV (M=1 decode). Reads IQ3_XXS weights directly: block = 98 bytes /
// 256 weights (f16 d + 64-B qs + 32-B scales_and_signs). Each qs byte indexes a uint32
// IQ3_XXS_GRID entry (4 packed 8-bit values); aux32 (4 bytes/group) carries packed signs
// (iq2_xxs_signs) + a 4-bit block scale (top nibble). Per-weight unpack mirrors
// dequantize_iq3_xxs_kernel. One block per output row; f32 activation. cols % 256 == 0.
__global__ void iq3_xxs_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 98;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 98;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* scales_and_signs = blk + 66;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const uint8_t* aux = scales_and_signs + 4 * group32;
    const uint32_t aux32 = static_cast<uint32_t>(aux[0]) |
                           (static_cast<uint32_t>(aux[1]) << 8) |
                           (static_cast<uint32_t>(aux[2]) << 16) |
                           (static_cast<uint32_t>(aux[3]) << 24);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((aux32 >> (7 * lane)) & 0x7f));
    const uint8_t q = qs[8 * group32 + 2 * lane + (j >= 4 ? 1 : 0)];
    const uint32_t grid = IQ3_XXS_GRID[q];
    const uint8_t value = static_cast<uint8_t>((grid >> (8 * (j & 3))) & 0xff);
    const float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
    acc += db * static_cast<float>(value) * sign * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq3_xxs_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq3_xxs_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ1_M GEMV (M=1 decode). Reads IQ1_M weights directly: block = 56 bytes / 256
// weights (32-B qs + 16-B qh + 8-B scales; no separate d). The f16 super-scale is
// reconstructed from the top nibbles of the four packed scale words; per-sub-block
// 3-bit scale + per-lane delta bit (in qh) + 11-bit grid index (qs + qh) into
// iq1_s_grid_code -> dl*(code + delta). Per-weight unpack mirrors dequantize_iq1_m_kernel.
// One block per output row; f32 activation. Requires cols % 256 == 0.
__global__ void iq1_m_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 56;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 56;
    const uint8_t* qs = blk;
    const uint8_t* qh = blk + 32;
    const uint8_t* scales = blk + 48;
    const uint16_t sc[4] = {
        static_cast<uint16_t>(static_cast<uint16_t>(scales[0]) |
                              (static_cast<uint16_t>(scales[1]) << 8)),
        static_cast<uint16_t>(static_cast<uint16_t>(scales[2]) |
                              (static_cast<uint16_t>(scales[3]) << 8)),
        static_cast<uint16_t>(static_cast<uint16_t>(scales[4]) |
                              (static_cast<uint16_t>(scales[5]) << 8)),
        static_cast<uint16_t>(static_cast<uint16_t>(scales[6]) |
                              (static_cast<uint16_t>(scales[7]) << 8)),
    };
    const uint16_t scale_bits = static_cast<uint16_t>(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) |
        (sc[3] & 0xf000));
    const float d = f16_bits_to_float(scale_bits);
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const int scale_index = 2 * group32 + lane / 2;
    const uint16_t scale_word = sc[scale_index / 4];
    const float dl =
        d * static_cast<float>(2 * ((scale_word >> (3 * (scale_index & 3))) & 0x07) + 1);
    const uint8_t qh_byte = qh[2 * group32 + lane / 2];
    const int qh_shift = 4 * (lane & 1);
    const float delta = (qh_byte & (0x08u << qh_shift)) != 0 ? -1.125f : -0.875f;
    const int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                           (static_cast<int>((qh_byte >> qh_shift) & 0x07) << 8);
    const uint8_t code = iq1_s_grid_code(grid_index, j);
    acc += dl * (static_cast<float>(code) + delta) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq1_m_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq1_m_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ1_S GEMV (M=1 decode). Reads IQ1_S weights directly: block = 50 bytes / 256
// weights (f16 d + 32-B qs + 16-B qh). Per group the qh word carries a 3-bit scale, a
// sign->delta bit, and 3 high bits of each 11-bit grid index (rest from qs); code comes
// from iq1_s_grid_code and reconstructs as dl*(code + delta). Per-weight unpack mirrors
// dequantize_iq1_s_kernel. One block per output row; f32 activation. cols % 256 == 0.
__global__ void iq1_s_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 50;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 50;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 34;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const uint16_t qh_word = static_cast<uint16_t>(qh[2 * group32]) |
                             (static_cast<uint16_t>(qh[2 * group32 + 1]) << 8);
    const float dl = d * static_cast<float>(2 * ((qh_word >> 12) & 7) + 1);
    const float delta = (qh_word & 0x8000) != 0 ? -1.125f : -0.875f;
    const int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                           (static_cast<int>((qh_word >> (3 * lane)) & 0x07) << 8);
    const uint8_t code = iq1_s_grid_code(grid_index, j);
    acc += dl * (static_cast<float>(code) + delta) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq1_s_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq1_s_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ2_XS GEMV (M=1 decode). Reads IQ2_XS weights directly: block = 74 bytes / 256
// weights (f16 d + 64-B uint16 qs + 8-B scales). Each uint16 packs a 9-bit grid index
// (into IQ2_XS_GRID, 2-bit values via IQ2_XXS_VALUES) + 7 sign bits (iq2_xxs_signs);
// per-lane 4-bit sub-block scale. Per-weight unpack mirrors dequantize_iq2_xs_kernel.
// One block per output row; f32 activation. Requires cols % 256 == 0.
__global__ void iq2_xs_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 74;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 74;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* scales = blk + 66;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const int q_index = group32 * 4 + lane;
    const uint16_t q = static_cast<uint16_t>(qs[2 * q_index]) |
                       (static_cast<uint16_t>(qs[2 * q_index + 1]) << 8);
    const uint8_t scale = scales[group32];
    const float db =
        d * (0.5f + static_cast<float>(lane < 2 ? (scale & 0x0f) : (scale >> 4))) * 0.25f;
    const uint16_t grid = IQ2_XS_GRID[q & 0x01ff];
    const uint8_t value = IQ2_XXS_VALUES[(grid >> (2 * j)) & 0x03];
    const uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((q >> 9) & 0x7f));
    const float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
    acc += db * static_cast<float>(value) * sign * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq2_xs_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq2_xs_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ2_S GEMV (M=1 decode). Reads IQ2_S weights directly: block = 82 bytes / 256
// weights (f16 d + 32-B qs + 32-B signs + 8-B qh + 8-B scales). Each group of 4 weights
// comes from a 10-bit grid index (qs byte + qh 2 bits) into iq2_s_grid_value, times a
// per-weight sign bit and a (0.5 + scale)*0.25 sub-block scale. Per-weight unpack mirrors
// dequantize_iq2_s_kernel. One block per output row; f32 activation. cols % 256 == 0.
__global__ void iq2_s_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 82;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 82;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* signs = blk + 34;
    const uint8_t* qh = blk + 66;
    const uint8_t* scales = blk + 74;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const float db = d * (0.5f + static_cast<float>(scales[group32] >> 4)) * 0.25f;
    const int grid_index = static_cast<int>(qs[4 * group32 + lane]) |
                           (static_cast<int>((qh[group32] >> (2 * lane)) & 0x03) << 8);
    const uint8_t value = iq2_s_grid_value(grid_index, j);
    const uint8_t signs_byte = signs[4 * group32 + lane];
    const float sign = (signs_byte & (1u << j)) != 0 ? -1.0f : 1.0f;
    acc += db * static_cast<float>(value) * sign * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq2_s_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq2_s_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ2_XXS GEMV (M=1 decode). Reads IQ2_XXS weights directly: block = 66 bytes /
// 256 weights (f16 d + 8 groups x 8 bytes). Per group: q[0..3] index IQ2_XXS_GRID (2-bit
// values via IQ2_XXS_VALUES), aux32 (q[4..7]) carries the packed signs (iq2_xxs_signs)
// and the 4-bit block scale (top nibble). Per-weight unpack mirrors
// dequantize_iq2_xxs_kernel. One block per output row; f32 activation. cols % 256 == 0.
__global__ void iq2_xxs_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 66;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 66;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const uint8_t* q = blk + 2 + group32 * 8;
    const uint32_t aux32 = static_cast<uint32_t>(q[4]) |
                           (static_cast<uint32_t>(q[5]) << 8) |
                           (static_cast<uint32_t>(q[6]) << 16) |
                           (static_cast<uint32_t>(q[7]) << 24);
    const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.25f;
    const uint16_t grid = IQ2_XXS_GRID[q[lane]];
    const uint8_t value = IQ2_XXS_VALUES[(grid >> (2 * j)) & 0x03];
    const uint8_t signs = iq2_xxs_signs(static_cast<uint8_t>((aux32 >> (7 * lane)) & 0x7f));
    const float sign = (signs & (1u << j)) != 0 ? -1.0f : 1.0f;
    acc += db * static_cast<float>(value) * sign * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq2_xxs_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq2_xxs_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ3_S GEMV (M=1 decode). Reads IQ3_S weights directly: block = 110 bytes / 256
// weights (f16 d + 64-B qs + 8-B qh + 32-B signs + 4-B scales). Each group of 4 weights
// comes from a 9-bit grid index (qs byte + qh high bit) into iq3_s_grid_value, times a
// per-weight sign bit and a (1 + 2*scale) 4-bit sub-block scale. Per-weight unpack
// mirrors dequantize_iq3_s_kernel. One block per output row; f32 activation. cols%256==0.
__global__ void iq3_s_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 110;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 110;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 66;
    const uint8_t* signs = blk + 74;
    const uint8_t* scales = blk + 106;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int lane = offset32 >> 3;
    const int j = offset32 & 7;
    const uint8_t scale_byte = scales[group32 >> 1];
    const uint8_t scale = (scale_byte >> (4 * (group32 & 1))) & 0x0f;
    const float db = d * static_cast<float>(1 + 2 * scale);
    const int q_slot = 2 * lane + (j >= 4 ? 1 : 0);
    const int grid_index = static_cast<int>(qs[8 * group32 + q_slot]) |
                           (static_cast<int>((qh[group32] >> q_slot) & 0x01) << 8);
    const uint8_t value = iq3_s_grid_value(grid_index, j & 3);
    const uint8_t signs_byte = signs[4 * group32 + lane];
    const float sign = (signs_byte & (1u << j)) != 0 ? -1.0f : 1.0f;
    acc += db * static_cast<float>(value) * sign * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq3_s_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq3_s_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ4_XS GEMV (M=1 decode). Reads IQ4_XS weights directly: block = 136 bytes / 256
// weights (f16 d + uint16 scales_h + 4-byte scales_l + 128-byte 4-bit indices into the
// fixed IQ4_NL_VALUES table). Per-32 sub-block 6-bit scale (scales_l low 4 bits +
// scales_h high 2 bits, minus 32). Per-weight unpack mirrors dequantize_iq4_xs_kernel.
// One block per output row; f32 activation. Requires cols % 256 == 0.
__global__ void iq4_xs_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 136;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 136;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint16_t scales_h =
        static_cast<uint16_t>(blk[2]) | (static_cast<uint16_t>(blk[3]) << 8);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const uint8_t scale_low = (scales_l[group32 >> 1] >> (4 * (group32 & 1))) & 0x0f;
    const uint8_t scale_high = static_cast<uint8_t>((scales_h >> (2 * group32)) & 0x03);
    const float dl =
        d * static_cast<float>(static_cast<int>(scale_low | (scale_high << 4)) - 32);
    const uint8_t packed = qs[group32 * 16 + (offset32 & 15)];
    const uint8_t quant = offset32 < 16 ? (packed & 0x0f) : (packed >> 4);
    acc += dl * static_cast<float>(IQ4_NL_VALUES[quant]) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq4_xs_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  iq4_xs_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused IQ4_NL GEMV (M=1 decode). Reads IQ4_NL weights directly: block = 18 bytes / 32
// weights (f16 d + 16-byte 4-bit indices into the fixed IQ4_NL_VALUES non-linear table).
// Per-weight unpack mirrors dequantize_iq4_nl_kernel. One block per output row; f32
// activation. Requires cols % 32 == 0.
__global__ void iq4_nl_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nblk = cols / 32;
  const size_t row_bytes = static_cast<size_t>(nblk) * 18;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int blk_id = c >> 5;
    const int within = c & 31;
    const uint8_t* blk = row_ptr + static_cast<size_t>(blk_id) * 18;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const uint8_t packed = blk[2 + (within & 15)];
    const uint8_t quant = within < 16 ? (packed & 0x0f) : (packed >> 4);
    acc += d * static_cast<float>(IQ4_NL_VALUES[quant]) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_iq4_nl_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 32 != 0 || stream == nullptr) {
    return 1;
  }
  iq4_nl_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused Q5_K GEMV (M=1 decode). Reads Q5_K weights directly, dequantizing each 5-bit
// weight on the fly into the dot product. Q5_K block = 176 bytes / 256 weights (f16 d +
// f16 dmin + 12-byte packed 6-bit sub-block scales/mins + 32-byte high bits + 128-byte
// 4-bit low quants); the per-weight unpack mirrors dequantize_q5_k_kernel. One block per
// output row; f32 activation. Requires cols % 256 == 0.
__global__ void q5_k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 176;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 176;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* qs = blk + 48;
    const int group32 = within >> 5;
    const int offset32 = within & 31;
    const int group64 = group32 >> 1;
    uint8_t scale;
    uint8_t mn;
    q4_k_scale_min(group32, scales, &scale, &mn);
    const uint8_t packed = qs[group64 * 32 + offset32];
    const uint8_t low = (group32 & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
    const uint8_t high = (qh[offset32] & (1u << group32)) != 0 ? 16 : 0;
    const uint8_t quant = low + high;
    acc += (d * static_cast<float>(scale) * static_cast<float>(quant) -
            dmin * static_cast<float>(mn)) *
           x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_q5_k_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  q5_k_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused Q4_K GEMV (M=1 decode). Reads Q4_K weights directly, dequantizing each 4-bit
// weight on the fly into the dot product instead of materializing the whole f32 weight
// matrix every token. Q4_K block = 144 bytes / 256 weights (f16 d + f16 dmin + 12-byte
// packed 6-bit sub-block scales/mins + 128-byte 4-bit quants); the per-weight unpack
// mirrors dequantize_q4_k_kernel. One block per output row; f32 activation. cols%256==0.
__global__ void q4_k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 144;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 144;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    const int group64 = within >> 6;
    const int offset64 = within & 63;
    const int scale_index = group64 * 2 + (offset64 >= 32 ? 1 : 0);
    uint8_t scale;
    uint8_t mn;
    q4_k_scale_min(scale_index, scales, &scale, &mn);
    const uint8_t packed = qs[group64 * 32 + (offset64 & 31)];
    const uint8_t quant = offset64 < 32 ? (packed & 0x0f) : (packed >> 4);
    acc += (d * static_cast<float>(scale) * static_cast<float>(quant) -
            dmin * static_cast<float>(mn)) *
           x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_q4_k_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  q4_k_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

// Fused Q6_K GEMV (M=1 decode). Reads Q6_K weights directly and dequantizes each
// weight on the fly into the dot product, instead of materializing the whole f32
// weight matrix every token. Q6_K block = 210 bytes / 256 weights (128 ql + 64 qh
// + 16 int8 scales + f16 d); the per-weight unpack mirrors dequantize_q6_k_kernel.
// One block per output row; f32 activation. Requires cols % 256 == 0.
__global__ void q6_k_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nsb = cols / 256;
  const size_t row_bytes = static_cast<size_t>(nsb) * 210;
  const uint8_t* row_ptr = weights + static_cast<size_t>(row) * row_bytes;
  float acc = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    const int sb = c >> 8;
    const int within = c & 255;
    const uint8_t* blk = row_ptr + static_cast<size_t>(sb) * 210;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    const int half = within >> 7;
    const int pos = within & 127;
    const int l = pos & 31;
    const int group = pos >> 5;
    const int ql_base = half * 64;
    const int qh_base = half * 32;
    const int scale_base = half * 8;
    const int is = l >> 4;
    const uint8_t high = blk[128 + qh_base + l];
    int quant;
    int scale_index = scale_base + is;
    switch (group) {
      case 0:
        quant = (blk[ql_base + l] & 0x0f) | (((high >> 0) & 0x03) << 4);
        break;
      case 1:
        quant = (blk[ql_base + l + 32] & 0x0f) | (((high >> 2) & 0x03) << 4);
        scale_index += 2;
        break;
      case 2:
        quant = (blk[ql_base + l] >> 4) | (((high >> 4) & 0x03) << 4);
        scale_index += 4;
        break;
      default:
        quant = (blk[ql_base + l + 32] >> 4) | (((high >> 6) & 0x03) << 4);
        scale_index += 6;
        break;
    }
    const int8_t scale = static_cast<int8_t>(blk[192 + scale_index]);
    acc += d * static_cast<float>(scale) * static_cast<float>(quant - 32) * x[c];
  }
  __shared__ float warp_sums[32];
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = blockDim.x >> 5;
    float v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) {
      output[row] = v;
    }
  }
}

extern "C" int hi_cuda_launch_q6_k_gemv(
    const void* weights,
    const void* x,
    void* output,
    int rows,
    int cols,
    void* stream) {
  if (weights == nullptr || x == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 256 != 0 || stream == nullptr) {
    return 1;
  }
  q6_k_gemv_kernel<<<rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weights), static_cast<const float*>(x),
      static_cast<float*>(output), rows, cols);
  return 0;
}

__global__ void q4_0_dp4a_gemv_kernel(
    const uint8_t* __restrict__ weight,
    const int8_t* __restrict__ xq,
    const float* __restrict__ dx,
    const int* __restrict__ xsum,
    float* __restrict__ y,
    int rows,
    int cols) {
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int nblocks = cols / 32;
  const uint8_t* row_base = weight + static_cast<size_t>(row) * nblocks * 18;
  float acc = 0.0f;
  for (int b = lane; b < nblocks; b += 32) {
    const uint8_t* blk = row_base + static_cast<size_t>(b) * 18;
    const float dw = __half2float(*reinterpret_cast<const __half*>(blk));
    const int8_t* xqb = xq + b * 32;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint8_t* pb = blk + 2 + j * 4;
      const uint32_t packed = static_cast<uint32_t>(pb[0]) |
                              (static_cast<uint32_t>(pb[1]) << 8) |
                              (static_cast<uint32_t>(pb[2]) << 16) |
                              (static_cast<uint32_t>(pb[3]) << 24);
      const int lo = static_cast<int>(packed & 0x0F0F0F0Fu);
      const int hi = static_cast<int>((packed >> 4) & 0x0F0F0F0Fu);
      const int xlo = *reinterpret_cast<const int*>(xqb + j * 4);
      const int xhi = *reinterpret_cast<const int*>(xqb + 16 + j * 4);
      sumi = dp4a_i8(lo, xlo, sumi);
      sumi = dp4a_i8(hi, xhi, sumi);
    }
    acc += dw * dx[b] * (static_cast<float>(sumi) - 8.0f * static_cast<float>(xsum[b]));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) {
    y[row] = acc;
  }
}

extern "C" int hi_cuda_launch_quantize_q8_row(
    const void* x, void* xq, void* dx, void* xsum, int k, void* stream) {
  if (x == nullptr || xq == nullptr || dx == nullptr || xsum == nullptr || k <= 0 ||
      k % 32 != 0 || stream == nullptr) {
    return 1;
  }
  quantize_q8_row_kernel<<<k / 32, 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(x), static_cast<int8_t*>(xq),
      static_cast<float*>(dx), static_cast<int*>(xsum), k);
  return 0;
}

extern "C" int hi_cuda_launch_q4_0_dp4a_gemv(
    const void* weight, const void* xq, const void* dx, const void* xsum, void* y,
    int rows, int cols, void* stream) {
  if (weight == nullptr || xq == nullptr || dx == nullptr || xsum == nullptr ||
      y == nullptr || rows <= 0 || cols <= 0 || cols % 32 != 0 || stream == nullptr) {
    return 1;
  }
  const int warps_per_block = 4;
  const int grid = (rows + warps_per_block - 1) / warps_per_block;
  q4_0_dp4a_gemv_kernel<<<grid, warps_per_block * 32, 0,
                          static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(weight), static_cast<const int8_t*>(xq),
      static_cast<const float*>(dx), static_cast<const int*>(xsum),
      static_cast<float*>(y), rows, cols);
  return 0;
}

// Flash-attention, causal + GQA, batched. One block per (batch, head, query-tile of
// HI_CUDA_FLASH_QWARPS queries); one warp per query. Each key tile is loaded to shared
// memory once and reused by all QWARPS queries in the block, so global K/V traffic is
// ~1/QWARPS of the per-query kernel. Online softmax over key tiles, keys processed in
// ascending order (matches tiled_causal_attention_batched_kernel for parity). Causal
// only (no sliding window). Shared layout: [KTILE*qk_head_dim K][KTILE*v_head_dim V].
__global__ void flashtile_causal_attention_batched_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    int ktile) {
  extern __shared__ float smem[];
  float* k_tile = smem;
  float* v_tile = smem + ktile * qk_head_dim;

  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int target = blockIdx.x * HI_CUDA_FLASH_QWARPS + warp_id;
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(qk_head_dim));
  const bool active = target < seq_len;

  constexpr int MAX_PER_LANE = HI_CUDA_FLASH_MAX_HEAD_DIM / 32;
  float q_reg[MAX_PER_LANE];
  float acc[MAX_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_PER_LANE; ++i) {
    const int d = lane + i * 32;
    q_reg[i] = (active && d < qk_head_dim)
                   ? q[((batch * seq_len + target) * heads + head) * qk_head_dim + d]
                   : 0.0f;
    acc[i] = 0.0f;
  }
  float m = -INFINITY;
  float l = 0.0f;

  const int block_max_target =
      min((blockIdx.x + 1) * HI_CUDA_FLASH_QWARPS - 1, seq_len - 1);
  // Sliding window: no query in this block needs keys before (block_min - window + 1);
  // start the tile loop there (rounded down to a tile) and mask per query below.
  int tile_lo = 0;
  if (window > 0) {
    const int block_min = blockIdx.x * HI_CUDA_FLASH_QWARPS - window + 1;
    tile_lo = block_min > 0 ? (block_min / ktile) * ktile : 0;
  }
  for (int tile_start = tile_lo; tile_start <= block_max_target; tile_start += ktile) {
    for (int e = threadIdx.x; e < ktile * qk_head_dim; e += blockDim.x) {
      const int j = e / qk_head_dim;
      const int d = e - j * qk_head_dim;
      const int source = tile_start + j;
      k_tile[e] = (source < seq_len)
                      ? k[((batch * seq_len + source) * kv_heads + kv_head) * qk_head_dim + d]
                      : 0.0f;
    }
    for (int e = threadIdx.x; e < ktile * v_head_dim; e += blockDim.x) {
      const int j = e / v_head_dim;
      const int d = e - j * v_head_dim;
      const int source = tile_start + j;
      v_tile[e] = (source < seq_len)
                      ? v[((batch * seq_len + source) * kv_heads + kv_head) * v_head_dim + d]
                      : 0.0f;
    }
    __syncthreads();

    if (active) {
      const int tile_end = min(tile_start + ktile, target + 1);
      // Causal upper bound is target+1; sliding-window lower bound is target-window+1.
      const int first = (window > 0 && target - window + 1 > tile_start)
                            ? target - window + 1
                            : tile_start;
      for (int source = first; source < tile_end; ++source) {
        const float* kt = k_tile + (source - tile_start) * qk_head_dim;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_PER_LANE; ++i) {
          const int d = lane + i * 32;
          if (d < qk_head_dim) {
            dot += q_reg[i] * kt[d];
          }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_down_sync(0xffffffffu, dot, off);
        }
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float next_m = fmaxf(m, score);
        const float rescale = isfinite(m) ? expf(m - next_m) : 0.0f;
        const float weight = expf(score - next_m);
        const float* vt = v_tile + (source - tile_start) * v_head_dim;
#pragma unroll
        for (int i = 0; i < MAX_PER_LANE; ++i) {
          const int d = lane + i * 32;
          if (d < v_head_dim) {
            acc[i] = acc[i] * rescale + weight * vt[d];
          }
        }
        l = l * rescale + weight;
        m = next_m;
      }
    }
    __syncthreads();
  }

  if (active) {
    const float inv = (l == 0.0f || !isfinite(l)) ? 0.0f : 1.0f / l;
#pragma unroll
    for (int i = 0; i < MAX_PER_LANE; ++i) {
      const int d = lane + i * 32;
      if (d < v_head_dim) {
        output[((batch * seq_len + target) * heads + head) * v_head_dim + d] = acc[i] * inv;
      }
    }
  }
}

// Tensor-core (WMMA) flash-attention, causal, batched. One warp per block per
// (batch, head, 16-query tile). Q*K^T and P*V run on tensor cores (f16 in, f32
// accumulate); the online softmax + O rescale live in shared memory (clear
// indexing) so no accumulator-fragment surgery is needed. head_dim must be a
// multiple of 16 (<=128). f16 matmuls, so NOT bit-parity with the f32 kernel;
// gated opt-in and validated by coherence/retrieval.
namespace wmma = nvcuda::wmma;
constexpr int HI_CUDA_WMMA_TILE = 16;

__global__ void wmma_causal_attention_batched_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int head_dim) {
  const int T = HI_CUDA_WMMA_TILE;
  extern __shared__ float smem_f[];
  // Layout (bytes): Q[T*head_dim half] K[T*head_dim half] V[T*head_dim half]
  //                 P[T*T half] S/O in float section.
  half* q_sh = reinterpret_cast<half*>(smem_f);
  half* k_sh = q_sh + T * head_dim;
  half* v_sh = k_sh + T * head_dim;
  half* p_sh = v_sh + T * head_dim;
  // float section: m[T], l[T], S[T*T], O[T*head_dim]
  float* s_sh = reinterpret_cast<float*>(p_sh + T * T);
  float* o_sh = s_sh + 2 * T + T * T;

  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int q0 = blockIdx.x * T;       // first query in tile
  const int kv_repeats = heads / kv_heads;
  const int kv_head = head / kv_repeats;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int tid = threadIdx.x;         // 0..31 (one warp)

  // load Q tile -> shared (f16), zero-pad rows past seq_len
  for (int e = tid; e < T * head_dim; e += 32) {
    const int r = e / head_dim, d = e - r * head_dim, qq = q0 + r;
    q_sh[e] = (qq < seq_len)
                  ? q[((batch * seq_len + qq) * heads + head) * head_dim + d]
                  : __float2half(0.0f);
  }
  for (int e = tid; e < T * head_dim; e += 32) o_sh[e] = 0.0f;
  float m_run = -INFINITY, l_run = 0.0f;  // per-query, held by thread==row later
  if (tid < T) { s_sh[tid] = -INFINITY; s_sh[T + tid] = 0.0f; }  // m,l per row in s_sh[0..],[T..]
  __syncwarp();

  const int hd_tiles = head_dim / T;
  const int block_max_q = min(q0 + T - 1, seq_len - 1);
  for (int kt = 0; kt <= block_max_q; kt += T) {
    // load K,V tile -> shared
    for (int e = tid; e < T * head_dim; e += 32) {
      const int r = e / head_dim, d = e - r * head_dim, src = kt + r;
      k_sh[e] = (src < seq_len)
                    ? k[((batch * seq_len + src) * kv_heads + kv_head) * head_dim + d]
                    : __float2half(0.0f);
      v_sh[e] = (src < seq_len)
                    ? v[((batch * seq_len + src) * kv_heads + kv_head) * head_dim + d]
                    : __float2half(0.0f);
    }
    __syncwarp();
    // S = Q @ K^T (tensor cores)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);
    for (int h = 0; h < hd_tiles; ++h) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
      wmma::load_matrix_sync(qf, q_sh + h * T, head_dim);
      wmma::load_matrix_sync(kf, k_sh + h * T, head_dim);
      wmma::mma_sync(s_frag, qf, kf, s_frag);
    }
    wmma::store_matrix_sync(s_sh + 2 * T, s_frag, T, wmma::mem_row_major);  // S at s_sh+2T
    __syncwarp();
    // softmax over this key tile (thread == query row), causal + rescale O
    if (tid < T) {
      const int qpos = q0 + tid;
      float* srow = s_sh + 2 * T + tid * T;
      float rmax = -INFINITY;
      for (int j = 0; j < T; ++j) {
        const int kpos = kt + j;
        float sc = (kpos < seq_len && kpos <= qpos) ? srow[j] * scale : -INFINITY;
        srow[j] = sc;
        rmax = fmaxf(rmax, sc);
      }
      const float m_old = s_sh[tid];
      const float new_m = fmaxf(m_old, rmax);
      const float factor = isfinite(m_old) ? __expf(m_old - new_m) : 0.0f;
      float rsum = 0.0f;
      for (int j = 0; j < T; ++j) {
        const float p = isfinite(srow[j]) ? __expf(srow[j] - new_m) : 0.0f;
        p_sh[tid * T + j] = __float2half(p);
        rsum += p;
      }
      s_sh[T + tid] = s_sh[T + tid] * factor + rsum;  // l
      s_sh[tid] = new_m;                                // m
      float* orow = o_sh + tid * head_dim;
      for (int d = 0; d < head_dim; ++d) orow[d] *= factor;  // rescale O
    }
    __syncwarp();
    // O += P @ V (tensor cores), per head_dim tile
    for (int h = 0; h < hd_tiles; ++h) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
      wmma::load_matrix_sync(pf, p_sh, T);
      wmma::load_matrix_sync(vf, v_sh + h * T, head_dim);
      wmma::load_matrix_sync(of, o_sh + h * T, head_dim, wmma::mem_row_major);
      wmma::mma_sync(of, pf, vf, of);
      wmma::store_matrix_sync(o_sh + h * T, of, head_dim, wmma::mem_row_major);
    }
    __syncwarp();
  }
  // write O / l
  if (tid < T) {
    const int qpos = q0 + tid;
    if (qpos < seq_len) {
      const float l = s_sh[T + tid];
      const float inv = (l == 0.0f || !isfinite(l)) ? 0.0f : 1.0f / l;
      float* orow = o_sh + tid * head_dim;
      for (int d = 0; d < head_dim; ++d)
        output[((batch * seq_len + qpos) * heads + head) * head_dim + d] = orow[d] * inv;
    }
  }
}

extern "C" int hi_cuda_launch_wmma_causal_attention_batched(
    const void* q, const void* k, const void* v, void* output,
    int batch_count, int seq_len, int heads, int kv_heads, int head_dim, void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch_count <= 0 || seq_len <= 0 || heads <= 0 || kv_heads <= 0 ||
      head_dim <= 0 || head_dim % 16 != 0 || head_dim > 128 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  const int T = HI_CUDA_WMMA_TILE;
  const int qtiles = (seq_len + T - 1) / T;
  dim3 grid(qtiles, heads, batch_count);
  dim3 block(32);
  size_t shared_bytes = (3 * T * head_dim + T * T) * sizeof(half) +
                        (2 * T + T * T + T * head_dim) * sizeof(float);
  wmma_causal_attention_batched_kernel<<<grid, block, shared_bytes,
                                         static_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(q), static_cast<const half*>(k),
      static_cast<const half*>(v), static_cast<float*>(output), batch_count, seq_len,
      heads, kv_heads, head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_flashtile_causal_attention_batched(
    const void* q, const void* k, const void* v, void* output,
    int batch_count, int seq_len, int heads, int kv_heads,
    int qk_head_dim, int v_head_dim, int window, void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch_count <= 0 || seq_len <= 0 || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  // Pick the largest key tile whose K+V shared-memory footprint fits (~45 KB margin);
  // 32 for head_dim<=~192, 16 for 256, smaller for wider heads.
  int ktile = HI_CUDA_FLASH_KTILE;
  const size_t per_tile = static_cast<size_t>(qk_head_dim + v_head_dim) * sizeof(float);
  while (ktile > 1 && static_cast<size_t>(ktile) * per_tile > 46080) {
    ktile >>= 1;
  }
  const int qtiles = (seq_len + HI_CUDA_FLASH_QWARPS - 1) / HI_CUDA_FLASH_QWARPS;
  dim3 grid(qtiles, heads, batch_count);
  dim3 block(HI_CUDA_FLASH_QWARPS * 32);
  size_t shared_bytes = static_cast<size_t>(ktile) * per_tile;
  flashtile_causal_attention_batched_kernel<<<grid, block, shared_bytes,
                                              static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(v), static_cast<float*>(output), batch_count, seq_len,
      heads, kv_heads, qk_head_dim, v_head_dim, window, ktile);
  return 0;
}

extern "C" int hi_cuda_launch_dequantize_matrix(
    const void* input,
    void* output,
    int elements,
    int quant_type,
    void* stream) {
  if (input == nullptr || output == nullptr || elements < 0 || stream == nullptr) {
    return 1;
  }
  if (elements == 0) {
    return 0;
  }
  if ((quant_type == 2 || quant_type == 3 || quant_type == 6 || quant_type == 7 ||
       quant_type == 8 || quant_type == 9 || quant_type == 20 ||
       quant_type == 31 || quant_type == 32 || quant_type == 33 ||
       quant_type == 36 || quant_type == 37 || quant_type == 38 ||
       quant_type == 39) &&
      elements % 32 != 0) {
    return 1;
  }
  if (quant_type == 40 && elements % 64 != 0) {
    return 1;
  }
  if (quant_type == 41 && elements % 128 != 0) {
    return 1;
  }
  if ((quant_type == 10 || quant_type == 11 || quant_type == 12 || quant_type == 13 ||
       quant_type == 14 || quant_type == 15 || quant_type == 16 || quant_type == 17 ||
       quant_type == 18 || quant_type == 19 || quant_type == 21 || quant_type == 22 ||
       quant_type == 23 || quant_type == 29 ||
       quant_type == 34 || quant_type == 35) &&
      elements % 256 != 0) {
    return 1;
  }
  dim3 block(256);
  dim3 grid((elements + block.x - 1) / block.x);
  switch (quant_type) {
    case 2:
    case 31:
    case 32:
    case 33:
      dequantize_q4_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 3:
      dequantize_q4_1_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 41:
      dequantize_q1_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 39:
      dequantize_mxfp4_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 40:
      dequantize_nvfp4_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 6:
      dequantize_q5_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 7:
      dequantize_q5_1_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 8:
      dequantize_q8_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 9:
      dequantize_q8_1_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 20:
    case 36:
    case 37:
    case 38:
      dequantize_iq4_nl_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 23:
      dequantize_iq4_xs_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 16:
      dequantize_iq2_xxs_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 17:
      dequantize_iq2_xs_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 18:
      dequantize_iq3_xxs_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 19:
      dequantize_iq1_s_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 21:
      dequantize_iq3_s_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 22:
      dequantize_iq2_s_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 29:
      dequantize_iq1_m_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 34:
      dequantize_tq1_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 35:
      dequantize_tq2_0_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 10:
      dequantize_q2_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 11:
      dequantize_q3_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 12:
      dequantize_q4_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 13:
      dequantize_q5_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 14:
      dequantize_q6_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    case 15:
      dequantize_q8_k_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const uint8_t*>(input),
          static_cast<float*>(output),
          elements);
      return 0;
    default:
      return 1;
  }
}

extern "C" int hi_cuda_launch_rope(
    void* values,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int split_half,
    void* stream) {
  if (values == nullptr || seq_len < 0 || heads <= 0 || head_dim <= 0 ||
      head_dim % 2 != 0 || base <= 0.0f || stream == nullptr) {
    return 1;
  }
  int total_pairs = seq_len * heads * (head_dim / 2);
  if (total_pairs == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total_pairs + block.x - 1) / block.x);
  rope_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      seq_len,
      heads,
      head_dim,
      base,
      scale,
      0,
      split_half);
  return 0;
}

extern "C" int hi_cuda_launch_rope_with_offset(
    void* values,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int position_offset,
    int split_half,
    void* stream) {
  if (values == nullptr || seq_len < 0 || heads <= 0 || head_dim <= 0 ||
      head_dim % 2 != 0 || base <= 0.0f || position_offset < 0 ||
      stream == nullptr) {
    return 1;
  }
  int total_pairs = seq_len * heads * (head_dim / 2);
  if (total_pairs == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total_pairs + block.x - 1) / block.x);
  rope_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      seq_len,
      heads,
      head_dim,
      base,
      scale,
      position_offset,
      split_half);
  return 0;
}

extern "C" int hi_cuda_launch_rope_batched_with_offset(
    void* values,
    int batch_count,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int position_offset,
    int split_half,
    void* stream) {
  if (values == nullptr || batch_count <= 0 || seq_len <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim % 2 != 0 || base <= 0.0f ||
      position_offset < 0 || stream == nullptr) {
    return 1;
  }
  int total_pairs = batch_count * seq_len * heads * (head_dim / 2);
  dim3 block(256);
  dim3 grid((total_pairs + block.x - 1) / block.x);
  rope_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      batch_count,
      seq_len,
      heads,
      head_dim,
      base,
      scale,
      position_offset,
      split_half);
  return 0;
}

extern "C" int hi_cuda_launch_rope_batched_positions(
    void* values,
    const void* positions,
    int batch_count,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int split_half,
    void* stream) {
  if (values == nullptr || positions == nullptr || batch_count <= 0 ||
      seq_len <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      base <= 0.0f || stream == nullptr) {
    return 1;
  }
  int pairs = batch_count * seq_len * heads * (head_dim / 2);
  dim3 block(256);
  dim3 grid((pairs + block.x - 1) / block.x);
  rope_batched_positions_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      static_cast<const uint32_t*>(positions),
      batch_count,
      seq_len,
      heads,
      head_dim,
      base,
      scale,
      split_half);
  return 0;
}

extern "C" int hi_cuda_launch_mrope(
    void* values,
    const void* pos_t,
    const void* pos_h,
    const void* pos_w,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    float scale,
    int section_t,
    int section_h,
    int section_w,
    int section_e,
    int split_half,
    void* stream) {
  int section_sum = section_t + section_h + section_w + section_e;
  if (values == nullptr || pos_t == nullptr || pos_h == nullptr ||
      pos_w == nullptr || seq_len < 0 || heads <= 0 || head_dim <= 0 ||
      head_dim % 2 != 0 || base <= 0.0f || section_t < 0 ||
      section_h < 0 || section_w < 0 || section_e < 0 ||
      section_sum <= 0 || section_sum > head_dim / 2 || stream == nullptr) {
    return 1;
  }
  int total_pairs = seq_len * heads * (head_dim / 2);
  if (total_pairs == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total_pairs + block.x - 1) / block.x);
  mrope_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      static_cast<const uint32_t*>(pos_t),
      static_cast<const uint32_t*>(pos_h),
      static_cast<const uint32_t*>(pos_w),
      seq_len,
      heads,
      head_dim,
      base,
      scale,
      section_t,
      section_h,
      section_w,
      section_e,
      split_half);
  return 0;
}

extern "C" int hi_cuda_launch_write_kv_cache(
    const void* values,
    void* cache,
    int row_count,
    int kv_heads,
    int head_dim,
    int max_seq,
    int start_pos,
    void* stream) {
  if (values == nullptr || cache == nullptr || row_count < 0 || kv_heads <= 0 ||
      head_dim <= 0 || max_seq <= 0 || start_pos < 0 ||
      start_pos + row_count > max_seq || stream == nullptr) {
    return 1;
  }
  int total = row_count * kv_heads * head_dim;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  write_kv_cache_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(values),
      static_cast<float*>(cache),
      row_count,
      kv_heads,
      head_dim,
      max_seq,
      start_pos);
  return 0;
}

extern "C" int hi_cuda_launch_write_kv_cache_batched(
    const void* values,
    void* cache,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int max_seq,
    int start_pos,
    void* stream) {
  if (values == nullptr || cache == nullptr || batch_count <= 0 ||
      row_count <= 0 || kv_heads <= 0 || head_dim <= 0 || max_seq <= 0 ||
      start_pos < 0 || start_pos + row_count > max_seq || stream == nullptr) {
    return 1;
  }
  int total = batch_count * row_count * kv_heads * head_dim;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  write_kv_cache_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(values),
      static_cast<float*>(cache),
      batch_count,
      row_count,
      kv_heads,
      head_dim,
      max_seq,
      start_pos);
  return 0;
}

extern "C" int hi_cuda_launch_write_paged_kv_cache(
    const void* values,
    void* pages,
    const void* page_table,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    int start_pos,
    void* stream) {
  if (values == nullptr || pages == nullptr || page_table == nullptr ||
      row_count < 0 || kv_heads <= 0 || head_dim <= 0 || page_size <= 0 ||
      page_table_len <= 0 || start_pos < 0 ||
      start_pos + row_count > page_size * page_table_len || stream == nullptr) {
    return 1;
  }
  int total = row_count * kv_heads * head_dim;
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  write_paged_kv_cache_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(values),
      static_cast<kv_t*>(pages),
      static_cast<const uint32_t*>(page_table),
      row_count,
      kv_heads,
      head_dim,
      page_size,
      page_table_len,
      start_pos);
  return 0;
}

extern "C" int hi_cuda_launch_write_paged_kv_cache_batched(
    const void* values,
    void* pages,
    const void* page_table,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    int start_pos,
    void* stream) {
  if (values == nullptr || pages == nullptr || page_table == nullptr ||
      batch_count <= 0 || row_count <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      page_size <= 0 || page_table_len <= 0 || start_pos < 0 ||
      start_pos + row_count > page_size * page_table_len || stream == nullptr) {
    return 1;
  }
  int total = batch_count * row_count * kv_heads * head_dim;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  write_paged_kv_cache_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(values),
      static_cast<kv_t*>(pages),
      static_cast<const uint32_t*>(page_table),
      batch_count,
      row_count,
      kv_heads,
      head_dim,
      page_size,
      page_table_len,
      start_pos);
  return 0;
}

extern "C" int hi_cuda_launch_write_paged_kv_cache_batched_positions(
    const void* values,
    void* pages,
    const void* page_table,
    const void* positions,
    int batch_count,
    int row_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    void* stream) {
  if (values == nullptr || pages == nullptr || page_table == nullptr ||
      positions == nullptr || batch_count <= 0 || row_count <= 0 ||
      kv_heads <= 0 || head_dim <= 0 || page_size <= 0 ||
      page_table_len <= 0 || stream == nullptr) {
    return 1;
  }
  int total = batch_count * row_count * kv_heads * head_dim;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  write_paged_kv_cache_batched_positions_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(values),
      static_cast<kv_t*>(pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<const uint32_t*>(positions),
      batch_count,
      row_count,
      kv_heads,
      head_dim,
      page_size,
      page_table_len);
  return 0;
}

extern "C" int hi_cuda_launch_copy_paged_kv_cache_prefix_batched(
    void* pages,
    const void* page_table,
    int batch_count,
    int token_count,
    int kv_heads,
    int head_dim,
    int page_size,
    int page_table_len,
    void* stream) {
  if (pages == nullptr || page_table == nullptr || batch_count <= 1 ||
      token_count <= 0 || kv_heads <= 0 || head_dim <= 0 || page_size <= 0 ||
      page_table_len <= 0 || token_count > page_size * page_table_len ||
      stream == nullptr) {
    return 1;
  }
  int total = (batch_count - 1) * token_count * kv_heads * head_dim;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  copy_paged_kv_cache_prefix_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<kv_t*>(pages),
      static_cast<const uint32_t*>(page_table),
      batch_count,
      token_count,
      kv_heads,
      head_dim,
      page_size,
      page_table_len);
  return 0;
}

extern "C" int hi_cuda_launch_causal_attention(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      seq_len < 0 || heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 ||
      v_head_dim <= 0 ||
      heads % kv_heads != 0 || stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads;
  if (total == 0) {
    return 0;
  }
  dim3 block(128);
  dim3 grid((total + block.x - 1) / block.x);
  causal_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_causal_attention_batched(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch_count <= 0 || seq_len <= 0 || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * seq_len * heads + block.x - 1) / block.x);
  causal_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      batch_count,
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_flash_causal_attention(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      seq_len < 0 || heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 ||
      v_head_dim <= 0 || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads;
  if (total == 0) {
    return 0;
  }
  dim3 block(128);
  dim3 grid((total + block.x - 1) / block.x);
  flash_causal_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_tiled_causal_attention(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      seq_len < 0 || heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 ||
      v_head_dim <= 0 || qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads;
  if (total == 0) {
    return 0;
  }
  dim3 block(32);
  dim3 grid(total);
  tiled_causal_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim,
      window);
  return 0;
}

extern "C" int hi_cuda_launch_flash_causal_attention_batched(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch_count <= 0 || seq_len <= 0 || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * seq_len * heads + block.x - 1) / block.x);
  flash_causal_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      batch_count,
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_tiled_causal_attention_batched(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int batch_count,
    int seq_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batch_count <= 0 || seq_len <= 0 || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(32);
  dim3 grid(batch_count * seq_len * heads);
  tiled_causal_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      batch_count,
      seq_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim,
      window);
  return 0;
}

extern "C" int hi_cuda_launch_full_attention(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int seq_len,
    int heads,
    int kv_heads,
    int head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      seq_len < 0 || heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      heads % kv_heads != 0 || stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads;
  if (total == 0) {
    return 0;
  }
  dim3 block(128);
  dim3 grid((total + block.x - 1) / block.x);
  full_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<float*>(output),
      seq_len,
      heads,
      kv_heads,
      head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_window_attention(
    const void* q,
    const void* k,
    const void* v,
    const void* window_start,
    const void* window_end,
    void* output,
    int seq_len,
    int heads,
    int kv_heads,
    int head_dim,
    void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || window_start == nullptr ||
      window_end == nullptr || output == nullptr || seq_len < 0 || heads <= 0 ||
      kv_heads <= 0 || head_dim <= 0 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads;
  if (total == 0) {
    return 0;
  }
  dim3 block(128);
  dim3 grid((total + block.x - 1) / block.x);
  window_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k),
      static_cast<const float*>(v),
      static_cast<const uint32_t*>(window_start),
      static_cast<const uint32_t*>(window_end),
      static_cast<float*>(output),
      seq_len,
      heads,
      kv_heads,
      head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_vision_rope(
    void* values,
    const void* pos_h,
    const void* pos_w,
    int seq_len,
    int heads,
    int head_dim,
    float base,
    void* stream) {
  if (values == nullptr || pos_h == nullptr || pos_w == nullptr ||
      seq_len < 0 || heads <= 0 || head_dim <= 0 || head_dim % 4 != 0 ||
      !isfinite(base) || base <= 0.0f || stream == nullptr) {
    return 1;
  }
  int total = seq_len * heads * (head_dim / 2);
  if (total == 0) {
    return 0;
  }
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  vision_rope_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<float*>(values),
      static_cast<const uint32_t*>(pos_h),
      static_cast<const uint32_t*>(pos_w),
      seq_len,
      heads,
      head_dim,
      base);
  return 0;
}

extern "C" int hi_cuda_launch_cached_decode_attention(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* output,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      output == nullptr || position < 0 || max_seq <= 0 || position >= max_seq ||
      heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
      heads % kv_heads != 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((heads + block.x - 1) / block.x);
  cached_decode_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k_cache),
      static_cast<const float*>(v_cache),
      static_cast<float*>(output),
      position,
      max_seq,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_flash_cached_decode_attention(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* output,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      output == nullptr || position < 0 || max_seq <= 0 || position >= max_seq ||
      heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(HI_CUDA_DECODE_WARPS * 32);
  dim3 grid(heads);
  size_t shared_bytes =
      (2 * HI_CUDA_DECODE_WARPS + HI_CUDA_DECODE_WARPS * v_head_dim) * sizeof(float);
  flash_cached_decode_attention_kernel<<<grid, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k_cache),
      static_cast<const float*>(v_cache),
      static_cast<float*>(output),
      position,
      max_seq,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_paged_decode_attention(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || position < 0 ||
      page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((heads + block.x - 1) / block.x);
  paged_decode_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_flash_paged_decode_attention(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || position < 0 ||
      page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((heads + block.x - 1) / block.x);
  flash_paged_decode_attention_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_tiled_paged_decode_attention(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || position < 0 ||
      page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(HI_CUDA_DECODE_WARPS * 32);
  dim3 grid(heads);
  size_t shared_bytes =
      (2 * HI_CUDA_DECODE_WARPS + HI_CUDA_DECODE_WARPS * v_head_dim) * sizeof(float);
  tiled_paged_decode_attention_kernel<<<grid, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim,
      window);
  return 0;
}

extern "C" int hi_cuda_launch_paged_decode_attention_batched(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || batch_count <= 0 ||
      position < 0 || page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * heads + block.x - 1) / block.x);
  paged_decode_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      batch_count,
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_flash_paged_decode_attention_batched(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || batch_count <= 0 ||
      position < 0 || page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * heads + block.x - 1) / block.x);
  flash_paged_decode_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      batch_count,
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_paged_decode_attention_batched_positions(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    const void* positions,
    void* output,
    int batch_count,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || positions == nullptr || output == nullptr ||
      batch_count <= 0 || page_size <= 0 || page_table_len <= 0 ||
      heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
      heads % kv_heads != 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * heads + block.x - 1) / block.x);
  paged_decode_attention_batched_positions_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<const uint32_t*>(positions),
      static_cast<float*>(output),
      batch_count,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_cached_decode_attention_batched(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* output,
    int batch_count,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      output == nullptr || batch_count <= 0 || position < 0 || max_seq <= 0 ||
      position >= max_seq || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(128);
  dim3 grid((batch_count * heads + block.x - 1) / block.x);
  cached_decode_attention_batched_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k_cache),
      static_cast<const float*>(v_cache),
      static_cast<float*>(output),
      batch_count,
      position,
      max_seq,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_tiled_paged_decode_attention_batched(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    void* output,
    int batch_count,
    int position,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || output == nullptr || batch_count <= 0 ||
      position < 0 || page_size <= 0 || page_table_len <= 0 ||
      position >= page_size * page_table_len || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(HI_CUDA_DECODE_WARPS * 32);
  dim3 grid(batch_count * heads);
  size_t shared_bytes =
      (2 * HI_CUDA_DECODE_WARPS + HI_CUDA_DECODE_WARPS * v_head_dim) * sizeof(float);
  tiled_paged_decode_attention_batched_kernel<<<grid, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<float*>(output),
      batch_count,
      position,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
	      v_head_dim,
      window);
	  return 0;
	}

extern "C" int hi_cuda_launch_tiled_paged_decode_attention_batched_positions(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* page_table,
    const void* positions,
    void* output,
    int batch_count,
    int page_size,
    int page_table_len,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    int window,
    void* stream) {
  if (q == nullptr || k_pages == nullptr || v_pages == nullptr ||
      page_table == nullptr || positions == nullptr || output == nullptr ||
      batch_count <= 0 || page_size <= 0 || page_table_len <= 0 ||
      heads <= 0 || kv_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(HI_CUDA_DECODE_WARPS * 32);
  dim3 grid(batch_count * heads);
  size_t shared_bytes =
      (2 * HI_CUDA_DECODE_WARPS + HI_CUDA_DECODE_WARPS * v_head_dim) * sizeof(float);
  tiled_paged_decode_attention_batched_positions_kernel<<<grid, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const kv_t*>(k_pages),
      static_cast<const kv_t*>(v_pages),
      static_cast<const uint32_t*>(page_table),
      static_cast<const uint32_t*>(positions),
      static_cast<float*>(output),
      batch_count,
      page_size,
      page_table_len,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim,
      window);
  return 0;
}

extern "C" int hi_cuda_launch_flash_cached_decode_attention_batched(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* output,
    int batch_count,
    int position,
    int max_seq,
    int heads,
    int kv_heads,
    int qk_head_dim,
    int v_head_dim,
    void* stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      output == nullptr || batch_count <= 0 || position < 0 || max_seq <= 0 ||
      position >= max_seq || heads <= 0 || kv_heads <= 0 ||
      qk_head_dim <= 0 || v_head_dim <= 0 ||
      qk_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM ||
      v_head_dim > HI_CUDA_FLASH_MAX_HEAD_DIM || heads % kv_heads != 0 ||
      stream == nullptr) {
    return 1;
  }
  dim3 block(HI_CUDA_DECODE_WARPS * 32);
  dim3 grid(batch_count * heads);
  size_t shared_bytes =
      (2 * HI_CUDA_DECODE_WARPS + HI_CUDA_DECODE_WARPS * v_head_dim) * sizeof(float);
  flash_cached_decode_attention_batched_kernel<<<grid, block, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(q),
      static_cast<const float*>(k_cache),
      static_cast<const float*>(v_cache),
      static_cast<float*>(output),
      batch_count,
      position,
      max_seq,
      heads,
      kv_heads,
      qk_head_dim,
      v_head_dim);
  return 0;
}

extern "C" int hi_cuda_launch_argmax(
    const void* logits,
    void* output_token,
    int len,
    void* stream) {
  if (logits == nullptr || output_token == nullptr || len <= 0 ||
      stream == nullptr) {
    return 1;
  }
  argmax_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<uint32_t*>(output_token),
      len);
  return 0;
}

extern "C" int hi_cuda_launch_argmax_last_row(
    const void* logits,
    void* output_token,
    int rows,
    int cols,
    void* stream) {
  if (logits == nullptr || output_token == nullptr || rows <= 0 || cols <= 0 ||
      stream == nullptr) {
    return 1;
  }
  argmax_last_row_kernel<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<uint32_t*>(output_token),
      rows,
      cols);
  return 0;
}

extern "C" int hi_cuda_launch_argmax_batched_last_token(
    const void* logits,
    void* output_tokens,
    int batch_count,
    int seq_len,
    int cols,
    void* stream) {
  if (logits == nullptr || output_tokens == nullptr || batch_count <= 0 ||
      seq_len <= 0 || cols <= 0 || stream == nullptr) {
    return 1;
  }
  dim3 block(256);
  dim3 grid(batch_count);
  argmax_batched_last_token_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<uint32_t*>(output_tokens),
      batch_count,
      seq_len,
      cols);
  return 0;
}

extern "C" int hi_cuda_launch_sample_last_row(
    const void* logits,
    void* output_token,
    int rows,
    int cols,
    float temperature,
    float top_p,
    int top_k,
    float sample,
    void* stream) {
  if (logits == nullptr || output_token == nullptr || rows <= 0 || cols <= 0 ||
      top_k < 0 || stream == nullptr) {
    return 1;
  }
  sample_last_row_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<uint32_t*>(output_token),
      rows,
      cols,
      temperature,
      top_p,
      top_k,
      sample);
  return 0;
}

extern "C" int hi_cuda_launch_sample_batched_last_token(
    const void* logits,
    void* output_tokens,
    const void* samples,
    int batch_count,
    int seq_len,
    int cols,
    float temperature,
    float top_p,
    int top_k,
    void* stream) {
  if (logits == nullptr || output_tokens == nullptr || samples == nullptr ||
      batch_count <= 0 || seq_len <= 0 || cols <= 0 || top_k < 0 ||
      stream == nullptr) {
    return 1;
  }
  sample_batched_last_token_kernel<<<batch_count, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(logits),
      static_cast<uint32_t*>(output_tokens),
      static_cast<const float*>(samples),
      batch_count,
      seq_len,
      cols,
      temperature,
      top_p,
      top_k);
  return 0;
}
