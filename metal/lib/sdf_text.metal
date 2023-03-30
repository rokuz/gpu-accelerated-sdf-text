// Copyright © 2022 Roman Kuznetsov.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <metal_stdlib>

using namespace metal;

#include "sdf_text_types.h"

// Calculates minimal distance between a point `pt` and a line [`from`; `to`].
float calculateMinDistance(float2 from, float2 to, float2 pt) {
  float2 v = to - from;
  
  float2 v1 = pt - from;
  float d1 = dot(v, v1);
  if (d1 < 0) {
    return length(v1);
  }
  
  float2 v2 = pt - to;
  float d2 = dot(v, v2);
  if (d2 > 0) {
    return length(v2);
  }
  
  return abs(v1.y * v.x - v1.x * v.y) / length(v);
}

// Returns 1 if there is an intersection between a ray and a line [`from`; `to`].
uint getIntersection(float2 rayOrigin, float2 rayDir, float2 from, float2 to) {
  float2 v = to - from;
  float2 v1 = float2(-rayDir.y, rayDir.x);
  float d = dot(v, v1);
  // A ray and a line are collinear, no intersection between them.
  if (d == 0.0) {
    return 0;
  }
  
  float2 v2 = rayOrigin - from;
  float t1 = (v2.y * v.x - v2.x * v.y) / d;
  if (t1 < 0.0) {
    return 0;
  }
  
  float t2 = dot(v1, v2) / d;
  return (t2 >= 0.0 && t2 < 1.0) ? 1 : 0;
}

constant float kFloatScalar = 100.0;

// Kernel for calculation the distance from some point to the closest glyph's outline
// and number of intersections between a ray emitted from some point and the glyph's outline.
// The kernel is executed for every pixel of SDF gliph, aformentioned point is a center of pixel.
kernel void sdfGenerate(
  // Generation parameters.
  device SdfGenParams & params [[buffer(SdfGenBufferParams)]],
                        
  // List of lines from which glyph's outline consists of.
  device Line * lines [[buffer(SdfGenBufferLines)]],
                        
  // Threadgroup (shared) memory to implement parallel reduction.
  // Number of elements in each is equal to number of threads in SIMD group (simdSize).
  threadgroup float * minDistance [[threadgroup(SdfGenSharedMemoryMinDistance)]],
  threadgroup uint * intersectionNumber [[threadgroup(SdfGenSharedMemoryIntersectionNumber)]],
                        
  // Output.
  device atomic_int * outMinDistance [[buffer(SdfGenBufferMinDistance)]],
  device atomic_uint * outIntersectionNumber [[buffer(SdfGenBufferIntersectionNumber)]],
                        
  // Thread indices.
  uint gid [[thread_position_in_grid]],
  uint threadGroupSize [[threads_per_threadgroup]],
  uint threadId [[thread_position_in_threadgroup]],
  uint threadGroupId [[threadgroup_position_in_grid]],
  uint simdSize [[threads_per_simdgroup]],
  uint simdLaneId [[thread_index_in_simdgroup]],
  uint simdGroupId [[simdgroup_index_in_threadgroup]]
) {
  // We do the first reduction step on load.
  // Overall thread number is equal to max(nextPowerOf2(linesCount / 2), simdSize).
  // Every thread takes i-th line and (i + threadGroupSize)-th line if it exists.
  uint i = threadGroupId * (threadGroupSize * 2) + threadId + params.lineBufferOffset;
  
  float minDist = 1000000.0; // very big (unreachable) value.
  uint iNum = 0;
  
  // Useful threads number can be less than simdSize.
  if (i < params.lineBufferOffset + params.linesCount) {
    minDist = calculateMinDistance(lines[i].from, lines[i].to, params.pointPos);
    // NOTE: ray direction can be arbitrary.
    // For a real graphics engine it's worth to optimize the math in getIntersection
    // using the fact one of component in ray direction is 0.
    iNum = getIntersection(params.pointPos, float2(1, 0), lines[i].from, lines[i].to);
  }
  
  uint i2 = i + threadGroupSize;
  if (i2 < params.lineBufferOffset + params.linesCount) {
    float d = calculateMinDistance(lines[i2].from, lines[i2].to, params.pointPos);
    minDist = min(minDist, d);
    iNum += getIntersection(params.pointPos, float2(1, 0), lines[i2].from, lines[i2].to);
  }
  
  // Wait for completion of on-load reduction.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  for (uint s = threadGroupSize / simdSize; s > 1; s /= simdSize) {
    // Perform per-SIMD partial reduction.
    minDist = simd_min(minDist);
    iNum = simd_sum(iNum);

    // Write per-SIMD partial reduction value to threadgroup memory.
    if (simdLaneId == 0) {
      minDistance[simdGroupId] = minDist;
      intersectionNumber[simdGroupId] = iNum;
    }

    // Wait for all partial reductions to complete.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Set 0 for sum reduction and very big value (1000000.0) for min reduction.
    // It allow us to remove influence of unused threads in further calculations.
    minDist = (threadId < s) ? minDistance[threadId] : 1000000.0;
    iNum = (threadId < s) ? intersectionNumber[threadId] : 0;
  }
  
  // Perform final per-SIMD partial reduction to calculate
  // the threadgroup partial reduction result.
  minDist = simd_min(minDist);
  iNum = simd_sum(iNum);
  
  // Atomically update the reduction result.
  if (threadId == 0) {
    // NOTE: atomic_float in Metal 3 support only add and sub operations.
    atomic_fetch_min_explicit(outMinDistance, int(minDist * kFloatScalar), memory_order_relaxed);
    atomic_fetch_add_explicit(outIntersectionNumber, iNum, memory_order_relaxed);
  }
}

// Kernel for SDF texture generation.
kernel void sdfWriteTexture(
  texture2d<int, access::read> inMinDistance [[texture(SdfTextureInMinDistance)]],
  texture2d<uint, access::read> inIntersectionNumber [[texture(SdfTextureInIntersectionNumber)]],
  texture2d<float, access::write> outTexture [[texture(SdfTextureOut)]],
  uint2 gid [[thread_position_in_grid]]
) {
  // Check if the pixel is within the bounds of the output texture.
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
    return;
  }
  
  float minDist = float(inMinDistance.read(gid).r) / kFloatScalar;

  // Distances inside glyph are negative. Odd number of intersections defines pixels inside gliph.
  uint iNum = inIntersectionNumber.read(gid).r;
  if (iNum % 2 != 0) {
    minDist = -minDist;
  }

  // Normalize before writing to the texture. Glyph's outline will be 0.75 in the texture.
  float constexpr kMinRange = -10.0;
  float constexpr kMaxRange = 30.0;
  float v = 1.0 - (clamp(minDist, kMinRange, kMaxRange) - kMinRange) / (kMaxRange - kMinRange);

  outTexture.write(float4(v, v, v, 1.0), gid);
}

struct FragmentInputText
{
  float4 position [[position]];
  float4 color;
  float2 uv;
};

constant float2 verticesQuad[] = {
  float2(-1, 1),
  float2(1, 1),
  float2(-1, -1),
  float2(1, -1)
};

vertex FragmentInputText vertexText(uint vertexID [[vertex_id]],
                                    uint instanceId [[instance_id]],
                                    constant FrameData & frameData [[buffer(TextRenderBufferFrame)]],
                                    const device Glyph * glyphs [[buffer(TextRenderBufferGlyphs)]]) {
  FragmentInputText out;
  Glyph g = glyphs[instanceId];
  out.position = frameData.projection * float4(verticesQuad[vertexID] * g.halfSize + g.center, 0.0, 1.0);
  out.color = g.color;
  out.uv = float2(1.0, -1.0) * verticesQuad[vertexID] * g.uvHalfSize + g.uvCenter;
  return out;
}

constexpr sampler kLinearSampler(filter::linear);

fragment float4 fragmentText(FragmentInputText in [[stage_in]],
                             texture2d<float> glyphTex [[texture(TextRenderTextureGlyphs)]]) {
  float4 out;
  float dist = glyphTex.sample(kLinearSampler, in.uv).r;
  float edgeWidth = length(float2(dfdx(dist), dfdy(dist)));
  float alpha = smoothstep(0.75 - edgeWidth, 0.75 + edgeWidth, dist);
  out = float4(in.color.rgb, in.color.a * alpha);
  return out;
}
