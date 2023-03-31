// Copyright © 2023 Roman Kuznetsov.
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

#include "glyph_texture.hpp"

#include <algorithm>
#include <limits>

#include "common/utils.hpp"
#include "sdf_text_types.h"

namespace sdf::gpu {
namespace {
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
uint32_t nextPowerOf2(uint32_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
}  // namespace

// static
MTL::Texture * GlyphTexture::generate(MTL::Device * const device,
                                      MTL::CommandQueue * const commandQueue,
                                      MTL::Library * library,
                                      GlyphSet const & glyphSet) {
  // Auto-release pool for temporary objects.
  NS::AutoreleasePool * autoreleasePool = NS::AutoreleasePool::alloc()->init();
  METAL_GUARD(autoreleasePool);

  // Initialize shaders.
  MTL::FunctionConstantValues * constantValues = MTL::FunctionConstantValues::alloc()->init();
  METAL_GUARD(constantValues);

  NS::Error * error = nullptr;
  MTL::Function * sdfGenerateFunction =
    library->newFunction(STR("sdfGenerate"), constantValues, &error);
  CHECK_AND_RETURN(error, nullptr);
  METAL_GUARD(sdfGenerateFunction);

  MTL::Function * sdfWriteTextureFunction =
    library->newFunction(STR("sdfWriteTexture"), constantValues, &error);
  CHECK_AND_RETURN(error, nullptr);
  METAL_GUARD(sdfWriteTextureFunction);

  // Create compute pipeline states.
  auto sdfGeneratePipelineStateDescriptor = MTL::ComputePipelineDescriptor::alloc()->init();
  METAL_GUARD(sdfGeneratePipelineStateDescriptor);
  sdfGeneratePipelineStateDescriptor->setComputeFunction(sdfGenerateFunction);
  sdfGeneratePipelineStateDescriptor->setSupportIndirectCommandBuffers(true);

  MTL::ComputePipelineState * sdfGeneratePipelineState =
    device->newComputePipelineState(sdfGeneratePipelineStateDescriptor,
                                    MTL::PipelineOptionNone,
                                    nullptr,
                                    &error);
  CHECK_AND_RETURN(error, nullptr);
  METAL_GUARD(sdfGeneratePipelineState);

  MTL::ComputePipelineState * sdfWriteTexturePipelineState =
    device->newComputePipelineState(sdfWriteTextureFunction, &error);
  CHECK_AND_RETURN(error, nullptr);
  METAL_GUARD(sdfWriteTexturePipelineState);

  auto const & glyphs = glyphSet.getGlyphs();

  // Calculate line buffer size and offsets.
  uint32_t linesBufferSize = 0;
  std::unordered_map<uint16_t, uint32_t> lineOffsets;
  for (auto const & [glyph, glyphData] : glyphs) {
    if (glyphData.m_lines.empty()) {
      continue;
    }
    lineOffsets[glyph] = linesBufferSize;
    METAL_ASSERT(glyphData.m_lines.size() < std::numeric_limits<uint32_t>::max() - linesBufferSize);
    linesBufferSize += static_cast<uint32_t>(glyphData.m_lines.size());
  }

  // Return default 1x1 black texture.
  if (linesBufferSize == 0) {
    MTL::TextureDescriptor * descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormatR8Unorm);
    descriptor->setWidth(1);
    descriptor->setHeight(1);
    descriptor->setMipmapLevelCount(1);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    METAL_GUARD(descriptor);

    MTL::Texture * t = device->newTexture(descriptor);
    uint8_t val = 0;
    t->replaceRegion(MTL::Region::Make2D(0, 0, 1, 1), 0, 0, &val, 1, 1);
    return t;
  }

  // Create and fill lines buffer.
  METAL_ASSERT(linesBufferSize < std::numeric_limits<uint32_t>::max() / sizeof(glm::vec4));
  MTL::Buffer * linesBuffer =
    device->newBuffer(linesBufferSize * sizeof(glm::vec4), MTL::ResourceStorageModeShared);
  METAL_GUARD(linesBuffer);

  auto contentPtr = static_cast<uint8_t *>(linesBuffer->contents());
  for (auto const & [glyph, glyphData] : glyphs) {
    if (glyphData.m_lines.empty()) {
      continue;
    }
    memcpy(contentPtr + lineOffsets[glyph] * sizeof(glm::vec4),
           glyphData.m_lines.data(),
           static_cast<uint32_t>(glyphData.m_lines.size() * sizeof(glm::vec4)));
  }

  // Initialize output buffers.
  auto const & atlasSize = glyphSet.getAtlasSize();
  auto const outBufferSize = atlasSize.x * atlasSize.y;
  MTL::Buffer * outMinDistance =
    device->newBuffer(outBufferSize * sizeof(int), MTL::ResourceStorageModeShared);
  METAL_GUARD(outMinDistance);

  auto outMinDistanceContentPtr = static_cast<int *>(outMinDistance->contents());

  MTL::Buffer * outIntersectionNumber =
    device->newBuffer(outBufferSize * sizeof(uint32_t), MTL::ResourceStorageModeShared);
  METAL_GUARD(outIntersectionNumber);

  auto outIntersectionNumberContentPtr = static_cast<uint32_t *>(outIntersectionNumber->contents());
  for (uint32_t i = 0; i < outBufferSize; ++i) {
    outMinDistanceContentPtr[i] = std::numeric_limits<int>::max();
    outIntersectionNumberContentPtr[i] = 0;
  }

  // Initialize textures over output buffers.
  MTL::TextureDescriptor * descriptor = MTL::TextureDescriptor::alloc()->init();
  descriptor->setTextureType(MTL::TextureType2D);
  descriptor->setPixelFormat(MTL::PixelFormatR32Sint);
  descriptor->setWidth(atlasSize.x);
  descriptor->setHeight(atlasSize.y);
  descriptor->setMipmapLevelCount(1);
  descriptor->setStorageMode(MTL::StorageModeShared);
  descriptor->setUsage(MTL::TextureUsageShaderRead);
  METAL_GUARD(descriptor);

  MTL::Texture * minDistanceTexture =
    outMinDistance->newTexture(descriptor, 0, atlasSize.x * sizeof(int));
  METAL_GUARD(minDistanceTexture);

  descriptor->setPixelFormat(MTL::PixelFormatR32Uint);
  MTL::Texture * intersectionNumberTexture =
    outIntersectionNumber->newTexture(descriptor, 0, atlasSize.x * sizeof(uint32_t));
  METAL_GUARD(intersectionNumberTexture);

  // Initialize output texture.
  descriptor->setPixelFormat(MTL::PixelFormatR8Unorm);
  descriptor->setStorageMode(MTL::StorageModePrivate);
  descriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
  MTL::Texture * outputTexture = device->newTexture(descriptor);
  outputTexture->setLabel(STR("SDF Glyphs Texture"));

  auto const simdGroupSize =
    static_cast<uint32_t>(sdfGeneratePipelineState->threadExecutionWidth());
  auto const maxThreadsInGroup =
    static_cast<uint32_t>(sdfGeneratePipelineState->maxTotalThreadsPerThreadgroup());

  // Thread group memory size (must be a multiple of 16 bytes).
  // Please check shader, we need one float per SIMD-group in the thread group
  // memory.
  auto const maxSimdInThreadGroup = maxThreadsInGroup / simdGroupSize;
  auto const minDistThreadGroupMemorySize =
    utils::getAligned(maxSimdInThreadGroup * sizeof(float), 16);
  auto const iNumThreadGroupMemorySize =
    utils::getAligned(maxSimdInThreadGroup * sizeof(uint32_t), 16);

  auto commandBuffer = commandQueue->commandBuffer();
  auto encoder = commandBuffer->computeCommandEncoder();
  encoder->setLabel(STR("SDF Texture Generation Command Encoder"));

  // Fill indirect buffers.
  uint32_t indirectBufferSize = 0;
  for (auto const & [_, glyphData] : glyphs) {
    if (glyphData.m_lines.empty()) {
      continue;
    }
    indirectBufferSize += glyphData.m_pixelSize.x * glyphData.m_pixelSize.y;
  }
  if (indirectBufferSize == 0) {
    return outputTexture;
  }

  auto icbDescriptor = MTL::IndirectCommandBufferDescriptor::alloc()->init();
  icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeConcurrentDispatchThreads);
  icbDescriptor->setInheritBuffers(false);
  icbDescriptor->setInheritPipelineState(true);
  icbDescriptor->setMaxKernelBufferBindCount(4);
  METAL_GUARD(icbDescriptor);

  auto icb = device->newIndirectCommandBuffer(icbDescriptor, indirectBufferSize, 0);
  METAL_GUARD(icb);

  MTL::Buffer * paramsBuffer =
    device->newBuffer(indirectBufferSize * sizeof(SdfGenParams), MTL::ResourceStorageModeShared);
  METAL_GUARD(paramsBuffer);
  auto paramsBufferPtr = static_cast<SdfGenParams *>(paramsBuffer->contents());

  uint32_t indirectBufferIndex = 0;
  for (auto const & [glyph, glyphData] : glyphs) {
    if (glyphData.m_lines.empty()) {
      continue;
    }
    for (uint32_t j = 0; j < glyphData.m_pixelSize.y; ++j) {
      for (uint32_t i = 0; i < glyphData.m_pixelSize.x; ++i) {
        auto const x = (glyphData.m_posInAtlas.x + i);
        auto const y = (glyphData.m_posInAtlas.y + j);
        auto const offset = y * atlasSize.x + x;

        SdfGenParams params;
        params.pointPos.x = static_cast<float>(i) + 0.5f;
        params.pointPos.y = static_cast<float>(j) + 0.5f;
        params.linesCount = static_cast<uint32_t>(glyphData.m_lines.size());
        params.lineBufferOffset = lineOffsets[glyph];
        memcpy(&paramsBufferPtr[indirectBufferIndex], &params, sizeof(params));

        auto icbCommand = icb->indirectComputeCommand(indirectBufferIndex);
        icbCommand->setKernelBuffer(linesBuffer, 0, SdfGenBufferLines);
        icbCommand->setKernelBuffer(outMinDistance, offset * sizeof(int), SdfGenBufferMinDistance);
        icbCommand->setKernelBuffer(outIntersectionNumber,
                                    offset * sizeof(uint32_t),
                                    SdfGenBufferIntersectionNumber);
        icbCommand->setKernelBuffer(paramsBuffer,
                                    indirectBufferIndex * sizeof(SdfGenParams),
                                    SdfGenBufferParams);

        icbCommand->setThreadgroupMemoryLength(minDistThreadGroupMemorySize,
                                               SdfGenSharedMemoryMinDistance);
        icbCommand->setThreadgroupMemoryLength(iNumThreadGroupMemorySize,
                                               SdfGenSharedMemoryIntersectionNumber);

        // We do the first stage of reduction on load, so we need up to 2x less
        // threads.
        auto const threadsCount =
          std::max(nextPowerOf2(static_cast<uint32_t>(glyphData.m_lines.size()) / 2),
                   simdGroupSize);
        auto const threadsInGroup =
          std::min(utils::getAligned(threadsCount, simdGroupSize), maxThreadsInGroup);

        icbCommand->concurrentDispatchThreads(MTL::Size::Make(threadsCount, 1, 1),
                                              MTL::Size::Make(threadsInGroup, 1, 1));
        indirectBufferIndex++;
      }
    }
  }

  // Run compute shaders for SDF generation.
  encoder->setComputePipelineState(sdfGeneratePipelineState);
  encoder->useResource(linesBuffer, MTL::ResourceUsageRead);
  encoder->useResource(paramsBuffer, MTL::ResourceUsageRead);
  encoder->useResource(outMinDistance, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  encoder->useResource(outIntersectionNumber, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);

  uint32_t constexpr kMaxCommands = 8192;
  for (uint32_t start = 0; start < indirectBufferSize; start += kMaxCommands) {
    encoder->executeCommandsInBuffer(
      icb,
      NS::Range::Make(start, std::min(indirectBufferSize - start, kMaxCommands)));
  }

  // Run compute shader to write output SDF texture.
  encoder->setComputePipelineState(sdfWriteTexturePipelineState);
  encoder->setTexture(minDistanceTexture, SdfTextureInMinDistance);
  encoder->setTexture(intersectionNumberTexture, SdfTextureInIntersectionNumber);
  encoder->setTexture(outputTexture, SdfTextureOut);
  encoder->dispatchThreads(MTL::Size::Make(atlasSize.x, atlasSize.y, 1), MTL::Size::Make(8, 8, 1));

  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  return outputTexture;
}
}  // namespace sdf::gpu
