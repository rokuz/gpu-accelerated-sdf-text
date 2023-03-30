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

#include "text_renderer.hpp"

#include <algorithm>

#include "common/utils.hpp"

namespace sdf::gpu {

uint32_t constexpr kGlyphBufferDefaultSize = 1000;

TextRenderer::~TextRenderer() {
  if (m_glyphBuffer) {
    m_glyphBuffer->release();
  }

  if (m_pipelineState) {
    m_pipelineState->release();
  }
}

bool TextRenderer::initialize(MTL::Device *device, MTL::Library *library) {
  // Initialize glyph buffer.
  m_glyphBufferSize = kGlyphBufferDefaultSize;
  m_glyphBuffer = device->newBuffer(m_glyphBufferSize * sizeof(Glyph),
                                    MTL::ResourceStorageModeShared);

  // Initialize shaders.
  MTL::FunctionConstantValues *constantValues =
      MTL::FunctionConstantValues::alloc()->init();
  METAL_GUARD(constantValues);

  NS::Error *error = nullptr;
  MTL::Function *vsFunction =
      library->newFunction(STR("vertexText"), constantValues, &error);
  CHECK_AND_RETURN(error, false);
  METAL_GUARD(vsFunction);

  MTL::Function *fsFunction =
      library->newFunction(STR("fragmentText"), constantValues, &error);
  CHECK_AND_RETURN(error, false);
  METAL_GUARD(fsFunction);

  // Initialize pipeline state.
  auto pipelineStateDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
  pipelineStateDescriptor->setLabel(STR("Text Render Pipeline State"));
  METAL_GUARD(pipelineStateDescriptor);
  pipelineStateDescriptor->setVertexFunction(vsFunction);
  pipelineStateDescriptor->setFragmentFunction(fsFunction);
  pipelineStateDescriptor->setSampleCount(1);
  auto colorAttachment = pipelineStateDescriptor->colorAttachments()->object(0);
  colorAttachment->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
  colorAttachment->setBlendingEnabled(true);
  colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
  colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
  colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
  colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
  colorAttachment->setDestinationRGBBlendFactor(
      MTL::BlendFactorOneMinusSourceAlpha);
  colorAttachment->setDestinationAlphaBlendFactor(
      MTL::BlendFactorOneMinusSourceAlpha);

  m_pipelineState =
      device->newRenderPipelineState(pipelineStateDescriptor, &error);
  CHECK_AND_RETURN(error, false);

  return true;
}

void TextRenderer::beginLayouting() {
  m_screenGlyphs.clear();
  m_prevScreenGlyphsHash = m_screenGlyphsHash;
  m_screenGlyphsHash = 0;
}

void TextRenderer::addText(std::string const &s, glm::vec2 const &leftTop,
                           glm::vec2 const &size, glm::vec4 const &color,
                           GlyphSet const &glyphSet) {
  if (s.empty()) {
    return;
  }

  // Combine hash.
  utils::hashCombine(m_screenGlyphsHash, s);
  utils::hashCombine(m_screenGlyphsHash, leftTop.x);
  utils::hashCombine(m_screenGlyphsHash, leftTop.y);
  utils::hashCombine(m_screenGlyphsHash, size.x);
  utils::hashCombine(m_screenGlyphsHash, size.y);
  utils::hashCombine(m_screenGlyphsHash, color.r);
  utils::hashCombine(m_screenGlyphsHash, color.g);
  utils::hashCombine(m_screenGlyphsHash, color.b);
  utils::hashCombine(m_screenGlyphsHash, color.a);

  auto const &glyphs = glyphSet.getGlyphs();

  // Place glyphs.
  m_screenGlyphs.reserve(m_screenGlyphs.size() + s.size());
  auto const startIndex = m_screenGlyphs.size();
  float offsetX = 0.0f;
  float maxY = 0.0;
  for (size_t i = 0; i < s.size(); ++i) {
    auto it = glyphs.find(s[i]);
    if (it == glyphs.end()) {
      it = glyphs.find(' ');
      assert(it != glyphs.end());
    }
    auto const &glyphData = it->second;

    auto const atlasSize = glm::vec2(glyphSet.getAtlasSize());
    auto const halfSize = glyphData.m_size * 0.5f;
    auto const uvHalfSize = glm::vec2(glyphData.m_pixelSize) * 0.5f / atlasSize;

    Glyph g;
    g.halfSize = make_packed_float2(halfSize);
    g.center = make_packed_float2(glm::vec2(offsetX, 0.0f) +
                                  glyphData.m_offset + halfSize);
    g.uvCenter = make_packed_float2(
        glm::vec2(glyphData.m_posInAtlas) / atlasSize + uvHalfSize);
    g.uvHalfSize =
        make_packed_float2(uvHalfSize - glm::vec2(GlyphSet::kBorderInPixels,
                                                  GlyphSet::kBorderInPixels) /
                                            atlasSize);
    g.color = make_packed_float4(color);
    m_screenGlyphs.push_back(std::move(g));

    if (i + 1 < s.size()) {
      offsetX += glyphData.m_advance;
      maxY = std::max(maxY, g.center.y + g.halfSize.y);
    }
  }

  // Do simple layouting.
  float const scale = std::min(size.x / offsetX, size.y / maxY);
  float const layoutOffsetX = (size.x - offsetX * scale) * 0.5f;
  float const layoutOffsetY = (size.y - maxY * scale) * 0.5f;
  for (size_t i = startIndex; i < m_screenGlyphs.size(); ++i) {
    m_screenGlyphs[i].center.x =
        leftTop.x + m_screenGlyphs[i].center.x * scale + layoutOffsetX;
    m_screenGlyphs[i].center.y =
        leftTop.y + m_screenGlyphs[i].center.y * scale + layoutOffsetY;
    m_screenGlyphs[i].halfSize.x *= scale;
    m_screenGlyphs[i].halfSize.y *= scale;
  }
}

void TextRenderer::endLayouting(MTL::Device *device) {
  // In theory hash-based solution can suffer from collisions (it's highly
  // unlikely though). Consider to improve it for production code.
  if (m_screenGlyphsHash == m_prevScreenGlyphsHash) {
    return;
  }

  // Update data buffer.
  auto newGlyphBufferSize = m_glyphBufferSize;
  while (m_screenGlyphs.size() > newGlyphBufferSize) {
    newGlyphBufferSize *= 2;
  }
  if (newGlyphBufferSize != m_glyphBufferSize) {
    // Reallocate buffer.
    m_glyphBufferSize = newGlyphBufferSize;
    m_glyphBuffer->release();
    m_glyphBuffer = device->newBuffer(m_glyphBufferSize * sizeof(Glyph),
                                      MTL::ResourceStorageModeShared);
  }
  memcpy(m_glyphBuffer->contents(), m_screenGlyphs.data(),
         m_screenGlyphs.size() * sizeof(Glyph));
}

void TextRenderer::render(glm::vec2 const &screenSize,
                          MTL::RenderCommandEncoder *commandEncoder,
                          MTL::Texture *glyphTexture) {
  if (m_screenGlyphs.empty()) {
    return;
  }
  FrameData frameData;
  auto const m = glm::ortho(0.0f, screenSize.x, 0.0f, screenSize.y);
  memcpy(&frameData.projection, glm::value_ptr(m), sizeof(m));

  commandEncoder->setRenderPipelineState(m_pipelineState);
  commandEncoder->setVertexBuffer(m_glyphBuffer, 0, TextRenderBufferGlyphs);
  commandEncoder->setVertexBytes(&frameData, sizeof(frameData),
                                 TextRenderBufferFrame);
  commandEncoder->setFragmentTexture(glyphTexture, TextRenderTextureGlyphs);
  commandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                                 0 /* vertexStart */, 4 /* vertexCount */,
                                 static_cast<uint32_t>(m_screenGlyphs.size()));
}

} // namespace sdf::gpu
