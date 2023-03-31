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

#pragma once

#include <Metal/Metal.hpp>
#include <string>

#include "glyph_set.hpp"
#include "glyph_texture.hpp"
#include "sdf_text_types.h"

namespace sdf::gpu {

class TextRenderer {
public:
  ~TextRenderer();
  bool initialize(MTL::Device * const device, MTL::Library * library);

  void beginLayouting();
  void addText(std::string const & s,
               glm::vec2 const & leftTop,
               glm::vec2 const & size,
               glm::vec4 const & color,
               GlyphSet const & glyphSet);
  void endLayouting(MTL::Device * const device);

  void render(glm::vec2 const & screenSize,
              MTL::RenderCommandEncoder * commandEncoder,
              MTL::Texture * glyphTexture);

private:
  MTL::Buffer * m_glyphBuffer = nullptr;
  uint32_t m_glyphBufferSize = 0;
  MTL::RenderPipelineState * m_pipelineState = nullptr;

  std::vector<Glyph> m_screenGlyphs;
  size_t m_screenGlyphsHash = 0;
  size_t m_prevScreenGlyphsHash = 0;
};

}  // namespace sdf::gpu
