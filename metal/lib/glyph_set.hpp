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

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "common/glm_math.hpp"

namespace sdf {

class GlyphSet {
public:
  static uint32_t constexpr kBorderInPixels = 4;

  explicit GlyphSet(std::vector<uint16_t> const & unicodeGlyphs,
                    uint32_t baseAtlasSize = 256,
                    uint32_t baseFontSize = 48);

  struct GlyphData {
    std::vector<glm::vec4> m_lines;
    float m_advance = 0.0;
    glm::vec2 m_offset;
    glm::vec2 m_size;
    glm::uvec2 m_pixelSize;
    glm::uvec2 m_posInAtlas;
  };
  auto const & getGlyphs() const { return m_glyphs; }
  glm::uvec2 const & getAtlasSize() const { return m_atlasSize; }

private:
  void packGlyphsToAtlas(uint32_t atlasSize);

  std::unordered_map<uint16_t, GlyphData> m_glyphs;
  glm::uvec2 m_atlasSize;
};

}  // namespace sdf
