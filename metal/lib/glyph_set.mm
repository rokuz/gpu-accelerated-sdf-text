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

#include "glyph_set.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>

#include <optional>

#if !__has_feature(objc_arc)
#error "ARC is off"
#endif

namespace sdf {
namespace {
char const *const kFontName = "Helvetica";

glm::vec2 getPointOnQuadBezierCurve(CGPoint const &p1, CGPoint const &p2,
                                    CGPoint const &p3, float t) {
  auto const oneMinusT = 1.0f - t;
  auto const a = oneMinusT * oneMinusT;
  auto const b = 2.0f * oneMinusT * t;
  auto const c = t * t;
  return glm::vec2{a * p1.x + b * p2.x + c * p3.x,
                   a * p1.y + b * p2.y + c * p3.y};
}

glm::vec2 getPointOnCubicBezierCurve(CGPoint const &p1, CGPoint const &p2,
                                     CGPoint const &p3, CGPoint const &p4,
                                     float t) {
  auto const oneMinusT = 1.0f - t;
  auto const a = oneMinusT * oneMinusT * oneMinusT;
  auto const b = 3.0f * oneMinusT * oneMinusT * t;
  auto const c = 3.0f * oneMinusT * t * t;
  auto const d = t * t * t;
  return glm::vec2{a * p1.x + b * p2.x + c * p3.x + d * p4.x,
                   a * p1.y + b * p2.y + c * p3.y + d * p4.y};
}

void subdivideCurve(std::function<glm::vec2(float)> const &getPoint,
                    std::vector<glm::vec4> &lines) {
  auto constexpr kMaxPointsNum = 50;
  auto constexpr kTangentTolerance = 0.01f;
  glm::vec2 prevPoint = getPoint(0.0f);
  glm::vec2 prevTangentVec;
  for (int i = 0; i < kMaxPointsNum; ++i) {
    auto const t = static_cast<float>(i + 1) / kMaxPointsNum;
    auto const currentPoint = getPoint(t);
    auto const tangentVec = glm::normalize(currentPoint - prevPoint);
    if (i == 0 || (i + 1 == kMaxPointsNum) ||
        glm::dot(prevTangentVec, tangentVec) < (1.0f - kTangentTolerance)) {
      lines.emplace_back(
          glm::vec4(prevPoint.x, prevPoint.y, currentPoint.x, currentPoint.y));
      prevPoint = currentPoint;
      prevTangentVec = tangentVec;
    }
  }
}

GlyphSet::GlyphData buildGlyphData(CTFontRef ctFont, CGFontRef cgFont,
                                   uint16_t code, float scale) {
  CGGlyph g;
  if (!CTFontGetGlyphsForCharacters(ctFont, &code, &g, 1)) {
    g = 0x30;
  }
  CGRect rect = {};
  CGFontGetGlyphBBoxes(cgFont, &g, 1, &rect);

  int advanceX = 0;
  CGFontGetGlyphAdvances(cgFont, &g, 1, &advanceX);

  GlyphSet::GlyphData data;
  data.m_advance = static_cast<float>(advanceX) * scale;
  data.m_offset = glm::vec2{static_cast<float>(rect.origin.x) * scale,
                            static_cast<float>(rect.origin.y) * scale};
  data.m_size = glm::vec2{static_cast<float>(rect.size.width) * scale,
                          static_cast<float>(rect.size.height) * scale};
  data.m_pixelSize = glm::uvec2{static_cast<uint32_t>(ceil(data.m_size.x)),
                                static_cast<uint32_t>(ceil(data.m_size.y))};

  auto const scaleX = static_cast<float>(data.m_pixelSize.x) / data.m_size.x;
  auto const scaleY = static_cast<float>(data.m_pixelSize.y) / data.m_size.y;

  data.m_pixelSize += 2 * GlyphSet::kBorderInPixels;

  auto glyphTransform = CGAffineTransformMake(
      scaleX, 0, 0, -scaleY,
      -data.m_offset.x * scaleX + GlyphSet::kBorderInPixels,
      data.m_pixelSize.y + data.m_offset.y * scaleY -
          GlyphSet::kBorderInPixels);
  CGPathRef path = CTFontCreatePathForGlyph(ctFont, g, &glyphTransform);
  if (path == nil) {
    return data;
  }

  __block CGPoint startPoint = {};
  __block CGPoint prevPoint = {};
  __block std::vector<glm::vec4> lines;
  CGPathApplyWithBlock(path, ^(CGPathElement const *e) {
    switch (e->type) {
    case kCGPathElementMoveToPoint: {
      startPoint = prevPoint = e->points[0];
      break;
    }
    case kCGPathElementAddLineToPoint: {
      lines.emplace_back(
          glm::vec4(prevPoint.x, prevPoint.y, e->points[0].x, e->points[0].y));
      prevPoint = e->points[0];
      break;
    }
    case kCGPathElementAddQuadCurveToPoint: {
      auto prev = prevPoint;
      subdivideCurve(
          [&](float t) {
            return getPointOnQuadBezierCurve(prev, e->points[0], e->points[1],
                                             t);
          },
          lines);
      prevPoint = e->points[1];
      break;
    }
    case kCGPathElementAddCurveToPoint: {
      auto prev = prevPoint;
      subdivideCurve(
          [&](float t) {
            return getPointOnCubicBezierCurve(prev, e->points[0], e->points[1],
                                              e->points[2], t);
          },
          lines);
      prevPoint = e->points[2];
      break;
    }
    case kCGPathElementCloseSubpath:
      lines.emplace_back(
          glm::vec4(prevPoint.x, prevPoint.y, startPoint.x, startPoint.y));
      break;
    }
  });
  data.m_lines = std::move(lines);

  CGPathRelease(path);
  return data;
}

class AtlasPacker {
public:
  explicit AtlasPacker(uint32_t atlasSize) : m_atlasSize(atlasSize) {}

  std::optional<glm::uvec2> pack(glm::uvec2 const &size) {
    if (m_cursor.x + size.x + 1 > m_atlasSize) {
      m_cursor.x = 1;
      m_cursor.y += (m_yStep + 1);
      m_yStep = 0;
    }

    if (m_cursor.y + size.y + 1 > m_atlasSize)
      return {};

    glm::uvec2 pos = m_cursor;
    m_cursor.x += (size.x + 1);
    m_yStep = std::max(static_cast<uint32_t>(size.y), m_yStep);
    return pos;
  }

private:
  uint32_t m_atlasSize = 0;
  glm::uvec2 m_cursor = glm::uvec2{1, 1};
  uint32_t m_yStep = 0;
};
} // namespace

GlyphSet::GlyphSet(std::vector<uint16_t> const &unicodeGlyphs,
                   uint32_t baseAtlasSize /* = 256 */,
                   uint32_t baseFontSize /* = 24 */) {
  CFStringRef fontName = CFStringCreateWithCString(nullptr, kFontName,
                                                   CFStringGetSystemEncoding());
  auto ctFont = CTFontCreateWithName(fontName, baseFontSize, nullptr);
  CFRelease(fontName);

  // Build glyphs.
  auto cgFont = CTFontCopyGraphicsFont(ctFont, nullptr);
  auto const scale =
      static_cast<float>(baseFontSize) / CGFontGetUnitsPerEm(cgFont);
  for (auto code : unicodeGlyphs) {
    m_glyphs[code] = buildGlyphData(ctFont, cgFont, code, scale);
  }

  CFRelease(cgFont);
  CFRelease(ctFont);

  // Pack glyphs.
  packGlyphsToAtlas(baseAtlasSize);
}

void GlyphSet::packGlyphsToAtlas(uint32_t atlasSize) {
  m_atlasSize = glm::uvec2{atlasSize, atlasSize};
  AtlasPacker packer(atlasSize);
  for (auto &[code, glyphData] : m_glyphs) {
    if (auto p = packer.pack(glyphData.m_pixelSize)) {
      glyphData.m_posInAtlas = p.value();
    } else {
      // Not enough space in atlas, extend and try again.
      packGlyphsToAtlas(atlasSize * 2);
      return;
    }
  }
}

} // namespace sdf
