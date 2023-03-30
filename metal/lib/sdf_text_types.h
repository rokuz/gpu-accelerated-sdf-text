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

#ifndef SDF_TEXT_TYPES_H
#define SDF_TEXT_TYPES_H

#include <simd/simd.h>

typedef struct Line {
  packed_float2 from;
  packed_float2 to;
} Line;

typedef struct SdfGenParams {
  packed_float2 pointPos;
  uint linesCount;
  uint lineBufferOffset;
} SdfGenParams;

typedef enum SdfGenBuffer {
  SdfGenBufferLines = 0,
  SdfGenBufferParams,
  SdfGenBufferMinDistance,
  SdfGenBufferIntersectionNumber
} SdfGenBuffer;

typedef enum SdfGenSharedMemory {
  SdfGenSharedMemoryMinDistance = 0,
  SdfGenSharedMemoryIntersectionNumber
} SdfGenSharedMemory;

typedef enum SdfTexture {
  SdfTextureOut = 0,
  SdfTextureInMinDistance,
  SdfTextureInIntersectionNumber
} SdfTexture;

typedef struct Glyph {
  packed_float2 center;
  packed_float2 halfSize;
  packed_float2 uvCenter;
  packed_float2 uvHalfSize;
  packed_float4 color;
} Glyph;

typedef enum TextRenderBuffer {
  TextRenderBufferFrame = 0,
  TextRenderBufferGlyphs
} TextRenderBuffer;

typedef enum TextRenderTexture {
  TextRenderTextureGlyphs = 0
} TextRenderTexture;

typedef struct FrameData {
  matrix_float4x4 projection;
} FrameData;

#endif /* SDF_TEXT_TYPES_H */
