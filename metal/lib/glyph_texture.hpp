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

#include <Metal/Metal.hpp>

#include "glyph_set.hpp"

namespace sdf::gpu {

class GlyphTexture {
public:
  static MTL::Texture *generate(MTL::Device *device, MTL::Library *library,
                                MTL::CommandQueue *commandQueue,
                                GlyphSet const &glyphSet);
};

} // namespace sdf::gpu
