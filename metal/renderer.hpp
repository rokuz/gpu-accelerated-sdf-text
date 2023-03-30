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

#include <memory>

#include "common/app.hpp"
#include "common/semaphore.hpp"
#include "lib/glyph_set.hpp"
#include "lib/text_renderer.hpp"

class Renderer : public App {
public:
  Renderer();

  char const *const getName() const override;
  
  bool onInitialize(MTL::Device *device, uint32_t screenWidth,
                    uint32_t screenHeight) override;
  void onDeinitialize() override;

  void onResize(uint32_t screenWidth, uint32_t screenHeight) override;

  void onMainLoopTick(CA::MetalDrawable *drawable,
                      double elapsedSeconds) override;

private:
  sdf::GlyphSet m_glyphs;
  std::unique_ptr<sdf::gpu::TextRenderer> m_textRenderer;

  MTL::Device *m_device = nullptr;
  uint32_t m_screenWidth = 0;
  uint32_t m_screenHeight = 0;

  MTL::CommandQueue *m_commandQueue = nullptr;
  MTL::Library *m_library = nullptr;

  MTL::Texture *m_glyphTexture = nullptr;

  semaphore m_semaphore;

  double m_fpsTimer = 0.0;
  uint32_t m_frameCounter = 0;
  double m_fps = 0.0;

  uint64_t m_glyphGenTimeMs = 0.0;
};
