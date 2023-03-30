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

#include "renderer.hpp"

#include <chrono>

#include "common/utils.hpp"
#include "lib/glyph_texture.hpp"

App *getApp() {
  static Renderer app;
  return &app;
}

char const *const kDemoName = "GPU Accelerated SDF Text";

namespace {
uint32_t constexpr kMaxFramesInFlight = 3;

std::vector<uint16_t> enumerateGlyphs() {
  static std::string const kGlyphs =
      "abcdefghijklmnopqrstuvwxyz "
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ-?!,.:;0123456789()";
  std::vector<uint16_t> v(kGlyphs.size());
  for (size_t i = 0; i < kGlyphs.size(); ++i) {
    v[i] = kGlyphs[i];
  }
  return v;
}
} // namespace

Renderer::Renderer()
    : m_glyphs(enumerateGlyphs()), m_semaphore(kMaxFramesInFlight) {}

char const *const Renderer::getName() const {
  return kDemoName;
}

bool Renderer::onInitialize(MTL::Device *const device, uint32_t screenWidth,
                            uint32_t screenHeight) {
  m_device = device;
  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;
  
  m_commandQueue = m_device->newCommandQueue();
  assert(m_commandQueue != 0);

  NS::Error *error = nullptr;
  auto libraryPath =
      NS::Bundle::mainBundle()->resourcePath()->stringByAppendingString(
          STR("/gpu-accelerated-sdf-text-lib.metallib"));
  m_library = m_device->newLibrary(libraryPath, &error);
  CHECK_AND_RETURN(error, false);
  assert(m_library != 0);

  auto t1 = std::chrono::steady_clock::now();
  m_glyphTexture = sdf::gpu::GlyphTexture::generate(m_device, m_library,
                                                    m_commandQueue, m_glyphs);
  auto const duration = std::chrono::steady_clock::now() - t1;
  m_glyphGenTimeMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  m_textRenderer = std::make_unique<sdf::gpu::TextRenderer>();
  if (!m_textRenderer->initialize(m_device, m_library)) {
    return false;
  }

  return true;
}

void Renderer::onDeinitialize() {
  auto commandBuffer = m_commandQueue->commandBuffer();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  if (m_glyphTexture) {
    m_glyphTexture->release();
  }

  m_textRenderer.reset();

  m_library->release();
  m_commandQueue->release();
}

void Renderer::onResize(uint32_t screenWidth, uint32_t screenHeight) {
  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;
}

void Renderer::onMainLoopTick(CA::MetalDrawable *drawable,
                              double elapsedSeconds) {
  NS::AutoreleasePool *autoreleasePool = NS::AutoreleasePool::alloc()->init();
  METAL_GUARD(autoreleasePool);

  m_fpsTimer += elapsedSeconds;
  m_frameCounter++;
  if (m_fpsTimer > 1.0) {
    m_fps = static_cast<double>(m_frameCounter) / m_fpsTimer;
    m_fpsTimer = 0;
    m_frameCounter = 0;
  }

  m_textRenderer->beginLayouting();

  auto const screenSz = glm::vec2(m_screenWidth, m_screenHeight);

  // Some content.
  {
    auto sz = glm::vec2(400, 200);
    m_textRenderer->addText("This text is rendered by",
                            glm::vec2(screenSz - sz) * 0.5f +
                                screenSz * glm::vec2(-0.25f, 0.25f),
                            sz, glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), m_glyphs);
    sz = glm::vec2(600, 200);
    m_textRenderer->addText("GPU Accelerated SDF algorithm",
                            glm::vec2(screenSz - sz) * 0.5f, sz,
                            glm::vec4(0.5f, 0.1f, 0.1f, 1.0f), m_glyphs);
    sz = glm::vec2(200, 200);
    m_textRenderer->addText("written by @rokuz",
                            glm::vec2(screenSz - sz) * 0.5f +
                                screenSz * glm::vec2(0.25f, -0.25f),
                            sz, glm::vec4(0.1f, 0.1f, 0.1f, 1.0f), m_glyphs);
  }

  // FPS.
  {
    auto sz = glm::vec2(150, 20);
    std::string s(16, '\0');
    if (auto w = std::snprintf(&s[0], s.size(), "FPS: %.2f", m_fps); w > 0) {
      s.resize(w);
    } else {
      s.clear();
    }
    m_textRenderer->addText(s, screenSz - sz - 50.0f, sz,
                            glm::vec4(0.0f, 0.5f, 0.0f, 1.0f), m_glyphs);
  }

  // Glyphs generation time.
  {
    auto sz = glm::vec2(300, 20);
    std::string s(50, '\0');
    if (auto w = std::snprintf(&s[0], s.size(), "SDF generation time: %llu ms",
                               m_glyphGenTimeMs);
        w > 0) {
      s.resize(w);
    } else {
      s.clear();
    }
    m_textRenderer->addText(s, glm::vec2(50.0f, screenSz.y - sz.y - 50.0f), sz,
                            glm::vec4(0.0f, 0.5f, 0.0f, 1.0f), m_glyphs);
  }

  m_textRenderer->endLayouting(m_device);

  m_semaphore.wait();

  auto renderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
  auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
  colorAttachment->setClearColor(MTL::ClearColor::Make(0.9, 0.9, 0.9, 1.0));
  colorAttachment->setLoadAction(MTL::LoadAction::LoadActionClear);
  colorAttachment->setStoreAction(MTL::StoreAction::StoreActionStore);

  auto tex = drawable->texture();
  colorAttachment->setTexture(tex);

  auto commandBuffer = m_commandQueue->commandBuffer();
  commandBuffer->setLabel(STR("Frame Command Buffer"));
  commandBuffer->addCompletedHandler(
      [this](MTL::CommandBuffer *) { m_semaphore.signal(); });

  MTL::RenderCommandEncoder *encoder =
      commandBuffer->renderCommandEncoder(renderPassDescriptor);
  encoder->setLabel(STR("Main Command Encoder"));
  m_textRenderer->render(glm::vec2(m_screenWidth, m_screenHeight), encoder,
                         m_glyphTexture);
  encoder->endEncoding();

  commandBuffer->presentDrawable(drawable);
  commandBuffer->commit();
}
