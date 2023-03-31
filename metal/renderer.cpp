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

App * getApp() {
  static Renderer app;
  return &app;
}

char const * const kDemoName = "GPU Accelerated SDF Text";

namespace {
uint32_t constexpr kMaxFramesInFlight = 3;

std::vector<uint16_t> enumerateGlyphs() {
  static std::string const kGlyphs =
    "abcdefghijklmnopqrstuvwxyz "
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ-?!,.:;0123456789()@";
  std::vector<uint16_t> v(kGlyphs.size());
  for (size_t i = 0; i < kGlyphs.size(); ++i) {
    v[i] = kGlyphs[i];
  }
  return v;
}
}  // namespace

Renderer::Renderer() : m_glyphs(enumerateGlyphs()) {}

char const * const Renderer::getName() const { return kDemoName; }

bool Renderer::onInitialize(MTL::Device * const device,
                            MTL::CommandQueue * const commandQueue,
                            uint32_t screenWidth,
                            uint32_t screenHeight) {
  METAL_ASSERT(device != 0);
  METAL_ASSERT(commandQueue != 0);
  m_context = std::unique_ptr<MetalContext>(new MetalContext{device, commandQueue});
  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;

  m_gpuFamily = utils::getMetalGpuFamily(device);

  NS::Error * error = nullptr;
  auto libraryPath = NS::Bundle::mainBundle()->resourcePath()->stringByAppendingString(
    STR("/gpu-accelerated-sdf-text-lib.metallib"));
  m_library = m_context->m_device->newLibrary(libraryPath, &error);
  CHECK_AND_RETURN(error, false);
  METAL_ASSERT(m_library != 0);

  auto t1 = std::chrono::steady_clock::now();
  m_glyphTexture = sdf::gpu::GlyphTexture::generate(m_context->m_device,
                                                    m_context->m_commandQueue,
                                                    m_library,
                                                    m_glyphs);
  auto const duration = std::chrono::steady_clock::now() - t1;
  m_glyphGenTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  m_textRenderer = std::make_unique<sdf::gpu::TextRenderer>();
  if (!m_textRenderer->initialize(m_context->m_device, m_library)) {
    return false;
  }

  return true;
}

void Renderer::onDeinitialize() {
  auto commandBuffer = m_context->m_commandQueue->commandBuffer();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  if (m_glyphTexture) {
    m_glyphTexture->release();
  }

  m_textRenderer.reset();

  m_library->release();
}

void Renderer::onResize(uint32_t screenWidth, uint32_t screenHeight) {
  m_screenWidth = screenWidth;
  m_screenHeight = screenHeight;
}

void Renderer::renderFrame(MTL::CommandBuffer * frameCommandBuffer,
                           MTL::Texture * outputTexture,
                           double elapsedSeconds) {
  NS::AutoreleasePool * autoreleasePool = NS::AutoreleasePool::alloc()->init();
  METAL_GUARD(autoreleasePool);

  m_textRenderer->beginLayouting();

  auto const screenSz = glm::vec2(m_screenWidth, m_screenHeight);

  // Some content.
  {
    auto sz = glm::vec2(400, 200);
    m_textRenderer->addText("This text is rendered by",
                            glm::vec2(screenSz - sz) * 0.5f + screenSz * glm::vec2(-0.25f, 0.1f),
                            sz,
                            glm::vec4(0.1f, 0.1f, 0.1f, 1.0f),
                            m_glyphs);
    sz = glm::vec2(600, 200);
    m_textRenderer->addText("GPU Accelerated SDF algorithm",
                            glm::vec2(screenSz - sz) * 0.5f,
                            sz,
                            glm::vec4(0.5f, 0.1f, 0.1f, 1.0f),
                            m_glyphs);
    sz = glm::vec2(200, 200);
    m_textRenderer->addText("written by @rokuz",
                            glm::vec2(screenSz - sz) * 0.5f + screenSz * glm::vec2(0.25f, -0.1f),
                            sz,
                            glm::vec4(0.1f, 0.1f, 0.1f, 1.0f),
                            m_glyphs);
  }

  m_textRenderer->endLayouting(m_context->m_device);

  auto renderPassDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
  auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
  colorAttachment->setTexture(outputTexture);
  colorAttachment->setClearColor(MTL::ClearColor::Make(0.9, 0.9, 0.9, 1.0));
  colorAttachment->setLoadAction(MTL::LoadAction::LoadActionClear);
  colorAttachment->setStoreAction(MTL::StoreAction::StoreActionStore);

  MTL::RenderCommandEncoder * encoder =
    frameCommandBuffer->renderCommandEncoder(renderPassDescriptor);
  encoder->setLabel(STR("Main Command Encoder"));

  encoder->pushDebugGroup(STR("Encode Text Rendering"));
  m_textRenderer->render(glm::vec2(m_screenWidth, m_screenHeight), encoder, m_glyphTexture);
  encoder->popDebugGroup();

  app::renderImGui(frameCommandBuffer, renderPassDescriptor, encoder, [=, this](ImGuiIO & io) {
    m_fpsTimer += (1.0 / io.Framerate);
    m_frameCounter++;
    if (m_fpsTimer > 1.0) {
      m_fps = static_cast<double>(m_frameCounter) / m_fpsTimer;
      m_fpsTimer = 0;
      m_frameCounter = 0;
    }

    static bool enableVSync = app::isEnabledVSync();
    ImVec2 sz = ImGui::GetMainViewport()->WorkSize;
    ImGui::SetNextWindowPos(ImVec2(sz.x - 10, 10), ImGuiCond_Appearing, ImVec2(1.0f, 0.0f));
    ImGui::Begin("Info & Controls");
    ImGui::Text("Device: %s", m_context->m_device->name()->utf8String());
    ImGui::Text("GPU Family: %s", m_gpuFamily.c_str());
    ImGui::Text("SDF texture gen time: %llu ms", m_glyphGenTimeMs);
    ImGui::Text("Avg time frame = %.3f ms (%.1f FPS)",
                m_fps == 0 ? 0.0f : (1000.0f / m_fps),
                m_fps);
    if (ImGui::Checkbox("Enable VSync", &enableVSync)) {
      app::setEnabledVSync(enableVSync);
    }
    if (ImGui::IsKeyReleased(ImGuiKey_Escape)) {
      app::closeApp();
    }
    ImGui::End();
  });
  encoder->endEncoding();
}
