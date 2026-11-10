#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;

struct CaptureFrame {
    const uint8_t* data;   // BGRA pixel data, owned by ScreenCapture
    uint32_t       width;
    uint32_t       height;
    uint32_t       stride; // bytes per row
};

class ScreenCapture {
public:
    ScreenCapture() = default;
    ~ScreenCapture();

    // Initialize DXGI Desktop Duplication on the given output index.
    // outputIndex 0 = primary monitor.
    // Returns false and logs reason on failure.
    bool Initialize(UINT adapterIndex = 0, UINT outputIndex = 0);

    // Capture next available frame (blocks up to timeoutMs).
    // Returns true if a new frame was placed into out_frame.
    // out_frame.data is valid until the next call to AcquireFrame().
    bool AcquireFrame(CaptureFrame& out_frame, DWORD timeoutMs = 100);

    // Must be called after AcquireFrame() to release DXGI resources.
    void ReleaseFrame();

    // Returns true if DXGI session is valid.
    bool IsValid() const { return m_duplication != nullptr; }

    // Re-initialize after desktop switch / mode change.
    bool Reinitialize();

    uint32_t OutputWidth()  const { return m_outputWidth; }
    uint32_t OutputHeight() const { return m_outputHeight; }

private:
    // Convert GPU texture to CPU-accessible staging texture and map pixels.
    bool MapStagingTexture(ID3D11Texture2D* gpuTexture, CaptureFrame& out_frame);

    ComPtr<ID3D11Device>             m_device;
    ComPtr<ID3D11DeviceContext>      m_context;
    ComPtr<IDXGIOutputDuplication>   m_duplication;
    ComPtr<ID3D11Texture2D>          m_stagingTex;
    ComPtr<IDXGIResource>            m_acquiredResource;

    uint32_t m_outputWidth  = 0;
    uint32_t m_outputHeight = 0;
    UINT     m_adapterIndex = 0;
    UINT     m_outputIndex  = 0;
    bool     m_frameMapped  = false;
};
