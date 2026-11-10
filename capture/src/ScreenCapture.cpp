#include "ScreenCapture.h"
#include <dxgi1_2.h>
#include <stdio.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

ScreenCapture::~ScreenCapture() {
    ReleaseFrame();
}

bool ScreenCapture::Initialize(UINT adapterIndex, UINT outputIndex) {
    m_adapterIndex = adapterIndex;
    m_outputIndex  = outputIndex;

    // Create D3D11 device on the requested adapter.
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        fprintf(stderr, "[capture] CreateDXGIFactory1 failed\n");
        return false;
    }

    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(factory->EnumAdapters1(adapterIndex, &adapter))) {
        fprintf(stderr, "[capture] No adapter at index %u\n", adapterIndex);
        return false;
    }

    D3D_FEATURE_LEVEL featureLevel;
    if (FAILED(D3D11CreateDevice(
            adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            0, nullptr, 0, D3D11_SDK_VERSION,
            &m_device, &featureLevel, &m_context))) {
        fprintf(stderr, "[capture] D3D11CreateDevice failed\n");
        return false;
    }

    // Get the requested output.
    ComPtr<IDXGIDevice> dxgiDevice;
    m_device.As(&dxgiDevice);

    ComPtr<IDXGIOutput> output;
    if (FAILED(adapter->EnumOutputs(outputIndex, &output))) {
        fprintf(stderr, "[capture] No output at index %u\n", outputIndex);
        return false;
    }

    DXGI_OUTPUT_DESC desc{};
    output->GetDesc(&desc);
    m_outputWidth  = desc.DesktopCoordinates.right  - desc.DesktopCoordinates.left;
    m_outputHeight = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;

    // Require IDXGIOutput1 for Desktop Duplication API.
    ComPtr<IDXGIOutput1> output1;
    if (FAILED(output.As(&output1))) {
        fprintf(stderr, "[capture] IDXGIOutput1 not supported\n");
        return false;
    }

    if (FAILED(output1->DuplicateOutput(m_device.Get(), &m_duplication))) {
        fprintf(stderr, "[capture] DuplicateOutput failed (need Desktop Window Manager)\n");
        return false;
    }

    // Pre-create staging texture at max resolution to avoid alloc in hot path.
    D3D11_TEXTURE2D_DESC stagingDesc{};
    stagingDesc.Width          = m_outputWidth;
    stagingDesc.Height         = m_outputHeight;
    stagingDesc.MipLevels      = 1;
    stagingDesc.ArraySize      = 1;
    stagingDesc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM;
    stagingDesc.SampleDesc     = {1, 0};
    stagingDesc.Usage          = D3D11_USAGE_STAGING;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    if (FAILED(m_device->CreateTexture2D(&stagingDesc, nullptr, &m_stagingTex))) {
        fprintf(stderr, "[capture] Staging texture creation failed\n");
        return false;
    }

    fprintf(stderr, "[capture] Initialized %ux%u on adapter %u output %u\n",
            m_outputWidth, m_outputHeight, adapterIndex, outputIndex);
    return true;
}

bool ScreenCapture::Reinitialize() {
    ReleaseFrame();
    m_duplication.Reset();
    m_stagingTex.Reset();
    m_device.Reset();
    m_context.Reset();
    return Initialize(m_adapterIndex, m_outputIndex);
}

bool ScreenCapture::AcquireFrame(CaptureFrame& out_frame, DWORD timeoutMs) {
    if (!m_duplication) return false;

    // Release previous frame before acquiring the next one.
    // Unmap staging texture first — otherwise a subsequent Map call will fail.
    if (m_frameMapped) {
        m_context->Unmap(m_stagingTex.Get(), 0);
        m_frameMapped = false;
    }
    if (m_acquiredResource) {
        m_duplication->ReleaseFrame();
        m_acquiredResource.Reset();
    }

    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    HRESULT hr = m_duplication->AcquireNextFrame(timeoutMs, &frameInfo, &m_acquiredResource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false; // no new frame within timeout — normal, not an error
    }
    if (hr == DXGI_ERROR_ACCESS_LOST) {
        fprintf(stderr, "[capture] Desktop duplication access lost, reinitializing\n");
        Reinitialize();
        return false;
    }
    if (FAILED(hr)) {
        fprintf(stderr, "[capture] AcquireNextFrame HRESULT=0x%08lX\n", hr);
        return false;
    }

    ComPtr<ID3D11Texture2D> gpuTex;
    if (FAILED(m_acquiredResource.As(&gpuTex))) {
        m_duplication->ReleaseFrame();
        m_acquiredResource.Reset();
        return false;
    }

    return MapStagingTexture(gpuTex.Get(), out_frame);
}

bool ScreenCapture::MapStagingTexture(ID3D11Texture2D* gpuTexture, CaptureFrame& out_frame) {
    // Copy GPU → staging (CPU-readable) texture.
    m_context->CopyResource(m_stagingTex.Get(), gpuTexture);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = m_context->Map(m_stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        fprintf(stderr, "[capture] Map staging texture failed HRESULT=0x%08lX\n", hr);
        return false;
    }

    m_frameMapped      = true;
    out_frame.data     = static_cast<const uint8_t*>(mapped.pData);
    out_frame.width    = m_outputWidth;
    out_frame.height   = m_outputHeight;
    out_frame.stride   = mapped.RowPitch;
    return true;
}

void ScreenCapture::ReleaseFrame() {
    if (m_frameMapped) {
        m_context->Unmap(m_stagingTex.Get(), 0);
        m_frameMapped = false;
    }
    if (m_acquiredResource && m_duplication) {
        m_duplication->ReleaseFrame();
        m_acquiredResource.Reset();
    }
}
