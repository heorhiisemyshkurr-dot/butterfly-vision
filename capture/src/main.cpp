#include "ScreenCapture.h"
#include "../../shared/ipc_config.h"
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <cstring>

// ── Globals ──────────────────────────────────────────────────────────────────

static volatile bool g_running = true;

static HANDLE       g_hFrameMapping  = nullptr;
static HANDLE       g_hReadySemaphore = nullptr;
static HANDLE       g_hConfigMapping  = nullptr;
static void*        g_pFrameView      = nullptr;
static void*        g_pConfigView     = nullptr;

// ── Signal handling ───────────────────────────────────────────────────────────

static BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT || signal == CTRL_CLOSE_EVENT) {
        fprintf(stderr, "[capture] Shutdown requested\n");
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

// ── IPC setup ─────────────────────────────────────────────────────────────────

static bool IpcCreate() {
    // Frame shared memory
    g_hFrameMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, (DWORD)BUTTERFLY_SHM_SIZE, BUTTERFLY_FRAME_NAME);
    if (!g_hFrameMapping) {
        fprintf(stderr, "[capture] CreateFileMapping(Frame) failed: %lu\n", GetLastError());
        return false;
    }

    g_pFrameView = MapViewOfFile(g_hFrameMapping, FILE_MAP_WRITE, 0, 0, BUTTERFLY_SHM_SIZE);
    if (!g_pFrameView) {
        fprintf(stderr, "[capture] MapViewOfFile(Frame) failed: %lu\n", GetLastError());
        return false;
    }
    ZeroMemory(g_pFrameView, BUTTERFLY_SHM_SIZE);

    // Ready semaphore (auto-reset, initial count 0)
    g_hReadySemaphore = CreateSemaphoreW(nullptr, 0, 1, BUTTERFLY_READY_NAME);
    if (!g_hReadySemaphore) {
        fprintf(stderr, "[capture] CreateSemaphore failed: %lu\n", GetLastError());
        return false;
    }

    // Config shared memory
    g_hConfigMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, (DWORD)BUTTERFLY_CFG_SIZE, BUTTERFLY_CONFIG_NAME);
    if (!g_hConfigMapping) {
        fprintf(stderr, "[capture] CreateFileMapping(Config) failed: %lu\n", GetLastError());
        return false;
    }

    g_pConfigView = MapViewOfFile(g_hConfigMapping, FILE_MAP_WRITE, 0, 0, BUTTERFLY_CFG_SIZE);
    if (!g_pConfigView) {
        fprintf(stderr, "[capture] MapViewOfFile(Config) failed: %lu\n", GetLastError());
        return false;
    }

    // Write default config: mask disabled, centered at 960×540
    ButterflyConfig* cfg = reinterpret_cast<ButterflyConfig*>(g_pConfigView);
    cfg->magic    = BUTTERFLY_CONFIG_MAGIC;
    cfg->center_x = 960;
    cfg->center_y = 540;
    cfg->radius   = 0;       // 0 = disabled
    cfg->enabled  = 0;

    return true;
}

static void IpcDestroy() {
    if (g_pConfigView)     { UnmapViewOfFile(g_pConfigView);  g_pConfigView = nullptr; }
    if (g_pFrameView)      { UnmapViewOfFile(g_pFrameView);   g_pFrameView  = nullptr; }
    if (g_hConfigMapping)  { CloseHandle(g_hConfigMapping);   g_hConfigMapping  = nullptr; }
    if (g_hFrameMapping)   { CloseHandle(g_hFrameMapping);    g_hFrameMapping   = nullptr; }
    if (g_hReadySemaphore) { CloseHandle(g_hReadySemaphore);  g_hReadySemaphore = nullptr; }
}

// ── Circular mask ─────────────────────────────────────────────────────────────

static void ApplyCircularMask(
        uint8_t*  dst,          // destination in shared memory (BGRA)
        const uint8_t* src,     // source from DXGI staging (BGRA)
        uint32_t  width,
        uint32_t  height,
        uint32_t  srcStride,
        int32_t   cx, int32_t cy, int32_t radius)
{
    const int32_t r2 = radius * radius;
    const uint32_t dstStride = width * BUTTERFLY_BYTES_PER_PX;

    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + (size_t)y * srcStride;
        uint8_t*       dstRow = dst + (size_t)y * dstStride;

        for (uint32_t x = 0; x < width; ++x) {
            int32_t dx = (int32_t)x - cx;
            int32_t dy = (int32_t)y - cy;
            if (dx * dx + dy * dy <= r2) {
                // Inside mask: copy pixel
                dstRow[x * 4 + 0] = srcRow[x * 4 + 0]; // B
                dstRow[x * 4 + 1] = srcRow[x * 4 + 1]; // G
                dstRow[x * 4 + 2] = srcRow[x * 4 + 2]; // R
                dstRow[x * 4 + 3] = srcRow[x * 4 + 3]; // A
            } else {
                // Outside mask: zero out (black transparent)
                dstRow[x * 4 + 0] = 0;
                dstRow[x * 4 + 1] = 0;
                dstRow[x * 4 + 2] = 0;
                dstRow[x * 4 + 3] = 0;
            }
        }
    }
}

static void CopyFullFrame(
        uint8_t*       dst,
        const uint8_t* src,
        uint32_t width, uint32_t height, uint32_t srcStride)
{
    const uint32_t dstStride = width * BUTTERFLY_BYTES_PER_PX;
    if (srcStride == dstStride) {
        memcpy(dst, src, (size_t)height * dstStride);
    } else {
        for (uint32_t y = 0; y < height; ++y) {
            memcpy(dst + (size_t)y * dstStride,
                   src + (size_t)y * srcStride,
                   dstStride);
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    UINT adapterIdx = 0;
    UINT outputIdx  = 0;
    if (argc >= 2) adapterIdx = (UINT)atoi(argv[1]);
    if (argc >= 3) outputIdx  = (UINT)atoi(argv[2]);

    fprintf(stderr, "[capture] butterfly_capture v1.0  adapter=%u output=%u\n",
            adapterIdx, outputIdx);

    if (!IpcCreate()) {
        fprintf(stderr, "[capture] IPC initialization failed\n");
        return 1;
    }

    ScreenCapture cap;
    if (!cap.Initialize(adapterIdx, outputIdx)) {
        fprintf(stderr, "[capture] DXGI initialization failed\n");
        IpcDestroy();
        return 1;
    }

    auto* hdr  = reinterpret_cast<ButterflyFrameHeader*>(g_pFrameView);
    auto* pixels = reinterpret_cast<uint8_t*>(g_pFrameView) + BUTTERFLY_PIXEL_OFFSET;
    auto* cfg  = reinterpret_cast<ButterflyConfig*>(g_pConfigView);

    hdr->magic  = BUTTERFLY_FRAME_MAGIC;
    hdr->width  = cap.OutputWidth();
    hdr->height = cap.OutputHeight();
    hdr->stride = cap.OutputWidth() * BUTTERFLY_BYTES_PER_PX;

    uint32_t frameId    = 0;
    uint32_t skipCount  = 0;
    DWORD    lastReport = GetTickCount();

    fprintf(stderr, "[capture] Running — press Ctrl+C to stop\n");

    while (g_running) {
        CaptureFrame frame{};
        if (!cap.AcquireFrame(frame, 100)) {
            ++skipCount;
            continue;
        }

        // Read current mask config (lock-free: single writer from config-ui)
        bool   maskEnabled = (cfg->enabled != 0) && (cfg->radius > 0);
        int32_t cx         = cfg->center_x;
        int32_t cy         = cfg->center_y;
        int32_t radius     = cfg->radius;

        if (maskEnabled) {
            ApplyCircularMask(pixels, frame.data,
                              frame.width, frame.height, frame.stride,
                              cx, cy, radius);
        } else {
            CopyFullFrame(pixels, frame.data,
                          frame.width, frame.height, frame.stride);
        }

        cap.ReleaseFrame();

        // Update header
        hdr->frame_id     = ++frameId;
        hdr->timestamp_ms = GetTickCount();
        hdr->width        = frame.width;
        hdr->height       = frame.height;
        hdr->stride       = frame.width * BUTTERFLY_BYTES_PER_PX;

        // Signal detector: semaphore count capped at 1 (detector may be slower)
        ReleaseSemaphore(g_hReadySemaphore, 1, nullptr);

        // Limit capture rate to ~15 FPS (1000ms / 15 ≈ 66ms)
        Sleep(66);

        // Print stats every 5 seconds
        DWORD now = GetTickCount();
        if (now - lastReport >= 5000) {
            fprintf(stderr, "[capture] frames=%u skipped=%u\n", frameId, skipCount);
            lastReport = now;
        }
    }

    fprintf(stderr, "[capture] Exiting gracefully\n");
    IpcDestroy();
    return 0;
}
