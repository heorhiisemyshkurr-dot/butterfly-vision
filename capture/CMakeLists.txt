cmake_minimum_required(VERSION 3.20)
project(butterfly_capture CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows-only
if(NOT WIN32)
    message(FATAL_ERROR "butterfly_capture requires Windows (DXGI API)")
endif()

add_executable(butterfly_capture
    src/main.cpp
    src/ScreenCapture.cpp
)

target_include_directories(butterfly_capture PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared
)

target_compile_definitions(butterfly_capture PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    UNICODE
    _UNICODE
)

target_link_libraries(butterfly_capture PRIVATE
    d3d11
    dxgi
    user32
    kernel32
)

# MSVC-specific flags
if(MSVC)
    target_compile_options(butterfly_capture PRIVATE
        /W4         # High warning level
        /WX-        # Warnings are not errors (DXGI has some)
        /O2         # Max optimization
        /GL         # Whole-program optimization
        /MT         # Static CRT (no runtime DLL dependency)
    )
    target_link_options(butterfly_capture PRIVATE
        /LTCG       # Link-time code generation
        /SUBSYSTEM:CONSOLE
    )
endif()

set_target_properties(butterfly_capture PROPERTIES
    OUTPUT_NAME "butterfly_capture"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)
