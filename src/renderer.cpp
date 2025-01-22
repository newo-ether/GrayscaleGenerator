// renderer.cpp

#include <vector>
#include <string>
#include <memory>
#include <limits>

#include "renderer.h"
#include "opencl.h"
#include "opencl_interface.h"
#include "bvh.h"

Renderer::Renderer(OpenCLInterface &clInterface):
    clInterface(clInterface),
    isBVHBufferSet(false),
    isTriangleBufferSet(false),
    isColorMapSet(false),
    isNormalMapSet(false),
    isRenderColorMap(false),
    isRenderNormalMap(false),
    isParameterSet(false) {}

int Renderer::setBVHBuffer(const std::vector<CLBVHNode> &bvhNodes) {
    cl_int errorCode = CL_SUCCESS;
    bvhBuffer = cl::Buffer(clInterface.getContext(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(CLBVHNode) * bvhNodes.size(),
                           (void *) bvhNodes.data(),
                           &errorCode);

    if (errorCode != CL_SUCCESS) {
        return -1;
    }

    bvhBufferSize = static_cast<int>(bvhNodes.size());
    isBVHBufferSet = true;
    return 0;
}

int Renderer::setTriangleBuffer(const std::vector<CLTriangle> &triangles) {
    cl_int errorCode = CL_SUCCESS;
    triangleBuffer = cl::Buffer(clInterface.getContext(),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(CLTriangle) * triangles.size(),
                                (void *) triangles.data(),
                                &errorCode);

    if (errorCode != CL_SUCCESS) {
        return -1;
    }

    triangleBufferSize = static_cast<int>(triangles.size());
    isTriangleBufferSet = true;
    return 0;
}

int Renderer::setColorMapBuffer(std::shared_ptr<unsigned char[]> colorMapArray, int width, int height) {
    cl_int errorCode = CL_SUCCESS;
    int maxLevel = static_cast<int>(std::ceil(std::log2(std::max(width, height))));
    colorMapImage = cl::Image2D(clInterface.getContext(),
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                (void *) colorMapArray.get(),
                                &errorCode);

    if (errorCode != CL_SUCCESS) {
        return -1;
    }

    colorMapWidth = width;
    colorMapHeight = height;
    colorMapMipMapLevel = maxLevel;
    isColorMapSet = true;
    return 0;
}

int Renderer::setNormalMapBuffer(std::shared_ptr<unsigned char[]> normalMapArray, int width, int height) {
    cl_int errorCode = CL_SUCCESS;
    int maxLevel = static_cast<int>(std::ceil(std::log2(std::max(width, height))));
    normalMapImage = cl::Image2D(clInterface.getContext(),
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                 width,
                                 height,
                                 0,
                                 (void *) normalMapArray.get(),
                                 &errorCode);

    if (errorCode != CL_SUCCESS) {
        return -1;
    }

    normalMapWidth = width;
    normalMapHeight = height;
    normalMapMipMapLevel = maxLevel;
    isNormalMapSet = true;
    return 0;
}

void Renderer::setParameter(int width, int height, int sampleNum, bool isRenderColorMap, bool isRenderNormalMap) {
    this->width = width;
    this->height = height;
    this->sampleNum = sampleNum;
    this->isRenderColorMap = isRenderColorMap;
    this->isRenderNormalMap = isRenderNormalMap;
    isParameterSet = true;
}

void Renderer::run() {
    cl_int errorCode = CL_SUCCESS;

    if (!(isBVHBufferSet && isTriangleBufferSet && isParameterSet)) {
        returnError(-1);
        return;
    }

    if (!(isColorMapSet && isRenderColorMap)) {
        colorMapImage = cl::Image2D(clInterface.getContext(),
                                    CL_MEM_READ_ONLY,
                                    cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                    1,
                                    1,
                                    0,
                                    nullptr,
                                    &errorCode);

        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    if (!(isNormalMapSet && isRenderNormalMap)) {
        normalMapImage = cl::Image2D(clInterface.getContext(),
                                     CL_MEM_READ_ONLY,
                                     cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                     1,
                                     1,
                                     0,
                                     nullptr,
                                     &errorCode);

        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    // Dispatch Render Kernel
    cl::Buffer grayscaleMapBuffer = cl::Buffer(clInterface.getContext(),
                                               CL_MEM_READ_WRITE,
                                               sizeof(float) * width * height,
                                               nullptr,
                                               &errorCode);

    if (errorCode != CL_SUCCESS) {
        returnError(-4);
        return;
    }

    cl::Buffer colorMapBuffer;
    if (isColorMapSet && isRenderColorMap) {
        colorMapBuffer = cl::Buffer(clInterface.getContext(),
                                    CL_MEM_READ_WRITE,
                                    sizeof(cl_uchar4) * width * height,
                                    nullptr,
                                    &errorCode);

        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    cl::Buffer normalMapBuffer;
    if (isNormalMapSet && isRenderNormalMap) {
        normalMapBuffer = cl::Buffer(clInterface.getContext(),
                                     CL_MEM_READ_WRITE,
                                     sizeof(cl_uchar4) * width * height,
                                     nullptr,
                                     &errorCode);

        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    cl::Kernel renderKernel = clInterface.getKernel("RenderKernel");

    renderKernel.setArg(0, bvhBuffer);
    renderKernel.setArg(1, bvhBufferSize);
    renderKernel.setArg(2, triangleBuffer);
    renderKernel.setArg(3, triangleBufferSize);
    renderKernel.setArg(4, grayscaleMapBuffer);
    renderKernel.setArg(5, width);
    renderKernel.setArg(6, height);
    renderKernel.setArg(7, sampleNum);
    renderKernel.setArg(8, colorMapImage);
    renderKernel.setArg(9, normalMapImage);
    if (isColorMapSet && isRenderColorMap) {
        renderKernel.setArg(10, colorMapBuffer);
    }
    else {
        renderKernel.setArg(10, nullptr);
    }
    if (isNormalMapSet && isRenderNormalMap) {
        renderKernel.setArg(11, normalMapBuffer);
    }
    else {
        renderKernel.setArg(11, nullptr);
    }

    errorCode = clInterface.getCommandQueue()
                           .enqueueNDRangeKernel(renderKernel,
                                                 cl::NDRange(0),
                                                 cl::NDRange(width * height),
                                                 cl::NullRange,
                                                 nullptr,
                                                 nullptr);
    if (errorCode != CL_SUCCESS) {
        returnError(-2);
        return;
    }

    // Dispatch Filter Kernel
    cl::Buffer filteredGrayscaleMapBuffer = cl::Buffer(clInterface.getContext(),
                                                       CL_MEM_READ_WRITE,
                                                       sizeof(float) * width * height,
                                                       nullptr,
                                                       &errorCode);
    if (errorCode != CL_SUCCESS) {
        returnError(-4);
        return;
    }

    cl::Buffer filteredColorMapBuffer;
    if (isColorMapSet && isRenderColorMap) {
        filteredColorMapBuffer = cl::Buffer(clInterface.getContext(),
                                            CL_MEM_READ_WRITE,
                                            sizeof(cl_uchar4) * width * height,
                                            nullptr,
                                            &errorCode);
        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    cl::Buffer filteredNormalMapBuffer;
    if (isNormalMapSet && isRenderNormalMap) {
        filteredNormalMapBuffer = cl::Buffer(clInterface.getContext(),
                                             CL_MEM_READ_WRITE,
                                             sizeof(cl_uchar4) * width * height,
                                             nullptr,
                                             &errorCode);
        if (errorCode != CL_SUCCESS) {
            returnError(-4);
            return;
        }
    }

    cl::Kernel filterKernel = clInterface.getKernel("FilterKernel");

    filterKernel.setArg(0, grayscaleMapBuffer);
    filterKernel.setArg(1, filteredGrayscaleMapBuffer);
    if (isColorMapSet && isRenderColorMap) {
        filterKernel.setArg(2, colorMapBuffer);
        filterKernel.setArg(3, filteredColorMapBuffer);
    }
    else {
        filterKernel.setArg(2, nullptr);
        filterKernel.setArg(3, nullptr);
    }
    if (isNormalMapSet && isRenderNormalMap) {
        filterKernel.setArg(4, normalMapBuffer);
        filterKernel.setArg(5, filteredNormalMapBuffer);
    }
    else {
        filterKernel.setArg(4, nullptr);
        filterKernel.setArg(5, nullptr);
    }
    filterKernel.setArg(6, width);
    filterKernel.setArg(7, height);

    errorCode = clInterface.getCommandQueue()
                           .enqueueNDRangeKernel(filterKernel,
                                                 cl::NDRange(0),
                                                 cl::NDRange(width * height),
                                                 cl::NullRange,
                                                 nullptr,
                                                 nullptr);

    if (errorCode != CL_SUCCESS) {
        returnError(-2);
        return;
    }

    // Dispatch Min Max Kernel
    cl::Buffer minBuffer = cl::Buffer(clInterface.getContext(),
                                      CL_MEM_READ_WRITE,
                                      sizeof(cl_uint),
                                      nullptr,
                                      nullptr);

    cl::Buffer maxBuffer = cl::Buffer(clInterface.getContext(),
                                      CL_MEM_READ_WRITE,
                                      sizeof(cl_uint),
                                      nullptr,
                                      nullptr);

    cl_uint uintMinData = std::numeric_limits<uint>::max();
    cl_uint uintMaxData = 0u;

    errorCode = clInterface.getCommandQueue()
                           .enqueueWriteBuffer(minBuffer, true, 0, sizeof(uint), &uintMinData);

    if (errorCode != CL_SUCCESS) {
        returnError(-4);
        return;
    }

    errorCode = clInterface.getCommandQueue()
                           .enqueueWriteBuffer(maxBuffer, true, 0, sizeof(uint), &uintMaxData);

    if (errorCode != CL_SUCCESS) {
        returnError(-4);
        return;
    }

    cl_int clBufferLength = width * height;

    cl::Kernel minMaxKernel = clInterface.getKernel("MinMaxKernel");

    minMaxKernel.setArg(0, filteredGrayscaleMapBuffer);
    minMaxKernel.setArg(1, clBufferLength);
    minMaxKernel.setArg(2, minBuffer);
    minMaxKernel.setArg(3, maxBuffer);

    errorCode = clInterface.getCommandQueue()
                           .enqueueNDRangeKernel(minMaxKernel,
                                                 cl::NDRange(0),
                                                 cl::NDRange(width * height),
                                                 cl::NullRange,
                                                 nullptr,
                                                 nullptr);

    if (errorCode != CL_SUCCESS) {
        returnError(-2);
        return;
    }

    // Read Min Max Buffer
    cl_uint clMin, clMax;

    errorCode = clInterface.getCommandQueue()
                           .enqueueReadBuffer(minBuffer,
                                              true,
                                              0,
                                              sizeof(cl_uint),
                                              &clMin,
                                              nullptr,
                                              nullptr);

    if (errorCode != CL_SUCCESS) {
        returnError(-3);
        return;
    }

    errorCode = clInterface.getCommandQueue()
                           .enqueueReadBuffer(maxBuffer,
                                              true,
                                              0,
                                              sizeof(cl_uint),
                                              &clMax,
                                              nullptr,
                                              nullptr);

    if (errorCode != CL_SUCCESS) {
        returnError(-3);
        return;
    }

    // Read Image Buffer
    std::shared_ptr<float[]> outputGrayscaleMapArray(new float[width * height]);
    errorCode = clInterface.getCommandQueue()
                           .enqueueReadBuffer(filteredGrayscaleMapBuffer,
                                              true,
                                              0,
                                              sizeof(float) * width * height,
                                              outputGrayscaleMapArray.get(),
                                              nullptr,
                                              nullptr);

    std::shared_ptr<unsigned char[]> outputColorMapArray;
    if (isColorMapSet && isRenderColorMap) {
        outputColorMapArray.reset(new unsigned char[width * height * 4]);
        errorCode = clInterface.getCommandQueue()
                               .enqueueReadBuffer(filteredColorMapBuffer,
                                                  true,
                                                  0,
                                                  sizeof(cl_uchar4) * width * height,
                                                  outputColorMapArray.get(),
                                                  nullptr,
                                                  nullptr);
    }

    std::shared_ptr<unsigned char[]> outputNormalMapArray;
    if (isNormalMapSet && isRenderNormalMap) {
        outputNormalMapArray.reset(new unsigned char[width * height * 4]);
        errorCode = clInterface.getCommandQueue()
                               .enqueueReadBuffer(filteredNormalMapBuffer,
                                                  true,
                                                  0,
                                                  sizeof(cl_uchar4) * width * height,
                                                  outputNormalMapArray.get(),
                                                  nullptr,
                                                  nullptr);
    }

    if (errorCode != CL_SUCCESS) {
        returnError(-3);
        return;
    }

    emit finishedRender(0,
                        outputGrayscaleMapArray,
                        outputColorMapArray,
                        outputNormalMapArray,
                        width,
                        height,
                        *reinterpret_cast<float*>(&clMin),
                        *reinterpret_cast<float*>(&clMax));
}

void Renderer::returnError(int result) {
    emit finishedRender(result,
                        std::shared_ptr<float[]>(),
                        std::shared_ptr<unsigned char[]>(),
                        std::shared_ptr<unsigned char[]>(),
                        0,
                        0,
                        0.0f,
                        0.0f);
}
