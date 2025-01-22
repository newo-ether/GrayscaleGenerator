// kernel.cl

#include "bvh.cl"
#include "bound3d.cl"
#include "intersectinfo.cl"
#include "ray.cl"
#include "triangle.cl"
#include "constant.cl"
#include "random.cl"

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_TRUE |
                                    CLK_ADDRESS_REPEAT |
                                    CLK_FILTER_LINEAR;

__kernel void RenderKernel(__global const BVHNode *bvhNodes,
                           const int bvhNodeCount,
                           __global const Triangle *triangles,
                           const int triangleCount,
                           __global float *outputBuffer,
                           const int width,
                           const int height,
                           const int sampleNum,
                           read_only image2d_t colorMapImage,
                           read_only image2d_t normalMapImage,
                           __global uchar4 *outputColorMapBuffer,
                           __global uchar4 *outputNormalMapBuffer) {

    int index = get_global_id(0);
    if (index < width * height) {
        Random random = RandomInit(index);
        int i = index / width;
        int j = index % width;

        float dhAngle = TWOPI / width;
        float dvAngle = PI / height;
        float hAngle = j * dhAngle;
        float vAngle = PI / 2 - i * dvAngle;

        float tHit = 0.0f;
        uchar4 colorMapValue = (uchar4){0, 0, 0, 255};
        uchar4 normalMapValue = (uchar4){0, 0, 0, 255};

        if (sampleNum == 0) {
            IntersectInfo isect = IntersectInfoInitNone();
            float2 sample = (float2){hAngle + 0.5f * dhAngle,
                                     vAngle - 0.5f * dvAngle};

            Ray ray = RayInit((float3){0.0f, 0.0f, 0.0f},
                              normalize((float3){cos(sample.y) * cos(sample.x),
                                                 cos(sample.y) * sin(sample.x),
                                                 sin(sample.y)}),
                              0.0f,
                              INFINITY);

            isect = BVHIntersect(bvhNodes, bvhNodeCount, triangles, ray);

            tHit = isect.isHit ? isect.tHit : 0.0f;

            if (outputColorMapBuffer) {
                colorMapValue = isect.isHit
                                ? (uchar4){convert_uchar3(read_imageui(colorMapImage, imageSampler, isect.hitUV).xyz), 255}
                                : (uchar4){0, 0, 0, 255};
            }

            if (outputNormalMapBuffer) {
                normalMapValue = isect.isHit
                                 ? (uchar4){convert_uchar3(read_imageui(normalMapImage, imageSampler, isect.hitUV).xyz), 255}
                                 : (uchar4){0, 0, 0, 255};
            }
        }
        else {
            float subdhAngle = dhAngle / sampleNum;
            float subdvAngle = dvAngle / sampleNum;

            float tHitSample = 0.0f;
            float3 colorMapSample = (float3){0.0f, 0.0f, 0.0f};
            float3 normalMapSample = (float3){0.0f, 0.0f, 0.0f};
            int sampleCount = 0;

            for (int i = 0; i < sampleNum; i++) {
                for (int j = 0; j < sampleNum; j++) {
                    float2 sample = (float2){hAngle + (i + Random01(&random)) * subdhAngle,
                                             vAngle - (j + Random01(&random)) * subdvAngle};

                    Ray ray = RayInit((float3){0.0f, 0.0f, 0.0f},
                                      normalize((float3){cos(sample.y) * cos(sample.x),
                                                         cos(sample.y) * sin(sample.x),
                                                         sin(sample.y)}),
                                      0.0f,
                                      INFINITY);

                    IntersectInfo isect = BVHIntersect(bvhNodes, bvhNodeCount, triangles, ray);
                    if (isect.isHit) {
                        tHitSample += isect.tHit;
                        if (outputColorMapBuffer) {
                            colorMapSample += convert_float3(read_imageui(colorMapImage, imageSampler, isect.hitUV).xyz);
                        }
                        if (outputNormalMapBuffer) {
                            normalMapSample += convert_float3(read_imageui(normalMapImage, imageSampler, isect.hitUV).xyz);
                        }
                        sampleCount++;
                    }
                }
            }
            tHit = sampleCount == 0 ? 0.0f : (tHitSample / sampleCount);

            colorMapValue = sampleCount == 0
                            ? (uchar4){0, 0, 0, 255}
                            : (uchar4){convert_uchar3(colorMapSample / sampleCount), 255};

            normalMapValue = sampleCount == 0
                             ? (uchar4){0, 0, 0, 255}
                             : (uchar4){convert_uchar3(normalMapSample / sampleCount), 255};
        }

        outputBuffer[index] = tHit;

        if (outputColorMapBuffer) {
            outputColorMapBuffer[index] = colorMapValue;
        }
        if (outputNormalMapBuffer) {
            outputNormalMapBuffer[index] = normalMapValue;
        }
    }
}

__kernel void FilterKernel(__global const float *inputBuffer,
                           __global float *outputBuffer,
                           __global const uchar4 *inputColorMapBuffer,
                           __global uchar4 *outputColorMapBuffer,
                           __global const uchar4 *inputNormalMapBuffer,
                           __global uchar4 *outputNormalMapBuffer,
                           const int width,
                           const int height) {

    int index = get_global_id(0);
    if (index < width * height) {
        int x = index % width;
        int y = index / width;
        float currentPixel = inputBuffer[index];
        if (currentPixel > 0.0f) {
            outputBuffer[index] = currentPixel;
            if (inputColorMapBuffer) {
                outputColorMapBuffer[index] = inputColorMapBuffer[index];
            }
            if (inputNormalMapBuffer) {
                outputNormalMapBuffer[index] = inputNormalMapBuffer[index];
            }
        }
        else {
            int count = 0;
            float pixelSample = 0.0f;
            float3 colorMapSample = (float3){0.0f, 0.0f, 0.0f};
            float3 normalMapSample = (float3){0.0f, 0.0f, 0.0f};

            for (int i = x - 1; i <= x + 1; i++) {
                for (int j = y - 1; j <= y + 1; j++) {
                    int n = i, m = j;
                    n = (n < 0 ? n + width : n);
                    n = (n >= width ? n - width : n);
                    m = (m < 0 ? m + height : m);
                    m = (m >= height ? m - height : m);

                    int currentIndex = m * width + n;
                    float pixel = inputBuffer[currentIndex];
                    if (pixel > 0.0f) {
                        pixelSample += pixel;
                        if (inputColorMapBuffer) {
                            colorMapSample += convert_float3(inputColorMapBuffer[currentIndex].xyz);
                        }
                        if (inputNormalMapBuffer) {
                            normalMapSample += convert_float3(inputNormalMapBuffer[currentIndex].xyz);
                        }
                        count++;
                    }
                }
            }

            outputBuffer[index] = count == 0 ? 0.0f : (pixelSample / count);

            if (outputColorMapBuffer) {
                outputColorMapBuffer[index] = count == 0
                                              ? (uchar4){0, 0, 0, 255}
                                              : (uchar4){convert_uchar3(colorMapSample / count), 255};
            }
            if (outputNormalMapBuffer) {
                outputNormalMapBuffer[index] = count == 0
                                               ? (uchar4){0, 0, 0, 255}
                                               : (uchar4){convert_uchar3(normalMapSample / count), 255};
            }
        }
    }
}

__kernel void MinMaxKernel(__global const float *buffer,
                           const int bufferLength,
                           __global uint *minValue,
                           __global uint *maxValue) {

    int index = get_global_id(0);
    if (index < bufferLength) {
        uint value = as_uint(max(0.0f, buffer[index]));
        atomic_min(minValue, value);
        atomic_max(maxValue, value);
    }
}
