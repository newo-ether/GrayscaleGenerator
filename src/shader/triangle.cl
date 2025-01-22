// Triangle.hlsl

#ifndef __TRIANGLE__
#define __TRIANGLE__

#include "intersectinfo.cl"
#include "ray.cl"

typedef struct Triangle {
    float3 p[3];
    float2 uv[3];
    char padding[8];
} Triangle;

IntersectInfo TriangleIntersect(Triangle tri, Ray ray) {
    float3 e0 = tri.p[0] - tri.p[2];
    float3 e1 = tri.p[1] - tri.p[0];
    float3 e2 = tri.p[2] - tri.p[1];
    
    float3 c0 = cross(e0, tri.p[0] + tri.p[2]);
    float3 c1 = cross(e1, tri.p[1] + tri.p[0]);
    float3 c2 = cross(e2, tri.p[2] + tri.p[1]);

    float b0 = dot(ray.dir, c0) / dot(e1, c0);
    float b1 = dot(ray.dir, c1) / dot(e2, c1);
    float b2 = dot(ray.dir, c2) / dot(e0, c2);
    
    IntersectInfo isect;

    if (b0 >= 0.0f && b1 >= 0.0f && b2 >= 0.0f) {
        float sum = b0 + b1 + b2;
        b0 = b0 / sum;
        b1 = b1 / sum;
        b2 = b2 / sum;

        float3 p = b2 * tri.p[0] + b0 * tri.p[1] + b1 * tri.p[2];
        float t = dot(ray.dir, p);
        if (t > ray.tMin && t < ray.tMax) {
            float3 normal = normalize(cross(e1, e2));
        
            float2 hitUV = b2 * tri.uv[0] + b0 * tri.uv[1] + b1 * tri.uv[2];
            hitUV.y = 1.0f - hitUV.y;

            isect = IntersectInfoInit(t, hitUV, normal);
        }
        else {
            isect = IntersectInfoInitNone();
        }
    }
    else {
        isect = IntersectInfoInitNone();
    }

    return isect;
}

bool TriangleIsIntersect(Triangle tri, Ray ray) {
    float3 e0 = tri.p[0] - tri.p[2];
    float3 e1 = tri.p[1] - tri.p[0];
    float3 e2 = tri.p[2] - tri.p[1];
    
    float3 normal = normalize(cross(e1, e2));
    
    float t = dot(tri.p[0] - ray.origin, normal) / dot(ray.dir, normal);
    
    bool isHit;

    if (t <= ray.tMin || t >= ray.tMax) {
        isHit = false;
    }
    else {
        float3 p = RayAt(ray, t);
    
        float b0 = dot(cross(e0, p - tri.p[2]), normal);
        float b1 = dot(cross(e1, p - tri.p[0]), normal);
        float b2 = dot(cross(e2, p - tri.p[1]), normal);
    
        float sum = b0 + b1 + b2;
        float epsilon = fabs(sum) * 0.001f;
        if ((b0 >= -epsilon && b1 >= -epsilon && b2 >= -epsilon) || (b0 <= epsilon && b1 <= epsilon && b2 <= epsilon)) {
            isHit = true;
        }
        else {
            isHit = false;
        }
    }

    return isHit;
}

#endif // __TRIANGLE__