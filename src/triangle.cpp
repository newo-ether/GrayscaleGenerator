// triangle.cpp

#include "triangle.h"
#include "vector2d.h"
#include "vector3d.h"

Triangle::Triangle() {}

Triangle::Triangle(Vector3D p0,
                   Vector3D p1,
                   Vector3D p2,
                   Vector2D uv0,
                   Vector2D uv1,
                   Vector2D uv2) {
    p[0] = p0;
    p[1] = p1;
    p[2] = p2;

    uv[0] = uv0;
    uv[1] = uv1;
    uv[2] = uv2;
}

CLTriangle Triangle::serialize() {
    return CLTriangle {
        .p = {
            p[0].serialize(),
            p[1].serialize(),
            p[2].serialize()
        },
        .uv = {
            uv[0].serialize(),
            uv[1].serialize(),
            uv[2].serialize()
        }
    };
}

Triangle Triangle::deserialize(CLTriangle tri) {
    return Triangle(Vector3D::deserialize(tri.p[0]),
                    Vector3D::deserialize(tri.p[1]),
                    Vector3D::deserialize(tri.p[2]),
                    Vector2D::deserialize(tri.uv[0]),
                    Vector2D::deserialize(tri.uv[1]),
                    Vector2D::deserialize(tri.uv[2]));
}

Vector3D Triangle::centroid() {
    Vector3D center;
    for (int i = 0; i < 3; i++) {
        center += p[i];
    }
    return center / 3;
}
