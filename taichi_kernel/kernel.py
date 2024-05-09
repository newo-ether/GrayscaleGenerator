import taichi as ti
import taichi.math as tm
from taichi.math import (vec3, inf, pi)

@ti.dataclass
class Ray:
    origin : vec3
    dir : vec3
    t : float

@ti.dataclass
class Triangle:
    p1 : vec3
    p2 : vec3
    p3 : vec3

@ti.dataclass
class HitResult:
    isHit : ti.u1
    hitPoint : vec3
    t : float

@ti.func
def rand() -> float:
    return ti.random(dtype=float) * 2 - 1

@ti.func
def Intersect(ray : Ray, tri : Triangle) -> HitResult:
    norm = tm.normalize(tm.cross(tri.p2 - tri.p1, tri.p3 - tri.p2))
    barycenter = (tri.p1 + tri.p2 + tri.p3) / 3
    t = tm.dot(barycenter - ray.origin, norm) / tm.dot(ray.dir, norm)
    hit_res = HitResult(0, vec3(0), inf)
    if t >= 0 and t < ray.t:
        p = ray.origin + t * ray.dir
        b1 = tm.dot(tm.cross(tri.p2 - tri.p1, p - tri.p1), norm)
        b2 = tm.dot(tm.cross(tri.p3 - tri.p2, p - tri.p2), norm)
        b3 = tm.dot(tm.cross(tri.p1 - tri.p3, p - tri.p3), norm)
        if (b1 >= 0 and b2 >= 0 and b3 >= 0) or (b1 <= 0 and b2 <= 0 and b3 <= 0):
            hit_res = HitResult(1, p, t)

    return hit_res

@ti.func
def CheckIntersect(
        h : float,
        v : float,
        dh : float,
        dv : float,
        verts : ti.types.ndarray(),
        faces : ti.types.ndarray(),
        useFast : ti.u1
    ) -> HitResult:
    h += dh * rand() * 0.5 * 0.1
    v += dv * rand() * 0.5 * 0.1
    ray = Ray(origin=vec3(0),
              dir=tm.normalize(vec3(tm.cos(v) * tm.cos(h), tm.cos(v) * tm.sin(h), tm.sin(v))),
              t=inf)
    hit_res = HitResult(0, vec3(0), inf)

    ti.loop_config(serialize=True)
    for n in range(0, faces.shape[0]):
        p1 = vec3(verts[faces[n, 0], 0], verts[faces[n, 0], 1], verts[faces[n, 0], 2])
        p2 = vec3(verts[faces[n, 1], 0], verts[faces[n, 1], 1], verts[faces[n, 1], 2])
        p3 = vec3(verts[faces[n, 2], 0], verts[faces[n, 2], 1], verts[faces[n, 2], 2])
        result = Intersect(ray, Triangle(p1, p2, p3))
        if result.isHit == 1:
            hit_res = result
            ray.t = result.t
            if useFast == 1:
                break

    return hit_res

@ti.kernel
def CreateGrayscaleMap(
        frame : ti.types.ndarray(),
        verts : ti.types.ndarray(),
        faces : ti.types.ndarray(),
        useFast : ti.u1
    ):
    dh = 2 * pi / frame.shape[1]
    dv = pi / frame.shape[0]
    for i,j in ti.ndrange(frame.shape[0], frame.shape[1]):
        h = j * dh
        v = pi / 2 - i * dv
        hit_res = CheckIntersect(h, v, dh, dv, verts, faces, useFast)
        if hit_res.isHit == 0:
            hit_res = CheckIntersect(h, v, dh, dv, verts, faces, useFast)
            if hit_res.isHit == 0:
                hit_res = CheckIntersect(h, v, dh, dv, verts, faces, useFast)
                if hit_res.isHit == 0:
                    frame[i, j] = 0.0
                else:
                    frame[i, j] = hit_res.t
            else:
                frame[i, j] = hit_res.t
        else:
            frame[i, j] = hit_res.t
