# Neural Rendering for Simulation — 3D Gaussian Splatting from a Real Capture

A self-directed project: capture a real-world scene on video, run the full
structure-from-motion → 3D Gaussian Splatting pipeline, and render photorealistic
novel viewpoints the camera never actually saw. Built to get hands-on with neural
rendering as a route to **synthetic sensor data** for closed-loop simulation.

Captured a short handheld video of a small object, recovered camera poses with
COLMAP, trained a Gaussian-splatting model with nerfstudio's `splatfacto`, and
rendered an interpolated camera path through the reconstructed scene.

## Result

![novel-view render](render.gif)

*A camera path gliding through the reconstructed 3D scene, including viewpoints not
present in the original capture.*

## Pipeline

```
phone video
  -> ns-process-data video   # COLMAP: structure-from-motion recovers camera poses
  -> ns-train splatfacto      # train the 3D gaussians against the real frames
  -> ns-render interpolate    # render novel views along an interpolated camera path
  -> ns-export gaussian-splat # export the trained scene as a .ply
```

**Stack:** nerfstudio (`splatfacto`), gsplat CUDA backend, COLMAP for pose
estimation, PyTorch 2.7 + CUDA 12.8.

## Run details

- Video frames extracted: 309
- Cameras registered by COLMAP: 309 / 309 (100%)
- Training steps: 30,000
- Render: 290-frame interpolated camera path
- Hardware: NVIDIA RTX 5080 (Blackwell), local Windows build

## What I took away

- A **3D Gaussian** stores position, covariance (shape + orientation), color
  (spherical harmonics), and opacity. A scene is millions of them.
- **Splatting** = projecting those 3D gaussians onto the 2D image plane for a given
  camera pose, then rasterizing to per-pixel color. Rasterization is cheap on GPUs,
  so rendering is real-time — the key advantage over NeRF's ray-marching.
- **Poses come first:** you can't train without knowing where each frame was shot
  from, which is why COLMAP (structure-from-motion) runs before training. Good
  reconstruction depends on textured subjects and heavy frame-to-frame overlap.
- Training is a differentiable-rendering loop: render the current gaussians, compare
  to the real frame, gradient-descend the parameters, densify/prune along the way.
- **Why this matters for simulation:** a splat trained on real captures can render
  viewpoints the camera never saw — a path to photorealistic synthetic sensor data
  with a smaller sim-to-real gap than hand-built simulation.

## Notes / limitations

- Reflective, transparent, and low-texture surfaces reconstruct poorly; textured
  subjects on a cluttered background register far better in COLMAP.
- Built on a very new GPU (RTX 5080) ahead of the Windows toolchain — required
  pinning PyTorch to a cu128 build compatible with the gsplat CUDA extension.

## How to run

    # Python 3.10, NVIDIA GPU, CUDA 12.8
    pip install nerfstudio

    ns-process-data video --data myscene.mp4 --output-dir processed/
    ns-train splatfacto --data processed/
    ns-render interpolate --load-config outputs/<run>/config.yml --output-path render.mp4
    ns-export gaussian-splat --load-config outputs/<run>/config.yml --output-dir exports/
