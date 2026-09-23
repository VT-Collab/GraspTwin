# GraspTwin: Semantic Grasping with VLM-Seeded Thompson-Sampling Pose Optimization

Code accompanying our submission. A vision-language model (Gemini
Robotics-ER) decides *what* to grasp and *how* (object, part, approach side);
a batched Bayesian optimizer then refines that seed into a 6-DoF grasp pose by
scoring candidates in a physics simulator, and the best pose is executed on a
Franka arm.

## What is here

| Path | Contents |
|---|---|
| `main.py` | The method. `run_once()` runs one full perceive → ask → optimize → deploy cycle. |
| `semantic_grasp/perception/` | Scene understanding: detection, segmentation, 3-D reconstruction, VLM queries, grasp-seed geometry, multi-view rendering. |
| `semantic_grasp/ik.py`, `isaac.py` | Clients for the batched IK / trajectory solver and the physics grasp evaluator. |
| `semantic_grasp/robot.py`, `camera.py`, `camera_model.py` | Clients for the arm and the RGB-D camera, plus the camera intrinsics/distortion model. |
| `semantic_grasp/config.py` | All constants, endpoints, camera calibration, and pipeline toggles. |
| `config/*.yaml` | The exact prompts sent to the VLM at each stage. |
| `servers/isaac_server.py` | Isaac Lab grasp evaluator: builds the digital twin, closes the gripper on thousands of candidate poses in parallel, and returns the lift / stability metrics that form the optimizer's cost. `run_isaac_server.sh` is its supervised launcher. |
| `servers/ik_server.py` | cuRobo server: batched collision-aware IK for candidate poses and trajectory planning for the chosen one. `curobo_local_edits.diff` is the small patch we applied to cuRobo. |
| `servers/perception_server.py`, `sam3d_server.py` | LangSAM segmentation and SAM 3D object reconstruction. |
| `config/grippers/franka_rubber/` | Gripper model (URDF, USD, collision spheres) shared by the IK and Isaac servers. |

## Reading guide

Start at `run_once()` in `main.py`. The Thompson-sampling search (Sobol
initialization, GP surrogate, `PathwiseThompsonSampling` acquisition, batched
scoring) is in the same file. The prompts in `config/` are the semantic half of
the method and are short enough to read in full.

## Running

`main.py` talks over ZMQ to four services in `servers/`, each started in its
own process. Endpoints are set in `semantic_grasp/config.py`.

```bash
conda create -n semantic_grasp python=3.10
conda activate semantic_grasp
pip install -r requirements.txt

python servers/perception_server.py            # needs lang-sam
python servers/sam3d_server.py                 # needs SAM 3D
python servers/ik_server.py                    # needs cuRobo (see servers/curobo_local_edits.diff)
servers/run_isaac_server.sh --headless         # needs Isaac Sim 4.5 + Isaac Lab

export GEMINI_API_KEY=...
python main.py
```

The simulator-side dependencies (Isaac Sim / Isaac Lab, cuRobo, lang-sam,
SAM 3D, pytorch3d) are not on PyPI in a form `pip install` can resolve, so
they are not in `requirements.txt`; install each per its own instructions.
The experiments ran with Isaac Sim 4.5.0 and Python 3.10.

cuRobo needs three small local edits, recorded in
`servers/curobo_local_edits.diff` and explained at the top of
`servers/ik_server.py`. One of them (the mesh SDF query radius) changes
collision behaviour and is required to reproduce our results:

```bash
git clone https://github.com/NVlabs/curobo && cd curobo
git checkout ca941586c33b8482ed9c0e74d60f23efd64b516a
git apply /path/to/this/repo/servers/curobo_local_edits.diff
pip install -e .
```

Two hardware-side services are not included: the RGB-D camera server and the
trajectory-replay server on the robot PC. Their clients
(`semantic_grasp/camera.py`, `semantic_grasp/robot.py`) document the message
format each expects, so they are straightforward to reimplement for another
camera or arm.

## License

MIT. See `LICENSE`.
