# Network Migration

The HIL-SERL network stack is now merged into a single project-local package:

```text
rl-serl/rl_launcher
```

`rl_launcher` is the only network/training entry package. It contains the former
upstream algorithm implementation directly, including agents, networks, vision
encoders, replay buffers, optimizer helpers, launcher utilities, and
task-agnostic wrappers.

New code should import through `rl_launcher`, for example:

```python
from rl_launcher.agents import make_sac_pixel_agent_hybrid_dual_arm
from rl_launcher.networks import create_classifier, load_classifier_func
from rl_launcher.data import MemoryEfficientReplayBufferDataStore
from rl_launcher.wrappers import SERLObsWrapper, ChunkingWrapper
```

## Components

```text
rl_launcher/agents      SAC, BC, hybrid single/dual-arm agents
rl_launcher/networks    actor, critic, gripper critic, classifier, MLP
rl_launcher/vision      ResNet encoder and image augmentations
rl_launcher/common      TrainState, ModuleDict, optimizer, typing helpers
rl_launcher/utils       launcher factories, training utilities, timers
rl_launcher/data        replay buffers and data stores
rl_launcher/wrappers    task-agnostic SERL observation/chunking wrappers
```

The old `serl_launcher` package directory was removed. Runtime path entries for
external `hil-serl` code were also removed from `examples/compat.py`.

## OpenArm-Specific Replacements

Only two `franka_env` dependencies were needed by `rl-serl`, so they were
replaced with OpenArm-local modules instead of copying the full Franka stack:

```text
openarm_env/utils/transformations.py
openarm_env/envs/reward_wrappers.py
```

## Remaining External Python Dependencies

The network code still needs its normal Python environment, including JAX, Flax,
Optax, Distrax, Chex, Agentlace, TensorFlow-related packages for pretrained
encoder loading, OpenCV, WandB, and replay-buffer utilities. These are
environment dependencies, not source-code dependencies on `hil-serl`.
