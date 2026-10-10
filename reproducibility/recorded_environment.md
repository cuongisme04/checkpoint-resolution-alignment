# Recorded benchmark environment

Reported benchmark rows were collected on rented RunPod cloud GPU pods using
the same GPU model. The focused validation reran six BLIP Pet cells and all 24
Flickr30k cells in a clean checkout; retained historical Pet rows were not
cross-pod replicated.

| Component | Retained historical environment | Focused validation environment |
|---|---|---|
| Access model | Rented RunPod GPU pod | Rented RunPod GPU pod |
| GPU | NVIDIA RTX 6000 Ada Generation | NVIDIA RTX 6000 Ada Generation |
| GPU memory | 48 GB | 48 GB |
| GPU driver | 570.124.06 | 550.127.05 |
| PyTorch CUDA runtime | 12.1 | 12.4 |
| System CUDA compiler | 12.8 (`nvcc` 12.8.93) | Not separately recorded |
| Python | 3.12.3 | 3.11.10 |
| PyTorch | 2.4.1 | 2.4.1+cu124 |
| torchvision | 0.19.1 | 0.19.1 |
| open_clip_torch | 2.26.1 | 2.26.1 |
| transformers | 4.44.2 | 4.44.2 |
| Operating system | Ubuntu Linux, kernel 6.8.0-63-generic | Linux; exact distribution not retained |
| Environment capture time | 2026-05-22 16:46:19 UTC | 2026-07-14 (focused run) |
| CPU model | Not recorded | AMD EPYC 75F3 (RunPod telemetry) |
| Host RAM | Not recorded | 57.74 GiB (RunPod telemetry) |
| Storage configuration | Not recorded | 20 GB pod disk; 50 GB network-volume quota |
| Container image identifier | Not recorded | Not recorded |

Timing may reflect instance-specific driver, library, allocator, cloud-host,
and storage variability. The large backing-filesystem capacity visible through
Linux filesystem tools is not the user's network-volume quota; RunPod telemetry
reported the focused pod's persistent quota as 50 GB. Results should not be
assumed to transfer directly to consumer GPUs, CPUs, or edge hardware.
