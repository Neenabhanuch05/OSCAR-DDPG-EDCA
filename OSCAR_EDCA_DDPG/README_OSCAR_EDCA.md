# OSCAR + EDCA (Ubuntu / ns-3.29)

This project is an executable extension of the original OSCAR ns-3/ns3-gym project.

**What changes:**
- OSCAR's original DDPG agent is retained.
- Instead of controlling one BE contention window, the DDPG action is used as an EDCA **CW compensation value (CWcomp)** in the range 0–896.
- Four EDCA access categories are configured: VO, VI, BE and BK.
- Stations are distributed across the four ACs using IPv4 ToS/DS-field values.
- The original OSCAR observation/reward/training loop and W&B logging are retained.

The uploaded PDCF-DRL paper defines EDCA parameters as AIFSN/CWmin/CWmax/TXOP and uses a CW compensation value to modify the four AC ranges. This implementation keeps those EDCA rules but substitutes OSCAR's DDPG controller for the paper's DQN controller.

## EDCA configuration

| AC | AIFSN | CWmin | CWmax | TXOP |
|---|---:|---:|---:|---:|
| AC_VO | 2 | 7 | 15 + CWcomp | 1.504 ms |
| AC_VI | 2 | 15 + CWcomp | 31 + CWcomp | 3.008 ms |
| AC_BE | 3 | 31 + CWcomp | 1023 | 0 |
| AC_BK | 7 | 31 + CWcomp | 1023 | 0 |

Values are clamped to ns-3's 1023 CW maximum.

## Traffic-to-AC mapping

- `0xE0` -> user priority 7 -> AC_VO
- `0xA0` -> user priority 5 -> AC_VI
- `0x60` -> user priority 3 -> AC_BE
- `0x40` -> user priority 2 -> AC_BK

Stations use these classes in round-robin order (`station_id % 4`).

## Ubuntu / ns-3.29 installation

### Option A — easiest

1. Download this ZIP.
2. On Ubuntu:

```bash
cd ~
ls -lh OSCAR-EDCA-Ubuntu-final.zip
unzip OSCAR-EDCA-Ubuntu-final.zip
cd OSCAR-EDCA-Ubuntu-final/OSCAR-contention-window-RL-main
```

3. Back up your existing OSCAR project, then copy this folder into:

```text
~/workspace/ns-allinone-3.29/ns-3.29/scratch/linear-mesh/
```

4. Activate your existing environment:

```bash
cd ~/workspace/ns-allinone-3.29/ns-3.29
source myenv/bin/activate
```

5. Run:

```bash
cd scratch/linear-mesh
python OSCAR_train.py
```

### Option B — installer

From the extracted project directory:

```bash
chmod +x install_edca.sh
./install_edca.sh ~/workspace/ns-allinone-3.29/ns-3.29
```

The script creates a timestamped backup of the existing `scratch/linear-mesh` before installing the EDCA version.

## Training settings

The default training script uses:
- station counts: 40, 45, 50
- 12 episodes per station count
- 15 s simulation per episode
- 10 ms decision epoch
- DDPG actor/critic from the original OSCAR implementation
- DDPG action scaled from `[-1, 1]` to `[0, 896]` CWcomp

You can change `N_WIFI_LIST`, `EPISODE_COUNT`, `simTime`, or `stepTime` in `OSCAR_train.py`.

## Important distinction

This is **OSCAR + EDCA**, not the PDCF-DRL algorithm reproduced exactly. The uploaded PDCF-DRL paper uses DQN to select CWcomp, whereas this extension intentionally keeps OSCAR's DDPG controller and applies its action to EDCA parameters.
