Import (GitHub / GHCR): UC3.3_SOMES_sequential.json
Same file also as uc33_somes.customization
SPICE / Harbor: uc33_somes.spice.customization

Version: 0.3.1
DAG: 29 nodes, one piece at a time, 2 GiB per pod.
Same graph as GitLab workflows/UC3.3_SOMES.

Space: SCDI
Inputs:  onedata:///SCDI/UC3.3_SOMES/inputs/
Outputs: onedata:///SCDI/UC3.3_SOMES/outputs/

Images:
- ghcr.io/scditech/uc3_3_somes:0.3.1-group0
- harbor.testbed.spice-platform.eu/partner/uc3/uc3_3_somes:0.3.1-group0

Domino Settings before Create:
- Storage Source = Local
- Name: letters/numbers/_ only
