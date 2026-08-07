Import: uc33_somes.customization (OneData)

Space: SCDI
Inputs:  onedata:///SCDI/UC3.3_SOMES/inputs/
Outputs: onedata:///SCDI/UC3.3_SOMES/outputs/

Domino Settings before Create:
- Storage Source = Local
- Name: letters/numbers/_ only

Seed: python scripts/seed_somes_onedata.py
Image: ghcr.io/scditech/uc3_3_somes:0.2.1-group0
