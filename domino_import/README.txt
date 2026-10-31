Import: uc33_somes.customization
SPICE import: uc33_somes.spice.customization

Space: SCDI
Inputs:  onedata:///SCDI/UC3.3_SOMES/inputs/
Outputs: onedata:///SCDI/UC3.3_SOMES/outputs/

DAG: 29 nodes including ShmuCalibrationPiece, AnomalyAlertPiece, DashboardPiece.
Image: ghcr.io/scditech/uc3_3_somes:0.3.0-group0

Domino Settings before Create:
- Storage Source = Local
- Name: letters/numbers/_ only
