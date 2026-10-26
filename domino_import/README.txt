Import: uc33_somes.customization

Full SoMES DAG (30 nodes / 29 unique pieces), including:
- UserInputPiece (form flags: enable_shmu, ems_endpoint_url as XCom outputs)
- load forecast chain + PVOUT/Open-Meteo/SHMU chain
- ops: connectors, prices, flex, BatterySim x2, grid, EMS, dashboard

Space: SCDI
Inputs:  onedata:///SCDI/UC3.3_SOMES/inputs/
Outputs: onedata:///SCDI/UC3.3_SOMES/outputs/

Domino Settings before Create:
- Storage Source = Local
- Name: letters/numbers/_ only
- Pieces repository must include ALL pieces from this file (scditech/uc3_3_somes release)
- Image: ghcr.io/scditech/uc3_3_somes:<version>-group0

If Domino creates fewer than 30 nodes, missing pieces are not registered in that workspace.
