#!/bin/bash
# Script d'exécution du client CIANNA_RTS

cd "$(dirname "$0")"

python client/src/scripts/emulate_client.py
