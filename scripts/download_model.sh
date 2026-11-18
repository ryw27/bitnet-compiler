#!/usr/bin/env bash

set -euo pipefail

TARGET_DIR="./models/bitnet-b1.58-2B-4T"

URL="huggingface.co"
REPO_ID="microsoft/bitnet-b1.58-2B-4T"
REVISION="main"

FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "model.safetensors"
)

# Define ANSI color codes (auto-disabled if stdout is not a terminal)
if [[ -t 1 ]]; then
    BOLD="\033[1m"
    CYAN="\033[36m"
    BLUE="\033[34m"
    YELLOW="\033[33m"
    GREEN="\033[32m"
    DIM="\033[2m"
    RESET="\033[0m"
else
    BOLD="" CYAN="" BLUE="" YELLOW="" GREEN="" DIM="" RESET=""
fi

mkdir -p "$TARGET_DIR"
echo -e "${BOLD}${CYAN}==>${RESET} ${BOLD}Downloading ${REPO_ID}${RESET} ${DIM}(revision: ${REVISION})${RESET} into ${CYAN}${TARGET_DIR}${RESET}...\n"

for FILE in "${FILES[@]}"; do
    DEST="$TARGET_DIR/$FILE"
    CUR_URL="$URL/$REPO_ID/resolve/$REVISION/$FILE"

    if [[ -f "$DEST" ]]; then
        echo -e "  ${YELLOW}[SKIP]${RESET}     ${BOLD}$FILE${RESET} ${DIM}already exists${RESET}"
    else
        echo -e "  ${BLUE}[DOWNLOAD]${RESET} Fetching ${BOLD}$FILE${RESET} ${DIM}($CUR_URL)${RESET}"
        curl -fL -# -C - "$CUR_URL" -o "$DEST"
    fi
done

echo -e "\n${GREEN}${BOLD}✔ Download complete!${RESET} Files in ${CYAN}$TARGET_DIR${RESET}"
# ls -lh --color=auto "$TARGET_DIR"
