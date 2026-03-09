#!/bin/bash
#SBATCH --job-name=osnn_ex1
#SBATCH --partition=gpu_a100_il
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=94000mb
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

# One task = one config (Width, NumLayer).
# Widths: 5..100 step 5 => 20 widths.
# Layers: 3 and 4 => 40 tasks total.
#SBATCH --array=0-39

# Single per-task log (stdout+stderr merged)
#SBATCH --output=slurm/%x_%A_%a.log

# NOTE: Adjust this path on the cluster if needed.
source /pfs/data6/home/ul/ul_student/ul_qbv80/venv_gpu/bin/activate

BASE_DIR="${SLURM_SUBMIT_DIR}"
cd "${BASE_DIR}"

mkdir -p slurm

# --- Startinformationen ---
echo "========================================================="
echo " Job gestartet auf $(date)"
echo " Host: $(hostname)"
echo " Submit dir: ${BASE_DIR}"
echo " Array: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================="

# Map array index -> (width, layer)
# 0..19  => L=3, W=5..100
# 20..39 => L=4, W=5..100
IDX=${SLURM_ARRAY_TASK_ID}
WIDTH_IDX=$((IDX % 20))
WIDTH=$((5 * (WIDTH_IDX + 1)))
if [ "$IDX" -lt 20 ]; then
	NUMLAYER=3
else
	NUMLAYER=4
fi
EPOACH=90000

# Use one shared run name for all array tasks.
# If RUN_ID is exported when submitting, all tasks will write into the same code/Run/<RUN_ID>/.
# If not exported, we still enforce ONE shared RUN_ID per array job by storing it in a file.
if [ -z "${RUN_ID}" ]; then
	RUN_ID_FILE="${BASE_DIR}/slurm/run_id_${SLURM_ARRAY_JOB_ID}.txt"
	LOCKDIR="${RUN_ID_FILE}.lock"
	# Simple inter-process lock (mkdir is atomic on POSIX filesystems)
	while ! mkdir "${LOCKDIR}" 2>/dev/null; do
		sleep 0.2
	done
	if [ -f "${RUN_ID_FILE}" ]; then
		RUN_ID="$(cat "${RUN_ID_FILE}")"
	else
		RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_ARRAY_JOB_ID}"
		echo "${RUN_ID}" > "${RUN_ID_FILE}"
	fi
	rmdir "${LOCKDIR}" 2>/dev/null || true
fi

echo "Config: W=${WIDTH}, L=${NUMLAYER}, RunId=${RUN_ID}"

# --- Python-Programm starten ---
# ex1.py writes to: 2.1/code/Run/<RunId>/{figs,data/W{W}_L{L}/...}
python "${BASE_DIR}/code/examples/ex1.py" \
	--Width "${WIDTH}" \
	--NumLayer "${NUMLAYER}" \
	--RunId "${RUN_ID}" \
	--SaveEvery 0 \
	--Epoach "${EPOACH}"

echo "========================================================="
echo " Job $SLURM_JOB_ID beendet auf $(date)"
echo "========================================================="
# --- Optionale Bereinigung des Scratch-Verzeichnisses ---
#rm -rf "$SCRATCH"