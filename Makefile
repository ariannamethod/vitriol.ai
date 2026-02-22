# yent.yo — Makefile
#
# Auto-detects hardware and downloads the right weights.
#
# Usage:
#   make setup    — detect hardware, download optimal weights
#   make build    — build Go binary (auto-detects BLAS + ORT)
#   make run      — generate an image (full pipeline)
#   make info     — show detected hardware

HF_REPO = ataeff/yent.yo
GO_DIR = go
WEIGHTS_DIR = weights
ONNX_FP16_DIR = $(WEIGHTS_DIR)/onnx_fp16
ONNX_INT8_DIR = $(WEIGHTS_DIR)/onnx_int8
CLIP_TOK_DIR = $(WEIGHTS_DIR)/clip_tokenizer
YENT_GGUF_DIR = $(WEIGHTS_DIR)/micro-yent
NANO_GGUF_DIR = $(WEIGHTS_DIR)/nano-yent

# ---- Hardware Detection ----

HAS_NVIDIA := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
HAS_CUDA := $(if $(HAS_NVIDIA),1,)

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  RAM_MB := $(shell sysctl -n hw.memsize 2>/dev/null | awk '{print int($$1/1048576)}')
  HAS_ACCELERATE := 1
  HAS_OPENBLAS :=
else
  RAM_MB := $(shell grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($$2/1024)}')
  HAS_ACCELERATE :=
  HAS_OPENBLAS := $(shell ldconfig -p 2>/dev/null | grep -q libopenblas && echo 1)
endif

# Weight selection:
#   GPU (CUDA)     → fp16 + onnxruntime-gpu (fastest)
#   CPU, RAM ≥12GB → fp16 + onnxruntime (full quality)
#   CPU, RAM <12GB → int8 + onnxruntime (lighter)
ENOUGH_RAM := $(shell [ "$(RAM_MB)" -ge 12000 ] 2>/dev/null && echo 1)

ifeq ($(HAS_CUDA),1)
  ONNX_VARIANT = fp16
  ONNX_DIR = $(ONNX_FP16_DIR)
  ORT_PKG = onnxruntime-gpu
else ifeq ($(ENOUGH_RAM),1)
  ONNX_VARIANT = fp16
  ONNX_DIR = $(ONNX_FP16_DIR)
  ORT_PKG = onnxruntime
else
  ONNX_VARIANT = int8
  ONNX_DIR = $(ONNX_INT8_DIR)
  ORT_PKG = onnxruntime
endif

.PHONY: setup build run run-yent run-dual serve info clean setup-weights setup-tokenizer setup-yent-gguf setup-nano-gguf deps

# ---- Main Targets ----

setup: info deps setup-weights setup-tokenizer setup-yent-gguf setup-nano-gguf build
	@echo ""
	@echo "=== Ready! ==="
	@echo "  Run:  make run-dual INPUT=\"who are you\""
	@echo "  Or:   make serve"
	@echo "  Or:   make run PROMPT=\"a cat on a roof\""

info:
	@echo "=== Hardware Detection ==="
	@echo "  OS:       $(UNAME)"
	@echo "  RAM:      $(RAM_MB) MB"
	@echo "  GPU:      $(if $(HAS_NVIDIA),$(HAS_NVIDIA),none)"
	@echo "  CUDA:     $(if $(HAS_CUDA),yes,no)"
	@echo "  BLAS:     $(if $(HAS_ACCELERATE),Apple Accelerate,$(if $(HAS_OPENBLAS),OpenBLAS,none))"
	@echo "  Weights:  $(ONNX_VARIANT) ($(if $(HAS_CUDA),GPU — fast,$(if $(ENOUGH_RAM),CPU fp16 — full quality,CPU int8 — light)))"
	@echo "  ORT pkg:  $(ORT_PKG)"
	@echo ""

deps:
	@echo "=== Installing dependencies ==="
	pip3 install --quiet huggingface_hub $(ORT_PKG) Pillow numpy 2>/dev/null || \
		pip install --quiet huggingface_hub $(ORT_PKG) Pillow numpy
	@echo "  Dependencies OK"

setup-weights:
	@if [ -d "$(ONNX_DIR)" ] && [ "$$(ls $(ONNX_DIR)/*.onnx 2>/dev/null | wc -l)" -ge 3 ]; then \
		echo "  $(ONNX_VARIANT) ONNX weights already present."; \
	else \
		echo "Downloading $(ONNX_VARIANT) ONNX weights..."; \
		mkdir -p $(ONNX_DIR); \
		python3 -c "from huggingface_hub import hf_hub_download; \
			[hf_hub_download('$(HF_REPO)', f'weights/onnx_$(ONNX_VARIANT)/{m}', \
			local_dir='.hf_cache') for m in ['clip_text_encoder.onnx','unet.onnx','vae_decoder.onnx']]"; \
		cp .hf_cache/weights/onnx_$(ONNX_VARIANT)/*.onnx $(ONNX_DIR)/; \
		echo "  $(ONNX_VARIANT) ONNX: $$(du -sh $(ONNX_DIR) | cut -f1)"; \
	fi

setup-tokenizer:
	@if [ -f "$(CLIP_TOK_DIR)/vocab.json" ]; then \
		echo "  CLIP tokenizer already present."; \
	else \
		echo "Downloading CLIP tokenizer..."; \
		mkdir -p $(CLIP_TOK_DIR); \
		python3 -c "from huggingface_hub import snapshot_download; \
			snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
			allow_patterns='weights/clip_tokenizer/*')"; \
		cp .hf_cache/weights/clip_tokenizer/* $(CLIP_TOK_DIR)/; \
		echo "  Tokenizer OK"; \
	fi

setup-yent-gguf:
	@if [ -f "$(YENT_GGUF_DIR)/micro-yent-q8_0.gguf" ]; then \
		echo "  micro-Yent GGUF already present."; \
	else \
		echo "Downloading micro-Yent Q8_0 (71 MB)..."; \
		mkdir -p $(YENT_GGUF_DIR); \
		python3 -c "from huggingface_hub import hf_hub_download; \
			hf_hub_download('$(HF_REPO)', 'weights/micro-yent/micro-yent-q8_0.gguf', \
			local_dir='.hf_cache')"; \
		cp .hf_cache/weights/micro-yent/micro-yent-q8_0.gguf $(YENT_GGUF_DIR)/; \
		echo "  micro-Yent: $$(du -sh $(YENT_GGUF_DIR) | cut -f1)"; \
	fi

setup-nano-gguf:
	@if [ -f "$(NANO_GGUF_DIR)/nano-yent-f16.gguf" ]; then \
		echo "  nano-Yent GGUF already present."; \
	else \
		echo "Downloading nano-Yent F16 (88 MB)..."; \
		mkdir -p $(NANO_GGUF_DIR); \
		python3 -c "from huggingface_hub import hf_hub_download; \
			hf_hub_download('$(HF_REPO)', 'weights/nano-yent/nano-yent-f16.gguf', \
			local_dir='.hf_cache')"; \
		cp .hf_cache/weights/nano-yent/nano-yent-f16.gguf $(NANO_GGUF_DIR)/; \
		echo "  nano-Yent: $$(du -sh $(NANO_GGUF_DIR) | cut -f1)"; \
	fi

build:
	@echo "=== Building yentyo ==="
	cd $(GO_DIR) && go build -o ../yentyo .
	@echo "  Built: yentyo ($$(du -h yentyo | cut -f1))"

# ---- Run Targets ----

INPUT ?= who are you
SEED ?= 42
run-yent: $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf $(ONNX_DIR)/unet.onnx
	@PROMPT=$$(./yentyo --prompt-only $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf "$(INPUT)" 30 0.8 2>/dev/null | tail -1); \
	WORDS=$$(echo "$$PROMPT" | sed 's/, oil painting.*//' | sed 's/, Picasso .*//' | sed 's/, social realism.*//' | sed 's/, street art.*//' | sed 's/, caricature.*//' | sed 's/, propaganda poster.*//'); \
	echo "Input:  $(INPUT)"; \
	echo "Yent:   $$PROMPT"; \
	echo "Words:  $$WORDS"; \
	python3 ort_generate.py $(ONNX_DIR) $(CLIP_TOK_DIR) "$$PROMPT" output_raw.png $(SEED) 10 7.5 --raw; \
	python3 artifact_mask.py output_raw.png output.png --text "$$WORDS"; \
	echo ""; \
	echo "=== Done: output.png ==="

run-dual: $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf $(NANO_GGUF_DIR)/nano-yent-f16.gguf $(ONNX_DIR)/unet.onnx
	@echo "=== Dual Yent Mode ==="
	@echo "Input: $(INPUT)"
	@./yentyo $(ONNX_DIR) --dual \
		$(YENT_GGUF_DIR)/micro-yent-q8_0.gguf \
		$(NANO_GGUF_DIR)/nano-yent-f16.gguf \
		"$(INPUT)" output.png

PORT ?= 8080
serve: $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf $(NANO_GGUF_DIR)/nano-yent-f16.gguf
	@echo "=== yent.yo Server ==="
	@echo "Open http://localhost:$(PORT)"
	@./yentyo --serve $(ONNX_DIR) \
		$(YENT_GGUF_DIR)/micro-yent-q8_0.gguf \
		$(NANO_GGUF_DIR)/nano-yent-f16.gguf \
		$(PORT)

PROMPT ?= a surreal painting of chaos
run:
	python3 ort_generate.py $(ONNX_DIR) $(CLIP_TOK_DIR) \
		"$(PROMPT)" output.png $(SEED) 10 7.5

run-prompt:
	./yentyo --prompt-only $(YENT_GGUF_DIR)/micro-yent-q8_0.gguf "$(INPUT)" 30 0.8

clean:
	rm -f yentyo output.png output_raw.png
	rm -rf $(WEIGHTS_DIR) .hf_cache
