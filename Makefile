# yent.yo — Makefile
#
# Downloads weights from HuggingFace and builds the Go binary.
# Usage:
#   make setup    — download all weights
#   make build    — build Go binary
#   make run      — generate an image

HF_REPO = ataeff/yent.yo
ONNX_DIR = onnx_fp16
ONNX_INT8_DIR = onnx_int8
CLIP_TOK_DIR = clip_tokenizer
GO_DIR = go
YENT_WEIGHTS = $(GO_DIR)/weights

.PHONY: setup setup-fp16 setup-int8 setup-yent setup-tokenizer build run clean

# Download everything needed to run
setup: setup-fp16 setup-tokenizer setup-yent
	@echo "Done! Run 'make run' to generate an image."

# Download fp16 ONNX models (948 MB — GPU recommended)
setup-fp16:
	@echo "Downloading fp16 ONNX models..."
	@mkdir -p $(ONNX_DIR)
	python3 -c "from huggingface_hub import snapshot_download; \
		snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
		allow_patterns='weights/onnx_fp16/*')"
	cp .hf_cache/weights/onnx_fp16/* $(ONNX_DIR)/
	@echo "fp16 ONNX: $$(du -sh $(ONNX_DIR) | cut -f1)"

# Download int8 ONNX models (476 MB — CPU friendly, impressionist style)
setup-int8:
	@echo "Downloading int8 ONNX models..."
	@mkdir -p $(ONNX_INT8_DIR)
	python3 -c "from huggingface_hub import snapshot_download; \
		snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
		allow_patterns='weights/onnx_int8/*')"
	cp .hf_cache/weights/onnx_int8/* $(ONNX_INT8_DIR)/
	@echo "int8 ONNX: $$(du -sh $(ONNX_INT8_DIR) | cut -f1)"

# Download CLIP tokenizer
setup-tokenizer:
	@echo "Downloading CLIP tokenizer..."
	@mkdir -p $(CLIP_TOK_DIR)
	python3 -c "from huggingface_hub import snapshot_download; \
		snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
		allow_patterns='weights/clip_tokenizer/*')"
	cp .hf_cache/weights/clip_tokenizer/* $(CLIP_TOK_DIR)/

# Download micro-Yent Q8_0 GGUF (71 MB) + tokenizer
setup-yent:
	@if [ -f $(YENT_WEIGHTS)/micro-yent-q8_0.gguf ]; then \
		echo "micro-Yent weights already present."; \
	else \
		echo "Downloading micro-Yent Q8_0..."; \
		mkdir -p $(YENT_WEIGHTS); \
		python3 -c "from huggingface_hub import snapshot_download; \
			snapshot_download('$(HF_REPO)', local_dir='.hf_cache', \
			allow_patterns='weights/micro-yent/*')"; \
		cp .hf_cache/weights/micro-yent/* $(YENT_WEIGHTS)/; \
		echo "micro-Yent: $$(du -sh $(YENT_WEIGHTS) | cut -f1)"; \
	fi

# Build Go binary
build:
	cd $(GO_DIR) && go build -o ../yentyo .

# Quick test — prompt only (no ONNX needed)
run-prompt:
	cd $(GO_DIR) && go run . --prompt-only "the meaning of life"

# Full pipeline (needs ONNX weights + onnxruntime)
run:
	python3 ort_generate.py $(ONNX_DIR) $(CLIP_TOK_DIR) \
		"a surreal painting of chaos" output.png 42 25 7.5

# CPU mode (int8, slower but no GPU needed)
run-cpu: setup-int8
	python3 ort_generate.py $(ONNX_INT8_DIR) $(CLIP_TOK_DIR) \
		"a surreal painting of chaos" output.png 42 25 7.5

clean:
	rm -f yentyo output.png
	rm -rf $(ONNX_DIR) $(ONNX_INT8_DIR) $(CLIP_TOK_DIR) .hf_cache
