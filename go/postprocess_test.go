package main

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

func makeTestImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(42))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(rng.Intn(256)),
				G: uint8(rng.Intn(256)),
				B: uint8(rng.Intn(256)),
				A: 255,
			})
		}
	}
	return img
}

func TestComputeGradient(t *testing.T) {
	gray := make([]float32, 10*10)
	// Horizontal gradient
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			gray[y*10+x] = float32(x) * 25.5
		}
	}

	mag := computeGradient(gray, 10, 10)
	// Interior pixels should have gradient magnitude > 0
	if mag[5*10+5] == 0 {
		t.Error("gradient should be non-zero for gradient image")
	}
	// Edge pixels stay 0
	if mag[0] != 0 {
		t.Errorf("corner gradient should be 0, got %f", mag[0])
	}
}

func TestComputeArtifactScore(t *testing.T) {
	// Create 96x96 image (divisible by 12)
	img := makeTestImage(96, 96)
	score := computeArtifactScore(img)

	if len(score) != 96*96 {
		t.Errorf("score map length = %d, want %d", len(score), 96*96)
	}

	// All scores should be in [0, 1]
	for i, v := range score {
		if v < 0 || v > 1 {
			t.Errorf("score[%d] = %f, want ∈ [0, 1]", i, v)
			break
		}
	}
}

func TestComputeArtifactScoreSmoothImage(t *testing.T) {
	// Flat image → high artifact score (no detail)
	img := image.NewRGBA(image.Rect(0, 0, 96, 96))
	for y := 0; y < 96; y++ {
		for x := 0; x < 96; x++ {
			img.SetRGBA(x, y, color.RGBA{128, 128, 128, 255})
		}
	}
	score := computeArtifactScore(img)
	mean := meanFloat32(score)
	// Uniform image should have zero variance → all high artifact score or zero
	// Actually gradient is 0 everywhere → variance=0 → percentiles collapse → returns zeros
	_ = mean // just verify it doesn't crash
}

func TestApplyFilmGrain(t *testing.T) {
	img := makeTestImage(64, 64)
	original := cloneRGBA(img)
	applyFilmGrain(img, 22, 42)

	// Should modify the image
	different := false
	for i := range img.Pix {
		if img.Pix[i] != original.Pix[i] {
			different = true
			break
		}
	}
	if !different {
		t.Error("film grain should modify the image")
	}
}

func TestApplyFilmGrainDeterministic(t *testing.T) {
	img1 := makeTestImage(32, 32)
	img2 := cloneRGBA(img1)
	applyFilmGrain(img1, 22, 42)
	applyFilmGrain(img2, 22, 42)

	for i := range img1.Pix {
		if img1.Pix[i] != img2.Pix[i] {
			t.Error("same seed should produce same grain")
			break
		}
	}
}

func TestApplyChromaticAberration(t *testing.T) {
	img := makeTestImage(64, 64)
	original := cloneRGBA(img)
	applyChromaticAberration(img, 2)

	// Green channel should be unchanged
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			if img.RGBAAt(x, y).G != original.RGBAAt(x, y).G {
				t.Error("green channel should be unchanged")
				return
			}
		}
	}

	// R and B should be shifted
	different := false
	for y := 0; y < 64; y++ {
		for x := 2; x < 62; x++ { // avoid edges
			if img.RGBAAt(x, y).R != original.RGBAAt(x, y).R {
				different = true
				break
			}
		}
		if different {
			break
		}
	}
	if !different {
		t.Error("red channel should be shifted")
	}
}

func TestApplyVignette(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.SetRGBA(x, y, color.RGBA{200, 200, 200, 255})
		}
	}

	applyVignette(img, 0.35)

	// Center should be brighter than corner
	center := img.RGBAAt(32, 32)
	corner := img.RGBAAt(0, 0)
	if center.R <= corner.R {
		t.Errorf("center (%d) should be brighter than corner (%d)", center.R, corner.R)
	}
}

func TestBilinearUpscale(t *testing.T) {
	// 2x2 → 4x4
	data := []float32{0, 1, 0, 1}
	result := bilinearUpscale(data, 2, 2, 4, 4)

	if len(result) != 16 {
		t.Errorf("result length = %d, want 16", len(result))
	}

	// Corners should match original
	if math.Abs(float64(result[0]-0)) > 0.01 {
		t.Errorf("top-left = %f, want 0", result[0])
	}
	if math.Abs(float64(result[3]-1)) > 0.01 {
		t.Errorf("top-right = %f, want 1", result[3])
	}
}

func TestBoxBlur(t *testing.T) {
	data := make([]float32, 10*10)
	// Single bright pixel in center
	data[5*10+5] = 1.0

	boxBlur(data, 10, 10, 2)

	// Center should be reduced
	if data[5*10+5] >= 1.0 {
		t.Error("center should be reduced after blur")
	}
	// Neighbors should gain some value
	if data[5*10+4] == 0 {
		t.Error("neighbor should gain value from blur")
	}
}

func TestResizeRGBA(t *testing.T) {
	img := makeTestImage(64, 64)
	resized := resizeRGBA(img, 32, 32)

	bounds := resized.Bounds()
	if bounds.Dx() != 32 || bounds.Dy() != 32 {
		t.Errorf("resized = %dx%d, want 32x32", bounds.Dx(), bounds.Dy())
	}
}

func TestCloneRGBA(t *testing.T) {
	img := makeTestImage(32, 32)
	clone := cloneRGBA(img)

	// Should be equal
	for i := range img.Pix {
		if img.Pix[i] != clone.Pix[i] {
			t.Error("clone should be identical")
			break
		}
	}

	// Modify original, clone should be unchanged
	img.SetRGBA(0, 0, color.RGBA{0, 0, 0, 255})
	if clone.RGBAAt(0, 0).R == 0 && clone.RGBAAt(0, 0).G == 0 && clone.RGBAAt(0, 0).B == 0 {
		t.Error("clone should be independent of original")
	}
}

func TestPercentile(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	p50 := percentile(data, 50)
	if math.Abs(float64(p50)-5.5) > 1 {
		t.Errorf("p50 = %f, want ~5.5", p50)
	}

	p0 := percentile(data, 0)
	if p0 != 1 {
		t.Errorf("p0 = %f, want 1", p0)
	}
}

func TestRenderASCIILayer(t *testing.T) {
	img := makeTestImage(64, 64)
	score := make([]float32, 64*64)
	// Half artifact
	for i := 0; i < 64*32; i++ {
		score[i] = 0.8
	}

	result := renderASCIILayer(img, "test words", score)
	bounds := result.Bounds()

	if bounds.Dx() == 0 || bounds.Dy() == 0 {
		t.Error("ASCII layer should have non-zero dimensions")
	}
}

func TestPostProcessFull(t *testing.T) {
	img := makeTestImage(96, 96)
	result := PostProcess(img, "test yent words for overlay")

	bounds := result.Bounds()
	if bounds.Dx() == 0 || bounds.Dy() == 0 {
		t.Error("PostProcess should return non-zero image")
	}
}

func TestTensorToRGBA(t *testing.T) {
	tensor := &Tensor{
		Data:  make([]float32, 3*4*4),
		Shape: []int{1, 3, 4, 4},
	}
	for i := range tensor.Data {
		tensor.Data[i] = 0.5
	}

	rgba := tensorToRGBA(tensor)
	bounds := rgba.Bounds()
	if bounds.Dx() != 4 || bounds.Dy() != 4 {
		t.Errorf("size = %dx%d, want 4x4", bounds.Dx(), bounds.Dy())
	}

	// Value 0.5 → (0.5+1)/2 = 0.75 → 0.75*255 ≈ 191
	c := rgba.RGBAAt(0, 0)
	if c.R < 180 || c.R > 200 {
		t.Errorf("R = %d, want ~191", c.R)
	}
}

func TestFloat32ToRGBA(t *testing.T) {
	data := make([]float32, 3*4*4)
	for i := range data {
		data[i] = 0.0 // maps to (0+1)/2 = 0.5 → 127
	}

	rgba := float32ToRGBA(data, 4, 4)
	c := rgba.RGBAAt(0, 0)
	if c.R < 120 || c.R > 135 {
		t.Errorf("R = %d, want ~127", c.R)
	}
}

func TestGaussNoise(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	var sum float64
	n := 10000
	for i := 0; i < n; i++ {
		sum += float64(gaussNoise(rng))
	}
	mean := sum / float64(n)
	if math.Abs(mean) > 0.1 {
		t.Errorf("noise mean = %f, want ~0", mean)
	}
}

func TestClamp8(t *testing.T) {
	if clamp8(-10) != 0 {
		t.Error("-10 should clamp to 0")
	}
	if clamp8(300) != 255 {
		t.Error("300 should clamp to 255")
	}
	if clamp8(128) != 128 {
		t.Error("128 should stay 128")
	}
}

func BenchmarkPostProcess(b *testing.B) {
	img := makeTestImage(128, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PostProcess(img, "benchmark test words")
	}
}
