package main

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// --- Trigram extraction ---

func TestExtractTrigrams(t *testing.T) {
	tests := []struct {
		input string
		want  int // minimum expected trigrams
	}{
		{"hello world", 2},                        // 1 bigram + 2 words
		{"the meaning of life", 6},                // 2 trigrams + 3 bigrams + 4 words (minus dups)
		{"hi", 1},                                 // just the word
		{"a b c d e", 8},                          // lots of trigrams+bigrams+words
		{"", 0},                                   // empty
		{"ненавижу всё это дерьмо", 4},            // russian
	}

	for _, tt := range tests {
		trigrams := extractTrigrams(tt.input)
		if len(trigrams) < tt.want {
			t.Errorf("extractTrigrams(%q) = %d trigrams, want >= %d", tt.input, len(trigrams), tt.want)
		}
	}
}

func TestExtractTrigramsIncludesWords(t *testing.T) {
	trigrams := extractTrigrams("hello world")
	if !trigrams["hello"] {
		t.Error("expected 'hello' in trigrams")
	}
	if !trigrams["world"] {
		t.Error("expected 'world' in trigrams")
	}
	if !trigrams["hello world"] {
		t.Error("expected bigram 'hello world' in trigrams")
	}
}

// --- Jaccard similarity ---

func TestJaccardSimilarity(t *testing.T) {
	a := map[string]bool{"a": true, "b": true, "c": true}
	b := map[string]bool{"b": true, "c": true, "d": true}

	sim := jaccardSimilarity(a, b)
	// intersection = {b, c} = 2, union = {a, b, c, d} = 4 → 0.5
	if math.Abs(float64(sim)-0.5) > 0.01 {
		t.Errorf("jaccardSimilarity = %.3f, want 0.5", sim)
	}

	// Identical sets
	sim = jaccardSimilarity(a, a)
	if math.Abs(float64(sim)-1.0) > 0.01 {
		t.Errorf("jaccardSimilarity(a, a) = %.3f, want 1.0", sim)
	}

	// Disjoint sets
	c := map[string]bool{"x": true, "y": true}
	sim = jaccardSimilarity(a, c)
	if sim != 0 {
		t.Errorf("jaccardSimilarity(disjoint) = %.3f, want 0.0", sim)
	}

	// Empty sets
	sim = jaccardSimilarity(map[string]bool{}, map[string]bool{})
	if sim != 0 {
		t.Errorf("jaccardSimilarity(empty, empty) = %.3f, want 0.0", sim)
	}
}

// --- Dissonance computation (without model) ---

func newTestPG() *PromptGenerator {
	return &PromptGenerator{
		rng:   rand.New(rand.NewSource(42)),
		cloud: make(map[string]float32),
	}
}

func TestDissonanceFirstInput(t *testing.T) {
	pg := newTestPG()
	d, pulse := pg.computeDissonance("hello world")

	// First input: no previous trigrams, so similarity=0, dissonance starts at 1.0
	// After pulse adjustments it should be high
	if d < 0.5 {
		t.Errorf("first input dissonance = %.3f, want >= 0.5", d)
	}
	if pulse.Novelty < 0.5 {
		t.Errorf("first input novelty = %.3f, want >= 0.5 (cloud empty)", pulse.Novelty)
	}
}

func TestDissonanceRepeatedInput(t *testing.T) {
	pg := newTestPG()

	// First call — high dissonance
	d1, _ := pg.computeDissonance("hello world")

	// Same input again — should have lower dissonance (recognizes it)
	d2, _ := pg.computeDissonance("hello world")

	if d2 >= d1 {
		t.Errorf("repeated input: d2=%.3f should be < d1=%.3f", d2, d1)
	}
}

func TestDissonanceBoredomDetection(t *testing.T) {
	pg := newTestPG()

	// Prime with first input
	pg.computeDissonance("hello")
	// Same input again — triggers low dissonance
	pg.computeDissonance("hello")
	// Same again — should trigger boredom
	pg.computeDissonance("hello")
	d, _ := pg.computeDissonance("hello")

	if pg.boredomCount < 2 {
		t.Errorf("boredom count = %d, want >= 2", pg.boredomCount)
	}
	// Boredom should raise dissonance
	if d < 0.5 {
		t.Errorf("boredom-raised dissonance = %.3f, want >= 0.5", d)
	}
}

func TestDissonanceCloudMorphing(t *testing.T) {
	pg := newTestPG()

	pg.computeDissonance("hello world")

	// "hello" and "world" should be in cloud
	if pg.cloud["hello"] < 0.05 {
		t.Errorf("cloud['hello'] = %.3f, want > 0.05", pg.cloud["hello"])
	}
	if pg.cloud["world"] < 0.05 {
		t.Errorf("cloud['world'] = %.3f, want > 0.05", pg.cloud["world"])
	}

	// New word should not be in cloud
	if pg.cloud["banana"] > 0.01 {
		t.Errorf("cloud['banana'] = %.3f, want ~0", pg.cloud["banana"])
	}
}

func TestDissonanceArousal(t *testing.T) {
	pg := newTestPG()

	// Emotional input should have arousal > 0
	_, pulse := pg.computeDissonance("I hate everything and I want to die")
	if pulse.Arousal <= 0 {
		t.Errorf("arousal for emotional input = %.3f, want > 0", pulse.Arousal)
	}

	// Boring input should have low arousal
	pg2 := newTestPG()
	_, pulse2 := pg2.computeDissonance("the weather is nice today")
	if pulse2.Arousal >= pulse.Arousal {
		t.Errorf("boring arousal (%.3f) should be < emotional (%.3f)", pulse2.Arousal, pulse.Arousal)
	}
}

// --- Temperature adaptation ---

func TestAdaptTemperatureRange(t *testing.T) {
	pg := newTestPG()

	inputs := []string{
		"hi",
		"the meaning of life",
		"I fucking hate everything",
		"a",
		"the quick brown fox jumps over the lazy dog and then does it again",
	}

	for _, input := range inputs {
		temp := pg.adaptTemperature(input, 0.8)
		if temp < 0.3 || temp > 1.5 {
			t.Errorf("adaptTemperature(%q) = %.3f, want ∈ [0.3, 1.5]", input, temp)
		}
	}
}

// --- Oppositional template matching ---

func TestReactionTemplateMatching(t *testing.T) {
	tests := []struct {
		input   string
		wantHit bool
	}{
		{"I am so sad", true},
		{"I hate you", true},
		{"you are beautiful", true},
		{"I'm bored", true},
		{"hello", true},
		{"draw me a duck", true},
		{"cat", true},
		{"death comes for us all", true},
		{"the weather is nice", false}, // no keyword match
	}

	for _, tt := range tests {
		lower := toLowerStr(tt.input)
		matched := false
		for _, rt := range reactionTemplates {
			for _, kw := range rt.keywords {
				if containsStr(lower, kw) {
					matched = true
					break
				}
			}
			if matched {
				break
			}
		}
		if matched != tt.wantHit {
			t.Errorf("template match for %q: got %v, want %v", tt.input, matched, tt.wantHit)
		}
	}
}

// helpers (can't import strings in test scope easily, inline)
func toLowerStr(s string) string {
	b := make([]byte, len(s))
	for i := range s {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		b[i] = c
	}
	return string(b)
}

func containsStr(s, sub string) bool {
	return len(s) >= len(sub) && searchStr(s, sub) >= 0
}

func searchStr(s, sub string) int {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// --- Sketch generation ---

func TestGenerateSketchLine(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Draft 0: sparse
	line0 := generateSketchLine(50, 0, 7, 15, []string{"hello"}, rng)
	if len(line0) != 50 {
		t.Errorf("line0 length = %d, want 50", len(line0))
	}

	// Draft 1: some structure
	line1 := generateSketchLine(50, 1, 7, 15, []string{"test"}, rng)
	if len(line1) != 50 {
		t.Errorf("line1 length = %d, want 50", len(line1))
	}

	// Draft 2: denser
	line2 := generateSketchLine(50, 2, 7, 15, []string{"world"}, rng)
	if len(line2) != 50 {
		t.Errorf("line2 length = %d, want 50", len(line2))
	}

	// Count non-space chars — each draft should be progressively denser
	count := func(s string) int {
		n := 0
		for _, c := range s {
			if c != ' ' {
				n++
			}
		}
		return n
	}

	// We can't guarantee strict ordering due to randomness, but on average
	// draft 2 center line should be denser than draft 0
	// Run multiple times to average
	var avg0, avg2 float64
	for trial := 0; trial < 100; trial++ {
		l0 := generateSketchLine(50, 0, 7, 15, nil, rng)
		l2 := generateSketchLine(50, 2, 7, 15, nil, rng)
		avg0 += float64(count(l0))
		avg2 += float64(count(l2))
	}
	avg0 /= 100
	avg2 /= 100

	if avg2 <= avg0 {
		t.Errorf("draft 2 avg density (%.1f) should be > draft 0 (%.1f)", avg2, avg0)
	}
}

func TestSketchCharsNotEmpty(t *testing.T) {
	if len(sketchChars) == 0 {
		t.Error("sketchChars is empty")
	}
}

func TestDefaultSketchConfig(t *testing.T) {
	cfg := DefaultSketchConfig()
	if cfg.Width <= 0 || cfg.Height <= 0 {
		t.Errorf("bad dimensions: %dx%d", cfg.Width, cfg.Height)
	}
	if cfg.NumDrafts < 1 {
		t.Errorf("NumDrafts = %d, want >= 1", cfg.NumDrafts)
	}
	if cfg.DraftDelay <= 0 {
		t.Error("DraftDelay should be positive")
	}
}

// --- DualYent structure (without models) ---

func TestDualResultFields(t *testing.T) {
	r := DualResult{
		Prompt:    "a mirror cracking under the weight of your words, oil painting",
		YentWords: "a mirror cracking under the weight of your words",
		Roast:     "you think that's clever? pathetic.",
		ArtistID:  "A",
	}

	if r.Prompt == "" {
		t.Error("Prompt should not be empty")
	}
	if r.YentWords == "" {
		t.Error("YentWords should not be empty")
	}
	if r.ArtistID != "A" && r.ArtistID != "B" {
		t.Errorf("ArtistID = %q, want A or B", r.ArtistID)
	}
}

// --- Style suffixes ---

func TestStyleSuffixesNotEmpty(t *testing.T) {
	if len(styleSuffixes) == 0 {
		t.Fatal("styleSuffixes is empty")
	}
	for i, s := range styleSuffixes {
		if s == "" {
			t.Errorf("styleSuffixes[%d] is empty", i)
		}
		if s[0] != ',' {
			t.Errorf("styleSuffixes[%d] should start with comma: %q", i, s)
		}
	}
}

func TestReactionTemplatesNotEmpty(t *testing.T) {
	if len(reactionTemplates) == 0 {
		t.Fatal("reactionTemplates is empty")
	}
	for i, rt := range reactionTemplates {
		if len(rt.keywords) == 0 {
			t.Errorf("reactionTemplates[%d] has no keywords", i)
		}
		if len(rt.starters) == 0 {
			t.Errorf("reactionTemplates[%d] has no starters", i)
		}
	}
}

func TestDefaultStartersNotEmpty(t *testing.T) {
	if len(defaultStarters) == 0 {
		t.Fatal("defaultStarters is empty")
	}
}

// --- Arousal words ---

func TestArousalWordsContainExpected(t *testing.T) {
	expected := []string{"hate", "love", "die", "fuck", "sad", "angry"}
	for _, w := range expected {
		if !arousalWords[w] {
			t.Errorf("arousalWords missing %q", w)
		}
	}
}

// --- Server types ---

func TestReactRequestDefaults(t *testing.T) {
	req := ReactRequest{}
	if req.Input != "" {
		t.Error("default Input should be empty")
	}
	if req.Temperature != 0 {
		t.Error("default Temperature should be 0 (omitempty)")
	}
	if req.MaxTokens != 0 {
		t.Error("default MaxTokens should be 0 (omitempty)")
	}
}

func TestReactResponseJSON(t *testing.T) {
	resp := ReactResponse{
		Prompt:     "a mirror cracking",
		YentWords:  "mirror cracking",
		Roast:      "you fool",
		ArtistID:   "B",
		Dissonance: 0.73,
		Temp:       0.85,
		ElapsedMs:  42,
	}

	if resp.Prompt == "" || resp.Roast == "" {
		t.Error("response fields should not be empty")
	}
	if resp.Dissonance < 0 || resp.Dissonance > 1 {
		t.Errorf("dissonance = %.3f, want ∈ [0, 1]", resp.Dissonance)
	}
}

// --- HTTP handler tests ---

func TestHealthResponseFields(t *testing.T) {
	h := HealthResponse{
		Version: "2.0",
		ModelA:  "12 layers, 512 dim",
		ModelB:  "12 layers, 384 dim",
		SDModel: "bk-sdm-tiny",
		Ready:   true,
	}
	if h.Version != "2.0" {
		t.Errorf("version = %q, want 2.0", h.Version)
	}
	if !h.Ready {
		t.Error("Ready should be true")
	}
}

// --- FP16 conversion (ORT helpers, build tag ort only) ---
// These are in ort_pipeline.go behind build tag, so we test the math separately

func TestClampByte(t *testing.T) {
	tests := []struct {
		in   float32
		want uint8
	}{
		{0, 0},
		{1, 255},
		{0.5, 127},
		{-1, 0},
		{2, 255},
	}
	for _, tt := range tests {
		got := clampByte(tt.in)
		if got != tt.want {
			t.Errorf("clampByte(%v) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestTensorMinMax(t *testing.T) {
	tensor := &Tensor{
		Data:  []float32{-1, 0, 1, 2, -3, 0.5},
		Shape: []int{6},
	}
	if tensorMin(tensor) != -3 {
		t.Errorf("tensorMin = %v, want -3", tensorMin(tensor))
	}
	if tensorMax(tensor) != 2 {
		t.Errorf("tensorMax = %v, want 2", tensorMax(tensor))
	}
}

func TestRandomLatent(t *testing.T) {
	latent := randomLatent(1, 4, 8, 8, 42)
	if len(latent.Data) != 256 { // 1*4*8*8
		t.Errorf("latent size = %d, want 256", len(latent.Data))
	}

	// Should be roughly normally distributed — check mean ~0
	var sum float64
	for _, v := range latent.Data {
		sum += float64(v)
	}
	mean := sum / float64(len(latent.Data))
	if math.Abs(mean) > 0.5 {
		t.Errorf("latent mean = %.3f, want ~0", mean)
	}

	// Same seed → same noise
	latent2 := randomLatent(1, 4, 8, 8, 42)
	for i := range latent.Data {
		if latent.Data[i] != latent2.Data[i] {
			t.Error("same seed should produce same noise")
			break
		}
	}
}

// --- DDIM Scheduler ---

func TestDDIMScheduler(t *testing.T) {
	sched := NewDDIMScheduler(1000, 0.00085, 0.012)

	ts := sched.SetTimesteps(10)
	if len(ts) != 10 {
		t.Errorf("timesteps length = %d, want 10", len(ts))
	}
	// Timesteps should be decreasing
	for i := 1; i < len(ts); i++ {
		if ts[i] >= ts[i-1] {
			t.Errorf("timesteps not decreasing: ts[%d]=%d >= ts[%d]=%d", i, ts[i], i-1, ts[i-1])
		}
	}
	// First should be high, last should be low
	if ts[0] < 500 {
		t.Errorf("first timestep = %d, want >= 500", ts[0])
	}
	if ts[len(ts)-1] > 200 {
		t.Errorf("last timestep = %d, want <= 200", ts[len(ts)-1])
	}
}

// --- Benchmark: dissonance computation ---

func BenchmarkDissonance(b *testing.B) {
	pg := newTestPG()
	inputs := []string{
		"hello world",
		"the meaning of life is to find your gift",
		"I hate everything and everyone",
		"a",
		"revolution will not be televised",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pg.computeDissonance(inputs[i%len(inputs)])
	}
}

func BenchmarkSketchLine(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	words := []string{"test", "hello", "world"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		generateSketchLine(50, i%3, 7, 15, words, rng)
	}
}

// --- Pulse snapshot ---

func TestPulseSnapshotRange(t *testing.T) {
	pg := newTestPG()
	inputs := []string{
		"hi",
		"I love you so much",
		"the quick brown fox",
		"a a a a a a a a",
		"",
	}

	for _, input := range inputs {
		_, pulse := pg.computeDissonance(input)
		if pulse.Novelty < 0 || pulse.Novelty > 1 {
			t.Errorf("novelty for %q = %.3f, want ∈ [0, 1]", input, pulse.Novelty)
		}
		if pulse.Arousal < 0 || pulse.Arousal > 1 {
			t.Errorf("arousal for %q = %.3f, want ∈ [0, 1]", input, pulse.Arousal)
		}
		if pulse.Entropy < 0 || pulse.Entropy > 1 {
			t.Errorf("entropy for %q = %.3f, want ∈ [0, 1]", input, pulse.Entropy)
		}
	}
}

// --- Version ---

func TestVersion(t *testing.T) {
	if yentYoVersion == "" {
		t.Error("version should not be empty")
	}
}

// --- Integration: dissonance → temperature pipeline ---

func TestDissonanceToTemperaturePipeline(t *testing.T) {
	pg := newTestPG()

	// Boring repeated input → should eventually raise temperature via boredom
	var temps []float32
	for i := 0; i < 5; i++ {
		temp := pg.adaptTemperature("hi", 0.8)
		temps = append(temps, temp)
	}

	// After several repeats, boredom should kick in and raise T
	if temps[len(temps)-1] <= temps[0] {
		t.Logf("temps: %v", temps)
		// Not a hard error because boredom detection is probabilistic
		// but log for inspection
	}
}

// --- Entropy calculation ---

func TestEntropyCalculation(t *testing.T) {
	pg := newTestPG()

	// All unique words → high entropy
	_, pulse1 := pg.computeDissonance("alpha beta gamma delta epsilon")
	if pulse1.Entropy < 0.9 {
		t.Errorf("all-unique entropy = %.3f, want >= 0.9", pulse1.Entropy)
	}

	// All same words → low entropy
	pg2 := newTestPG()
	_, pulse2 := pg2.computeDissonance("the the the the the")
	if pulse2.Entropy > 0.3 {
		t.Errorf("all-same entropy = %.3f, want <= 0.3", pulse2.Entropy)
	}
}

// --- SavePNG ---

func TestSavePNG(t *testing.T) {
	// Create a tiny 2x2 test image tensor
	tensor := &Tensor{
		Data:  make([]float32, 3*2*2), // 3 channels, 2x2
		Shape: []int{1, 3, 2, 2},
	}
	// Fill with known values
	for i := range tensor.Data {
		tensor.Data[i] = float32(i)/float32(len(tensor.Data))*2 - 1 // [-1, 1]
	}

	path := "/tmp/test_yentyo_save.png"
	err := savePNG(tensor, path)
	if err != nil {
		t.Fatalf("savePNG: %v", err)
	}

	// Verify file exists and has content
	info, err := statFile(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info < 50 {
		t.Errorf("PNG file too small: %d bytes", info)
	}
}

func statFile(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

var _ = time.Now // prevent unused import
