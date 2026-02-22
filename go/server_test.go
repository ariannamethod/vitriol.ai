package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestServer() *Server {
	return &Server{
		images: make(map[string][]byte),
	}
}

func TestHandleUI(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	srv.handleUI(w, req)

	resp := w.Result()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/html") {
		t.Errorf("content-type = %q, want text/html", ct)
	}
	body := w.Body.String()
	if !strings.Contains(body, "yent.yo") {
		t.Error("UI should contain 'yent.yo'")
	}
}

func TestHandleUINotFound(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("GET", "/nonexistent", nil)
	w := httptest.NewRecorder()
	srv.handleUI(w, req)

	if w.Code != 404 {
		t.Errorf("status = %d, want 404 for non-root path", w.Code)
	}
}

func TestHandleReactMethodNotAllowed(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("GET", "/react", nil)
	w := httptest.NewRecorder()
	srv.handleReact(w, req)

	if w.Code != 405 {
		t.Errorf("status = %d, want 405 for GET on /react", w.Code)
	}
}

func TestHandleReactBadJSON(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("POST", "/react", strings.NewReader("{broken"))
	w := httptest.NewRecorder()
	srv.handleReact(w, req)

	if w.Code != 400 {
		t.Errorf("status = %d, want 400 for bad JSON", w.Code)
	}
}

func TestHandleReactEmptyInput(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("POST", "/react", strings.NewReader(`{"input":""}`))
	w := httptest.NewRecorder()
	srv.handleReact(w, req)

	if w.Code != 400 {
		t.Errorf("status = %d, want 400 for empty input", w.Code)
	}
}

func TestHandleImage(t *testing.T) {
	srv := newTestServer()

	// Store a test image
	srv.images["test123"] = []byte{0x89, 0x50, 0x4E, 0x47} // PNG magic bytes

	req := httptest.NewRequest("GET", "/image/test123", nil)
	w := httptest.NewRecorder()
	srv.handleImage(w, req)

	if w.Code != 200 {
		t.Errorf("status = %d, want 200", w.Code)
	}
	ct := w.Result().Header.Get("Content-Type")
	if ct != "image/png" {
		t.Errorf("content-type = %q, want image/png", ct)
	}
	if w.Body.Len() != 4 {
		t.Errorf("body length = %d, want 4", w.Body.Len())
	}
}

func TestHandleImageNotFound(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("GET", "/image/nonexistent", nil)
	w := httptest.NewRecorder()
	srv.handleImage(w, req)

	if w.Code != 404 {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestHandleImageCacheHeader(t *testing.T) {
	srv := newTestServer()
	srv.images["cached"] = []byte{0xFF}

	req := httptest.NewRequest("GET", "/image/cached", nil)
	w := httptest.NewRecorder()
	srv.handleImage(w, req)

	cc := w.Result().Header.Get("Cache-Control")
	if !strings.Contains(cc, "max-age") {
		t.Errorf("Cache-Control = %q, want max-age", cc)
	}
}

func TestHealthResponseJSON(t *testing.T) {
	h := HealthResponse{
		Version: "2.0",
		ModelA:  "12 layers, 512 dim",
		ModelB:  "12 layers, 384 dim",
		SDModel: "dummy",
		Ready:   true,
	}

	data, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}

	var decoded HealthResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Version != "2.0" {
		t.Errorf("version = %q, want 2.0", decoded.Version)
	}
	if !decoded.Ready {
		t.Error("ready should be true")
	}
}

func TestReactResponseSerialization(t *testing.T) {
	resp := ReactResponse{
		Prompt:     "test prompt",
		YentWords:  "test words",
		Roast:      "you suck",
		ArtistID:   "A",
		ImageURL:   "/image/123",
		Dissonance: 0.5,
		Temp:       0.8,
		ElapsedMs:  42,
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}

	var decoded ReactResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Prompt != resp.Prompt {
		t.Errorf("prompt = %q, want %q", decoded.Prompt, resp.Prompt)
	}
	if decoded.ArtistID != resp.ArtistID {
		t.Errorf("artist_id = %q, want %q", decoded.ArtistID, resp.ArtistID)
	}
	if decoded.ElapsedMs != 42 {
		t.Errorf("elapsed_ms = %d, want 42", decoded.ElapsedMs)
	}
}

func TestImageConcurrentAccess(t *testing.T) {
	srv := newTestServer()

	// Simulate concurrent read/write
	done := make(chan bool, 2)

	go func() {
		for i := 0; i < 100; i++ {
			srv.imagesMu.Lock()
			srv.images["test"] = []byte{0xFF}
			srv.imagesMu.Unlock()
		}
		done <- true
	}()

	go func() {
		for i := 0; i < 100; i++ {
			srv.imagesMu.RLock()
			_ = srv.images["test"]
			srv.imagesMu.RUnlock()
		}
		done <- true
	}()

	<-done
	<-done
}

func TestHandleUIServesEmbeddedHTML(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	srv.handleUI(w, req)

	body := w.Body.String()

	// Should contain key UI elements
	expected := []string{
		"<!DOCTYPE html>",
		"yent.yo",
		"chat-container",
		"send-btn",
		"/react",
		"/health",
	}

	for _, exp := range expected {
		if !strings.Contains(body, exp) {
			t.Errorf("UI HTML missing expected content: %q", exp)
		}
	}
}

func TestTryGenerateImageNoModel(t *testing.T) {
	srv := newTestServer()
	srv.sdModelDir = "/nonexistent/path"

	result := srv.tryGenerateImage("test prompt")
	if result != nil {
		t.Error("should return nil when SD model not available")
	}
}

// Test that all mux routes are registered correctly
func TestServerRoutes(t *testing.T) {
	mux := http.NewServeMux()
	srv := newTestServer()

	mux.HandleFunc("/", srv.handleUI)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/react", srv.handleReact)
	mux.HandleFunc("/image/", srv.handleImage)

	routes := []struct {
		path   string
		method string
		want   int
	}{
		{"/", "GET", 200},
		{"/nonexistent", "GET", 404},
		{"/react", "GET", 405},
		{"/react", "POST", 400}, // empty body
		{"/image/missing", "GET", 404},
	}

	for _, r := range routes {
		req := httptest.NewRequest(r.method, r.path, nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != r.want {
			t.Errorf("%s %s: got %d, want %d", r.method, r.path, w.Code, r.want)
		}
	}
}
