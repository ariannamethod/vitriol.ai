package main

// dual_yent.go — Two Yents working in parallel
//
// Artist: generates visual prompt (oppositional reaction) → feeds to diffusion
// Commentator: mocks the user in real-time while image generates
//
// Both models loaded simultaneously (micro-yent + nano-yent, ~160MB total)
// Roles alternate or are assigned randomly per interaction.

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"
)

// DualYent orchestrates two prompt generators
type DualYent struct {
	A    *PromptGenerator // first model
	B    *PromptGenerator // second model
	rng  *rand.Rand
	turn int // for alternating roles
}

// NewDualYent loads two models
func NewDualYent(pathA, pathB string) (*DualYent, error) {
	fmt.Fprintf(os.Stderr, "[dual] loading model A: %s\n", pathA)
	a, err := NewPromptGenerator(pathA)
	if err != nil {
		return nil, fmt.Errorf("model A: %w", err)
	}

	fmt.Fprintf(os.Stderr, "[dual] loading model B: %s\n", pathB)
	b, err := NewPromptGenerator(pathB)
	if err != nil {
		return nil, fmt.Errorf("model B: %w", err)
	}

	fmt.Fprintf(os.Stderr, "[dual] both models loaded\n")

	return &DualYent{
		A:   a,
		B:   b,
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// DualResult holds outputs from both yents
type DualResult struct {
	Prompt    string // artist's visual prompt (for diffusion)
	YentWords string // artist's words (for ASCII overlay)
	Roast     string // commentator's verbal mockery
	ArtistID  string // which model was artist ("A" or "B")
}

// React runs both yents in parallel on user input
func (dy *DualYent) React(userInput string, maxTokens int, temperature float32) DualResult {
	// Alternate roles each turn
	dy.turn++
	var artist, commentator *PromptGenerator
	var artistID string
	if dy.turn%2 == 0 {
		artist, commentator = dy.A, dy.B
		artistID = "A"
	} else {
		artist, commentator = dy.B, dy.A
		artistID = "B"
	}

	fmt.Fprintf(os.Stderr, "[dual] turn=%d artist=%s\n", dy.turn, artistID)

	var prompt, roast string
	var wg sync.WaitGroup
	wg.Add(2)

	// Artist: generate visual prompt
	go func() {
		defer wg.Done()
		prompt = artist.React(userInput, maxTokens, temperature)
	}()

	// Commentator: roast the user (stream to stderr for now)
	go func() {
		defer wg.Done()
		roast = commentator.Roast(userInput, 50, temperature+0.2)
	}()

	wg.Wait()

	// Extract yent words (before style suffix) for ASCII overlay
	yentWords := prompt
	for _, sep := range []string{", oil painting", ", abstract ", ", dark symbolic",
		", street art", ", surreal", ", Soviet poster", ", Picasso",
		", social realism", ", propaganda", ", caricature"} {
		if idx := strings.Index(yentWords, sep); idx >= 0 {
			yentWords = yentWords[:idx]
		}
	}

	return DualResult{
		Prompt:    prompt,
		YentWords: yentWords,
		Roast:     roast,
		ArtistID:  artistID,
	}
}

// StreamCommentary prints the commentator's roast with typing effect
func StreamCommentary(roast string) {
	fmt.Fprintf(os.Stderr, "\n")
	words := strings.Fields(roast)
	for i, w := range words {
		if i > 0 {
			fmt.Fprintf(os.Stderr, " ")
		}
		fmt.Fprintf(os.Stderr, "%s", w)
		// Typing effect: variable delay
		delay := 30 + rand.Intn(70) // 30-100ms per word
		time.Sleep(time.Duration(delay) * time.Millisecond)
	}
	fmt.Fprintf(os.Stderr, "\n\n")
}

// Free releases both models
func (dy *DualYent) Free() {
	if dy.A != nil {
		dy.A.Free()
	}
	if dy.B != nil {
		dy.B.Free()
	}
}
