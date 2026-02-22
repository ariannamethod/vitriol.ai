package main

// sketch.go — ASCII sketch animation: Yent's "creative process"
//
// Before the final image, Yent shows 2-3 "draft" sketches in ASCII,
// erasing each one with ANSI escape codes. The commentator mocks
// each attempt. Final image replaces the last sketch.
//
// Think of it as a loading screen with personality.

import (
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
)

// ASCII character sets — from lightest to darkest
var sketchChars = []byte(" .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")

// SketchConfig controls the sketch animation
type SketchConfig struct {
	Width       int           // sketch width in chars
	Height      int           // sketch height in chars
	NumDrafts   int           // how many "attempts" before final
	DraftDelay  time.Duration // how long each draft stays visible
	EraseDelay  time.Duration // pause between erase and next draft
	UseComments bool          // commentator comments on each draft
}

// DefaultSketchConfig returns sensible defaults
func DefaultSketchConfig() SketchConfig {
	return SketchConfig{
		Width:       50,
		Height:      15,
		NumDrafts:   3,
		DraftDelay:  800 * time.Millisecond,
		EraseDelay:  300 * time.Millisecond,
		UseComments: true,
	}
}

// SketchAnimation runs the "creative process" animation to stderr
func SketchAnimation(cfg SketchConfig, prompt string, rng *rand.Rand) {
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Comments Yent makes about each attempt
	comments := [][]string{
		{
			"[yent] hmm no...",
			"[yent] what is this garbage...",
			"[yent] are you kidding me...",
			"[yent] let me try again...",
		},
		{
			"[yent] ...closer",
			"[yent] getting somewhere maybe",
			"[yent] not terrible, still bad",
			"[yent] hmm...",
		},
		{
			"[yent] fine. this will do.",
			"[yent] good enough for you",
			"[yent] here, take it",
			"[yent] whatever",
		},
	}

	// Extract seed words from prompt for biasing the sketch
	words := strings.Fields(strings.ToLower(prompt))

	for draft := 0; draft < cfg.NumDrafts; draft++ {
		// Comment on previous attempt
		if cfg.UseComments && draft < len(comments) {
			comment := comments[draft][rng.Intn(len(comments[draft]))]
			fmt.Fprintf(os.Stderr, "\033[2m%s\033[0m\n", comment) // dim text
			time.Sleep(200 * time.Millisecond)
		}

		// Draw the box
		fmt.Fprintf(os.Stderr, "\u250c%s\u2510\n", strings.Repeat("\u2500", cfg.Width))

		// Generate sketch content
		for y := 0; y < cfg.Height; y++ {
			fmt.Fprintf(os.Stderr, "\u2502")
			line := generateSketchLine(cfg.Width, draft, y, cfg.Height, words, rng)
			fmt.Fprintf(os.Stderr, "%s", line)
			fmt.Fprintf(os.Stderr, "\u2502\n")

			// Progressive reveal effect: slight delay per line
			if draft == cfg.NumDrafts-1 {
				time.Sleep(20 * time.Millisecond)
			}
		}

		fmt.Fprintf(os.Stderr, "\u2514%s\u2518\n", strings.Repeat("\u2500", cfg.Width))

		// Hold the draft
		time.Sleep(cfg.DraftDelay)

		// Erase if not the last draft
		if draft < cfg.NumDrafts-1 {
			// Move cursor up and clear lines (box + content + comment)
			lines := cfg.Height + 3 // top border + content + bottom border + comment
			for i := 0; i < lines; i++ {
				fmt.Fprintf(os.Stderr, "\033[A\033[2K") // up + clear
			}
			time.Sleep(cfg.EraseDelay)
		}
	}
}

// generateSketchLine creates one line of ASCII sketch
func generateSketchLine(width, draft, y, height int, words []string, rng *rand.Rand) string {
	buf := make([]byte, width)

	switch draft {
	case 0:
		// First draft: sparse, mostly noise
		for x := 0; x < width; x++ {
			if rng.Float32() < 0.15 {
				buf[x] = sketchChars[rng.Intn(len(sketchChars)/3)] // light chars only
			} else {
				buf[x] = ' '
			}
		}

	case 1:
		// Second draft: some structure emerging (diagonal/radial patterns)
		cx, cy := width/2, height/2
		for x := 0; x < width; x++ {
			dx := float32(x-cx) / float32(width)
			dy := float32(y-cy) / float32(height)
			dist := dx*dx + dy*dy

			if dist < 0.15 && rng.Float32() < 0.6 {
				idx := int(dist*float32(len(sketchChars))) + rng.Intn(10)
				if idx >= len(sketchChars) {
					idx = len(sketchChars) - 1
				}
				buf[x] = sketchChars[idx]
			} else if rng.Float32() < 0.08 {
				buf[x] = sketchChars[rng.Intn(len(sketchChars)/2)]
			} else {
				buf[x] = ' '
			}
		}

		// Bleed some prompt words through
		if len(words) > 0 && y == height/2 {
			word := words[rng.Intn(len(words))]
			pos := rng.Intn(width - len(word) - 2)
			if pos >= 0 && pos+len(word) < width {
				for i, ch := range word {
					if rng.Float32() < 0.7 { // partial reveal
						buf[pos+i] = byte(ch)
					}
				}
			}
		}

	default:
		// Final draft: denser, more defined shapes
		cx, cy := width/2, height/2
		for x := 0; x < width; x++ {
			dx := float32(x-cx) / float32(width)
			dy := float32(y-cy) / float32(height)
			dist := dx*dx + dy*dy

			if dist < 0.2 {
				intensity := 1.0 - dist/0.2
				idx := int(intensity * float32(len(sketchChars)-1))
				idx += rng.Intn(5) - 2 // jitter
				if idx < 0 {
					idx = 0
				}
				if idx >= len(sketchChars) {
					idx = len(sketchChars) - 1
				}
				buf[x] = sketchChars[idx]
			} else if rng.Float32() < 0.12 {
				buf[x] = sketchChars[rng.Intn(len(sketchChars)/3)]
			} else {
				buf[x] = ' '
			}
		}

		// More words bleeding through
		if len(words) > 0 && (y == height/3 || y == height*2/3) {
			word := words[rng.Intn(len(words))]
			pos := rng.Intn(width - len(word) - 2)
			if pos >= 0 && pos+len(word) < width {
				for i, ch := range word {
					buf[pos+i] = byte(ch)
				}
			}
		}
	}

	return string(buf)
}

// SketchTransition shows a brief "thinking" animation between sketch and final image
func SketchTransition(rng *rand.Rand) {
	frames := []string{
		"[yent] rendering",
		"[yent] rendering.",
		"[yent] rendering..",
		"[yent] rendering...",
	}

	for i := 0; i < 8; i++ {
		fmt.Fprintf(os.Stderr, "\r\033[2m%s\033[0m", frames[i%len(frames)])
		time.Sleep(250 * time.Millisecond)
	}
	fmt.Fprintf(os.Stderr, "\r\033[2K") // clear line
}
