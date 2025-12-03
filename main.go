package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

type SpeculativeConfig struct {
	Model                string
	Method               string
	NumSpeculativeTokens int
	SpeculativeTokenTree string
}

type GPUSlice []string

func (g *GPUSlice) String() string {
	return fmt.Sprint(*g)
}

func (g *GPUSlice) Set(value string) error {
	parts := strings.Split(value, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		*g = append(*g, part)
	}
	return nil
}

func (s SpeculativeConfig) String() string {
	return fmt.Sprintf("{\"model\": %q, \"method\": %q, \"num_speculative_tokens\": %d \"speculative_token_tree\": %q}", s.Model, s.Method, s.NumSpeculativeTokens, s.SpeculativeTokenTree)
}

func main() {
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	full := fs.Bool("full", false, "run bench on all valid tree combinations")
	var gpus GPUSlice
	fs.Var(&gpus, "gpus", "comma-separated list of GPU devices to use")
	enableCudagraph := fs.Bool("cudagraph", true, "enable cudagraph")
	width := fs.Int("width", 3, "tree width")
	depth := fs.Int("depth", 3, "tree depth")
	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
	if *full {
		if err := fullRun(gpus, os.Environ()); err != nil {
			log.Fatal(err)
		}
	} else {
		if err := run(*enableCudagraph, *width, *depth, os.Environ()); err != nil {
			log.Fatal(err)
		}
	}
}

type work struct {
	enableCudagraph bool
	width           int
	depth           int
}

func fullRun(gpus []string, environ []string) error {
	ch := generateWork()
	done := startWorkers(ch, gpus, environ)
	var errs []error
	for err := range done {
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func startWorkers(ch <-chan work, gpus []string, environ []string) <-chan error {
	c := make(chan error, 1)
	var wg errgroup.Group
	for _, gpu := range gpus {
		wg.Go(func() error {
			gpu := "CUDA_VISIBLE_DEVICES=" + gpu
			for w := range ch {
				if err := run(w.enableCudagraph, w.width, w.depth, append(environ, gpu)); err != nil {
					c <- fmt.Errorf("failed to run(%v, %v, %v): %v", w.enableCudagraph, w.width, w.depth, err)
				}
			}
			return nil
		})
	}
	go func() {
		if err := wg.Wait(); err != nil {
			c <- fmt.Errorf("failed to wait for workers: %v", err)
		}
		close(c)
	}()
	return c
}

func generateWork() <-chan work {
	out := make(chan work, 1)
	go func() {
		for cg := range 2 {
			for d := 1; d <= 32; d++ {
				maxWidth := 64 / d
				if d*maxWidth >= 64 {
					maxWidth--
				}
				for w := 2; w <= maxWidth; w++ {
					enableCudagraph := cg == 0
					out <- work{
						enableCudagraph: enableCudagraph,
						width:           w,
						depth:           d,
					}
				}
			}
		}
		close(out)
	}()
	return out
}

func run(enableCudagraph bool, width, depth int, env []string) error {
	myenv := make([]string, len(env))
	copy(myenv, env)
	myenv = append(myenv, "VLLM_USE_V1=1")

	serveArgs := strings.Split("vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --tensor-parallel-size=1 --max-num-seqs=64 --max-model-len=32768 --no-enable-prefix-caching", " ")
	if !enableCudagraph {
		myenv = append(myenv, "CUDA_LAUNCH_BLOCKING=1")
		serveArgs = append(serveArgs, "--enforce-eager")
	}
	tree := makeTreeString(width, depth)
	specConfig := SpeculativeConfig{
		Model:                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
		Method:               "eagle",
		NumSpeculativeTokens: width * depth,
		SpeculativeTokenTree: tree,
	}
	serveArgs = append(serveArgs, specConfig.String())

	benchArgs := strings.Split("vllm bench serve --model=meta-llama/Llama-3.1-8B-Instruct --tokenizer=meta-llama/Llama-3.1-8B-Instruct --dataset-name=hf --dataset-path=philschmid/mt-bench --ignore-eos --request-rate=inf --max-concurrency=1 --num-prompts=80", " ")

	serveBuf := bytes.Buffer{}
	serveCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var wg errgroup.Group

	wg.Go(func() error {
		serveCmd := exec.CommandContext(serveCtx, serveArgs[0], serveArgs[1:]...)
		serveCmd.Env = myenv
		serveCmd.Stdout = &serveBuf
		serveCmd.Stderr = &serveBuf
		serveCmd.WaitDelay = 10 * time.Second
		if err := serveCmd.Run(); err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			} else if errors.Is(err, os.ErrProcessDone) {
				return nil
			}
			return fmt.Errorf("serve error: %w", err)
		}
		return nil
	})

	benchBuf := bytes.Buffer{}
	wg.Go(func() error {
		defer cancel()
		benchCmd := exec.Command(benchArgs[0], benchArgs[1:]...)
		benchCmd.Env = myenv
		benchCmd.Stdout = &benchBuf
		benchCmd.Stderr = &benchBuf

		if err := benchCmd.Run(); err != nil {
			return fmt.Errorf("failed to run bench command: %w", err)
		}
		return nil
	})
	if err := wg.Wait(); err != nil {
		log.Println(err)
	}

	cudagraphLabel := "cudagraph"
	if !enableCudagraph {
		cudagraphLabel = "no-cudagraph"
	}
	outputPath := fmt.Sprintf("vllm_server.%dx%d.%s.log", width, depth, cudagraphLabel)
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)

	scanner := bufio.NewScanner(&benchBuf)
	shouldWrite := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "Serving Benchmark Result") {
			shouldWrite = true
		}
		if shouldWrite {
			if _, err := fmt.Fprintln(w, line); err != nil {
				log.Printf("failed to write bench result: %v", err)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("failed to scan bench result: %v", err)
	}
	if err := w.Flush(); err != nil {
		log.Printf("failed to flush bench result: %v", err)
	}
	scanner = bufio.NewScanner(&serveBuf)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, "Avg prompt throughput") || strings.Contains(line, "SpecDecoding metrics") {
			if _, err := fmt.Fprintln(w, line); err != nil {
				log.Printf("failed to write serve result: %v", err)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("failed to scan serve result: %v", err)
	}
	if err := w.Flush(); err != nil {
		log.Printf("failed to flush serve result: %v", err)
	}
	return f.Close()
}

func makeTreeString(width int, depth int) string {
	temp := []string{}
	for i := 1; i < depth; i++ {
		for j := 1; j < width; j++ {
			temp = append(temp, fmt.Sprintf("(%v", width)+strings.Repeat(", 0", depth-1)+")")
		}
	}
	out := strings.Join(temp, ",")
	out = "[" + out + "]"
	return out
}
