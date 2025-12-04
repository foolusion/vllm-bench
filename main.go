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
	"os/signal"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"
)

type SpeculativeConfig struct {
	Model                string
	Method               string
	NumSpeculativeTokens int
	SpeculativeTokenTree string
}

func (s SpeculativeConfig) String() string {
	return fmt.Sprintf("--speculative-config={\"model\": %q, \"method\": %q, \"num_speculative_tokens\": %d, \"speculative_token_tree\": %q}", s.Model, s.Method, s.NumSpeculativeTokens, s.SpeculativeTokenTree)
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

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if *full {
		if err := fullRun(ctx, gpus, os.Environ()); err != nil {
			log.Fatal(err)
		}
	} else {
		if err := run(ctx, *enableCudagraph, *width, *depth, os.Environ(), 9000); err != nil {
			log.Fatal(err)
		}
	}
	select {
	case <-ctx.Done():
		stop()
	default:
	}
}

type work struct {
	enableCudagraph bool
	width           int
	depth           int
}

func fullRun(ctx context.Context, gpus []string, environ []string) error {
	ch := generateWork(ctx)
	done := startWorkers(ctx, ch, gpus, environ)
	var errs []error
	for err := range done {
		if err != nil {
			log.Println(err)
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func generateWork(ctx context.Context) <-chan work {
	out := make(chan work, 1)
	go func() {
		defer close(out)
		for cg := range 2 {
			for d := 1; d <= 32; d++ {
				maxWidth := 64 / d
				if d*maxWidth >= 64 {
					maxWidth--
				}
				for w := 2; w <= maxWidth; w++ {
					enableCudagraph := cg == 0
					select {
					case <-ctx.Done():
						return
					case out <- work{
						enableCudagraph: enableCudagraph,
						width:           w,
						depth:           d,
					}:
					}
				}
			}
		}
	}()
	return out
}

func startWorkers(ctx context.Context, ch <-chan work, gpus []string, environ []string) <-chan error {
	c := make(chan error, 1)
	var wg errgroup.Group
	port := 9000
	for _, gpu := range gpus {
		wg.Go(func(port int) func() error {
			return func() error {
				gpu := "CUDA_VISIBLE_DEVICES=" + gpu
				for w := range ch {
					log.Printf("worker %v: enableCudagraph=%v, width=%v, depth=%v", port, w.enableCudagraph, w.width, w.depth)
					if err := run(ctx, w.enableCudagraph, w.width, w.depth, append(environ, gpu), port); err != nil {
						c <- fmt.Errorf("failed to run(%v, %v, %v): %v", w.enableCudagraph, w.width, w.depth, err)
					}
				}
				return nil
			}
		}(port))
		port++
	}
	go func() {
		if err := wg.Wait(); err != nil {
			c <- fmt.Errorf("failed to wait for workers: %v", err)
		}
		close(c)
	}()
	return c
}

func run(ctx context.Context, enableCudagraph bool, width, depth int, env []string, port int) error {
	myenv := make([]string, len(env))
	copy(myenv, env)
	myenv = append(myenv, "VLLM_USE_V1=1")

	serveArgs := strings.Split("vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --tensor-parallel-size=1 --max-num-seqs=64 --max-model-len=32768 --no-enable-prefix-caching", " ")
	if !enableCudagraph {
		myenv = append(myenv, "CUDA_LAUNCH_BLOCKING=1")
		serveArgs = append(serveArgs, "--enforce-eager")
	}
	serveArgs = append(serveArgs, fmt.Sprintf("--port=%d", port))
	tree := makeTreeString(width, depth)
	specConfig := SpeculativeConfig{
		Model:                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
		Method:               "eagle",
		NumSpeculativeTokens: width * depth,
		SpeculativeTokenTree: tree,
	}
	serveArgs = append(serveArgs, specConfig.String())

	benchArgs := strings.Split("vllm bench serve --model=meta-llama/Llama-3.1-8B-Instruct --tokenizer=meta-llama/Llama-3.1-8B-Instruct --dataset-name=hf --dataset-path=philschmid/mt-bench --ignore-eos --request-rate=inf --max-concurrency=1 --num-prompts=80", " ")
	benchArgs = append(benchArgs, fmt.Sprintf("--port=%d", port))

	serveBuf := bytes.Buffer{}
	serveCmd := exec.Command(serveArgs[0], serveArgs[1:]...)
	serveCmd.Env = myenv
	serveCmd.Stdout = &serveBuf
	serveCmd.Stderr = &serveBuf
	serveCmd.WaitDelay = 10 * time.Second
	log.Println("starting serve cmd")
	if err := serveCmd.Start(); err != nil {
		return fmt.Errorf("failed to start server command: %w", err)
	}

	benchBuf := bytes.Buffer{}
	benchCmd := exec.Command(benchArgs[0], benchArgs[1:]...)
	benchCmd.Env = myenv
	benchCmd.Stdout = &benchBuf
	benchCmd.Stderr = &benchBuf
	benchCmd.WaitDelay = 10 * time.Second
	log.Println("starting bench cmd")
	if err := benchCmd.Start(); err != nil {
		return fmt.Errorf("failed to start bench command: %w", err)
	}

	serveDone := make(chan error)
	go func() {
		serveDone <- serveCmd.Wait()
		log.Println("serve done")
		close(serveDone)
	}()
	benchDone := make(chan error)
	go func() {
		benchDone <- benchCmd.Wait()
		log.Println("bench done")
		close(benchDone)
	}()

	for {
		select {
		case <-ctx.Done():
			log.Println("stopping workers")
			benchCmd.Process.Signal(syscall.SIGTERM)
			serveCmd.Process.Signal(syscall.SIGTERM)
		case err := <-serveDone:
			if err != nil {
				log.Printf("serve failed: %v", err)
			}
			serveDone = nil
		case err := <-benchDone:
			if err != nil {
				log.Printf("bench failed: %v", err)
			}
			benchDone = nil
			serveCmd.Process.Signal(syscall.SIGTERM)
		}
		if serveDone == nil && benchDone == nil {
			break
		}
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
	_, err = w.ReadFrom(&benchBuf)
	if err != nil {
		log.Printf("failed to write bench result: %v", err)
	}
	_, err = w.WriteString("\n\n")
	if err != nil {
		log.Printf("failed to write string: %v", err)
	}
	err = w.Flush()
	if err != nil {
		log.Printf("failed to flush bench result: %v", err)
	}
	_, err = w.ReadFrom(&serveBuf)
	if err != nil {
		log.Printf("failed to write serve result: %v", err)
	}
	err = w.Flush()
	if err != nil {
		log.Printf("failed to flush serve result: %v", err)
	}
	return f.Close()
}

func makeTreeString(width int, depth int) string {
	var temp []string
	zeros := strings.Split(strings.Repeat("0", depth), "")
	for i := 0; i < depth; i++ {
		for j := 0; j < width; j++ {
			temp = append(temp, fmt.Sprintf("(%v,", j)+strings.Join(zeros[:i], ",")+")")
		}
	}
	out := strings.Join(temp, ",")
	out = "[" + out + "]"
	return out
}
