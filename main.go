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
	"strconv"
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

type TreeRange []int

func (i *TreeRange) String() string {
	return fmt.Sprint(*i)
}

func (i *TreeRange) Set(value string) error {
	if value == "" {
		*i = append(*i, 3) // default to 3
		return nil
	} else if strings.ToLower(value) == "all" {
		for j := 1; j <= 64; j++ {
			*i = append(*i, j)
		}
		return nil
	}
	parts := strings.Split(value, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		val, err := strconv.Atoi(part)
		if err != nil {
			return fmt.Errorf("could not parse int: %w", err)
		}
		*i = append(*i, val)
	}
	return nil
}

func main() {
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	full := fs.Bool("full", false, "run bench on all valid tree combinations")
	var gpus GPUSlice
	fs.Var(&gpus, "gpus", "comma-separated list of GPU devices")
	var widths TreeRange
	fs.Var(&widths, "width", "comma-separated list of tree widths")
	var depths TreeRange
	fs.Var(&depths, "depth", "comma-separated list of tree depths")
	draftModel := fs.String("draft_model", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "the model to use for drafting")
	targetModel := fs.String("target_model", "meta-llama/Llama-3.1-8B-Instruct", "the model to use for verification")
	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if *full {
		depths = depths[:0]
		widths = widths[:0]
		for i := 1; i <= 64; i++ {
			depths = append(depths, i)
			widths = append(widths, i)
		}
	}

	if err := fullRun(ctx, gpus, widths, depths, os.Environ(), *draftModel, *targetModel); err != nil {
		log.Fatal(err)
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

func fullRun(ctx context.Context, gpus []string, widths, depths []int, environ []string, draftModel, targetModel string) error {
	ch := generateWork(ctx, widths, depths)
	errsCh := startWorkers(ctx, ch, gpus, environ, draftModel, targetModel)
	var errs []error
	for err := range errsCh {
		if err != nil {
			log.Println(err)
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func generateWork(ctx context.Context, widths, depths []int) <-chan work {
	out := make(chan work, 1)
	go func() {
		defer close(out)
		for _, d := range depths {
			if d == 64 {
				continue
			}
			for _, w := range widths {
				if w == 1 || w*d >= 64 {
					continue
				}
				select {
				case <-ctx.Done():
					return
				case out <- work{
					enableCudagraph: true,
					width:           w,
					depth:           d,
				}:
				}
			}
		}
	}()
	return out
}

func startWorkers(ctx context.Context, ch <-chan work, gpus []string, environ []string, draftModel, targetModel string) <-chan error {
	c := make(chan error, 1)
	var wg errgroup.Group
	port := 9000
	for _, gpu := range gpus {
		wg.Go(func(port int) func() error {
			return func() error {
				gpu := "CUDA_VISIBLE_DEVICES=" + gpu
				for w := range ch {
					log.Printf("worker %v: enableCudagraph=%v, width=%v, depth=%v", port, w.enableCudagraph, w.width, w.depth)
					if err := run(ctx, w.enableCudagraph, w.width, w.depth, append(environ, gpu), port, draftModel, targetModel); err != nil {
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

func run(ctx context.Context, enableCudagraph bool, width, depth int, env []string, port int, draftModel, targetModel string) error {
	myenv := make([]string, len(env))
	copy(myenv, env)
	myenv = append(myenv, "VLLM_USE_V1=1")

	serveArgs := []string{
		"vllm",
		"serve",
		targetModel,
		"--disable-log-requests",
		"--tensor-parallel-size=1",
		"--max-num-seqs=64",
		"--max-model-len=32768",
		"--no-enable-prefix-caching",
		fmt.Sprintf("--port=%d", port),
	}
	if !enableCudagraph {
		myenv = append(myenv, "CUDA_LAUNCH_BLOCKING=1")
		serveArgs = append(serveArgs, "--enforce-eager")
	}
	tree := makeTreeString(width, depth)
	specConfig := SpeculativeConfig{
		Model:                draftModel,
		Method:               "eagle",
		NumSpeculativeTokens: width * depth,
		SpeculativeTokenTree: tree,
	}
	serveArgs = append(serveArgs, specConfig.String())

	benchArgs := []string{
		"vllm",
		"bench",
		"serve",
		"--model=" + targetModel,
		"--tokenizer=" + targetModel,
		"--dataset-name=hf",
		"--dataset-path=philschmid/mt-bench",
		"--ignore-eos",
		"--request-rate=inf",
		"--max-concurrency=1",
		"--num-prompts=80",
		fmt.Sprintf("--port=%d", port),
	}

	serveBuf := bytes.Buffer{}
	serveCtx, serveCancel := context.WithCancel(ctx)
	defer serveCancel()
	serveCmd := exec.CommandContext(serveCtx, serveArgs[0], serveArgs[1:]...)
	serveCmd.Env = myenv
	serveCmd.Stdout = &serveBuf
	serveCmd.Stderr = &serveBuf
	serveCmd.WaitDelay = 10 * time.Second
	serveCmd.Cancel = func() error {
		return serveCmd.Process.Signal(syscall.SIGTERM)
	}
	log.Println("starting serve cmd")
	if err := serveCmd.Start(); err != nil {
		return fmt.Errorf("failed to start server command: %w", err)
	}

	benchBuf := bytes.Buffer{}
	benchCtx, benchCancel := context.WithCancel(ctx)
	defer benchCancel()
	benchCmd := exec.CommandContext(benchCtx, benchArgs[0], benchArgs[1:]...)
	benchCmd.Env = myenv
	benchCmd.Stdout = &benchBuf
	benchCmd.Stderr = &benchBuf
	benchCmd.WaitDelay = 10 * time.Second
	benchCmd.Cancel = func() error {
			return benchCmd.Process.Signal(syscall.SIGTERM)
	}
	log.Println("starting bench cmd")
	if err := benchCmd.Start(); err != nil {
		return fmt.Errorf("failed to start bench command: %w", err)
	}

	serveDone := make(chan error)
	go func() {
		err := serveCmd.Wait()
		if err != nil {
			log.Printf("serve failed: %v", err)
		}
		log.Println("serve done")
		benchCancel()
		close(serveDone)
	}()
	benchDone := make(chan error)
	go func() {
		err := benchCmd.Wait()
		if err != nil {
			log.Printf("bench failed: %v", err)
		}
		log.Println("bench done")
		serveCancel()
		close(benchDone)
	}()

	doneCtx, doneCancel := context.WithCancel(ctx)
	defer doneCancel()
	done := make(chan struct{})
	go func() {
		<-doneCtx.Done()
		if done != nil {
			close(done)
		}
	}()

	for {
		select {
		case <-done:
			log.Println("stopping workers")
			err := benchCmd.Process.Signal(syscall.SIGTERM)
			if err != nil {
				log.Printf("failed to send SIGTERM to benchCmd: %v", err)
			}
			err = serveCmd.Process.Signal(syscall.SIGTERM)
			if err != nil {
				log.Printf("failed to send SIGTERM to serveCmd: %v", err)
			}
			done = nil
		case err := <-benchDone:
			if err != nil {
				log.Printf("bench failed: %v", err)
			}
			benchDone = nil
			err = serveCmd.Process.Signal(syscall.SIGTERM)
			if err != nil {
				log.Printf("failed to send SIGTERM to serveCmd: %v", err)
			}
		case err := <-serveDone:
			if err != nil {
				log.Printf("serve failed: %v", err)
			}
			serveDone = nil
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
		return fmt.Errorf("failed to create output file: %w", err)
	}
	w := bufio.NewWriter(f)
	_, err = w.ReadFrom(&benchBuf)
	if err != nil {
		log.Printf("failed to write bench result: %v", err)
	}
	_, err = w.WriteString("\n\n")
	if err != nil {
		log.Printf("failed to write newlines: %v", err)
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
	err = f.Close()
	if err != nil {
		return fmt.Errorf("failed to close file: %w", err)
	}
	return nil
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
